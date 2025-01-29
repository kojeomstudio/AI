import os
import json
import asyncio
import aiosqlite
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager
import ollama  
from ollama import Client
import aiofiles

# FastAPI 앱 인스턴스 생성
scheduler = AsyncIOScheduler()
app = FastAPI()

# 템플릿 및 정적 파일 설정
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")

# 현재 진행 중인 파일 개수 상태 변수
processing_files = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    """ FastAPI의 lifespan 이벤트 핸들러 """
    await create_directories([TEXT_FILE_PATH, OUTPUT_FILE_PATH])
    scheduler.start()
    print("📌 Scheduler Started")
    yield
    scheduler.shutdown()
    print("📌 Scheduler Shutdown")

app = FastAPI(lifespan=lifespan)

# 설정 파일 로드 (동적 로딩 지원)
def load_config():
    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        return json.load(f)

config = load_config()

TEXT_FILE_PATH = Path(__file__).parent / config["text_file_path"]
OUTPUT_FILE_PATH = Path(__file__).parent / config["output_file_path"]
DB_PATH = Path(__file__).parent / config["db_path"]
OLLAMA_MODEL = config["ollama_model"]
CHECK_DAYS = config["check_days"]
CHECK_TIME = config["check_time"]
OLLAMA_HOST_URL = config["ollama_host_url"]

ollama_client = Client(host=OLLAMA_HOST_URL)

async def create_directories(directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS processed_files (
            filename TEXT PRIMARY KEY
        )
        """)
        await db.commit()

async def process_files():
    global processing_files
    files = os.listdir(TEXT_FILE_PATH)
    processing_files = len(files)  # 현재 처리 중인 파일 개수 설정
    async with aiosqlite.connect(DB_PATH) as db:
        for file in files:
            file_path = TEXT_FILE_PATH / file
            if not file_path.suffix in [".txt", ".md"]:
                print(f"Skipping unsupported file: {file}")
                continue
            
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            if not content.strip():
                continue
            try:
                result = ollama_client.generate(OLLAMA_MODEL, content)
                output_path = OUTPUT_FILE_PATH / f"{file}.out"
                async with aiofiles.open(output_path, "w", encoding="utf-8") as out_f:
                    await out_f.write(result)
                
                await db.execute("INSERT INTO processed_files (filename) VALUES (?)", (file,))
                await db.commit()
            except Exception as e:
                print(f"Error processing {file}: {e}")
    processing_files = 0  # 모든 작업이 완료되면 0으로 설정

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/config/reload")
async def reload_config():
    global config, TEXT_FILE_PATH, OUTPUT_FILE_PATH, OLLAMA_MODEL
    config = load_config()
    TEXT_FILE_PATH = Path(__file__).parent / config["text_file_path"]
    OUTPUT_FILE_PATH = Path(__file__).parent / config["output_file_path"]
    OLLAMA_MODEL = config["ollama_model"]
    return {"message": "Config reloaded successfully"}

@app.post("/api/process")
async def trigger_processing(background_tasks: BackgroundTasks):
    background_tasks.add_task(process_files)
    return {"message": "Processing started in the background"}

@app.get("/api/status")
async def get_status():
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM processed_files")
        completed_files = await cursor.fetchone()
    return {"processing": processing_files, "completed": completed_files[0]}

@app.get("/debug/static-files")
def debug_static_files():
    static_path = Path(BASE_DIR / "static")
    if static_path.exists():
        files = os.listdir(static_path)
        return {"static_exists": True, "files": files}
    return {"static_exists": False, "message": "Static folder not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
