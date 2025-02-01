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
from ollama import Client
import aiofiles
from enum import Enum

import prompt_helper

# FastAPI 앱 인스턴스 생성
scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(webserver_app: FastAPI):
    """ FastAPI의 lifespan 이벤트 핸들러 """
    await create_directories([TEXT_FILE_PATH, OUTPUT_FILE_PATH])
    scheduler.start()
    print("📌 Scheduler Started")
    yield
    scheduler.shutdown()
    print("📌 Scheduler Shutdown")

webserver_app = FastAPI(lifespan=lifespan)

# 템플릿 및 정적 파일 설정
BASE_DIR = Path(__file__).parent
local_static_folder_path = Path(BASE_DIR) / "static"
local_template_folder_path = Path(BASE_DIR) / "templates"

templates = Jinja2Templates(directory=local_template_folder_path)
webserver_app.mount("/static", StaticFiles(directory=local_static_folder_path), name="static")

print(f"TEST -> : {local_static_folder_path}, // {local_template_folder_path}")

# 현재 진행 중인 파일 개수 상태 변수
processing_files = 0

class LogType(Enum):
    ERROR = 0
    WARN = 1
    INFO = 2

class SimpleLogger:
    @staticmethod
    def Log(msg : str, log_type : LogType):

        prefix = ""
        if log_type == LogType.ERROR:
            prefix = "[ERROR]"
        elif log_type == LogType.WARN:
            prefix = "[WARN]"
        elif log_type == LogType.INFO:
            prefix = "[INFO]"
            
        print(f"{prefix} {msg}")
    

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
                SimpleLogger.Log(f"Skipping unsupported file: {file}", LogType.WARN)
                continue
            
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            if not content.strip():
                continue
            try:
                result = ollama_client.generate(OLLAMA_MODEL, prompt_helper.OLLAMA_PROMPT.format(content=content))
                output_path = OUTPUT_FILE_PATH / f"{file}.out"
                async with aiofiles.open(output_path, "w", encoding="utf-8") as out_f:
                    await out_f.write(result)
                
                await db.execute("INSERT INTO processed_files (filename) VALUES (?)", (file,))
                await db.commit()
            except Exception as e:
                SimpleLogger.Log(f"Error processing {file}: {e}", LogType.ERROR)
    processing_files = 0  # 모든 작업이 완료되면 0으로 설정

@webserver_app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@webserver_app.get("/api/config/reload")
async def reload_config():
    global config, TEXT_FILE_PATH, OUTPUT_FILE_PATH, OLLAMA_MODEL
    config = load_config()
    TEXT_FILE_PATH = Path(__file__).parent / config["text_file_path"]
    OUTPUT_FILE_PATH = Path(__file__).parent / config["output_file_path"]
    OLLAMA_MODEL = config["ollama_model"]

    SimpleLogger.Log(f"input file path : {TEXT_FILE_PATH}, output file path : {OUTPUT_FILE_PATH}, ollama_model : {OLLAMA_MODEL}", LogType.INFO)
    SimpleLogger.Log(f"Config reloaded successfully!", LogType.INFO)

    return {"message": "Config reloaded successfully"}

@webserver_app.post("/api/process")
async def trigger_processing(background_tasks: BackgroundTasks):
    background_tasks.add_task(process_files)
    SimpleLogger.Log(f"Processing started in the background", LogType.INFO)
    return {"message": "Processing started in the background"}

@webserver_app.get("/api/status")
async def get_status():
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM processed_files")
        completed_files = await cursor.fetchone()
    SimpleLogger.Log(f"Processing.  files : {process_files}, completed : {completed_files}", LogType.INFO)
    return {"processing": processing_files, "completed": completed_files[0]}

@webserver_app.get("/debug/static-files")
def debug_static_files():
    static_path = Path(BASE_DIR / "static")
    if static_path.exists():
        files = os.listdir(static_path)
        return {"static_exists": True, "files": files}
    
    return {"static_exists": False, "message": "Static folder not found"}

@webserver_app.middleware("http")
async def log_requests(request: Request, call_next):
    SimpleLogger.Log(f"Request path: {request.url.path}", LogType.INFO)  # 요청된 경로 출력
    response = await call_next(request)
    return response

if __name__ == "__main__":
    uvicorn.run("local_llm_analyzer:webserver_app", host="0.0.0.0", port=8000, reload=True)
    #uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
