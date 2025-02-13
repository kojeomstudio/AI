import os
import json
import asyncio
import aiosqlite
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
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
############################################################################################
BASE_DIR = Path(__file__).parent
local_static_folder_path = Path(BASE_DIR) / "static"
local_template_folder_path = Path(BASE_DIR) / "templates"

templates = Jinja2Templates(directory=local_template_folder_path)
webserver_app.mount("/static", StaticFiles(directory=local_static_folder_path), name="static")
############################################################################################

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

global_config = load_config()

TEXT_FILE_PATH = Path(__file__).parent / global_config["text_file_path"]
OUTPUT_FILE_PATH = Path(__file__).parent / global_config["output_file_path"]
DB_PATH = Path(__file__).parent / global_config["db_path"]
OLLAMA_MODEL = global_config["ollama_model"]
CHECK_DAYS = global_config["check_days"]
CHECK_TIME = global_config["check_time"]
OLLAMA_HOST_URL = global_config["ollama_host_url"]

ollama_client = Client(host=OLLAMA_HOST_URL)

async def create_directories(directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

import asyncio
import os
import aiofiles
import aiosqlite
from pathlib import Path

# 전역 락 선언
file_processing_lock = asyncio.Lock()

async def process_files():
    global processing_files
    async with file_processing_lock:  # 여러 실행 방지
        processing_files = 0
        files = await asyncio.to_thread(os.listdir, TEXT_FILE_PATH)  # 블로킹 방지
        async with aiosqlite.connect(DB_PATH) as db:
            for file in files:
                file_path = TEXT_FILE_PATH / file
                if not file_path.suffix in [".txt", ".md"]:
                    SimpleLogger.Log(f"Skipping unsupported file: {file}", LogType.WARN)
                    continue

                async with db.execute("SELECT filename FROM processed_files WHERE filename = ?", (file,)) as cursor:
                    existing_file = await cursor.fetchone()
                    if existing_file:
                        SimpleLogger.Log(f"File {file} already processed, skipping.", LogType.INFO)
                        continue

                try:
                    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                        file_content = await f.read()
                        if not file_content.strip():
                            continue

                    #SimpleLogger.Log(f"Loaded file content: {file_content}", LogType.INFO)

                    # LLM 모델 실행 시 예외 핸들링 및 재시도 로직 추가
                    result = None
                    for _ in range(3):  # 최대 3회 재시도
                        try:
                            result = ollama_client.generate(
                                OLLAMA_MODEL,
                                prompt_helper.OLLAMA_PROMPT.format(content=file_content)
                            )
                            break  # 성공 시 루프 탈출
                        except Exception as e:
                            SimpleLogger.Log(f"Error generating LLM output, retrying: {e}", LogType.WARN)
                            await asyncio.sleep(1)  # 딜레이 추가 후 재시도

                    if not result:
                        SimpleLogger.Log(f"Failed to process file {file} after retries.", LogType.ERROR)
                        continue

                    output_path = OUTPUT_FILE_PATH / f"{file}.out"
                    async with aiofiles.open(output_path, "w", encoding="utf-8") as out_f:
                        await out_f.write(result)

                    await db.execute("INSERT INTO processed_files (filename) VALUES (?)", (file,))
                    await db.commit()

                except Exception as e:
                    SimpleLogger.Log(f"Error processing {file}: {e}", LogType.ERROR)

        processing_files = 0  # 모든 작업 완료 후 리셋

# 개선된 DB 초기화 함수
async def init_db():
    global db_connection
    if db_connection is None:  # 싱글톤 패턴 적용
        db_connection = await aiosqlite.connect(DB_PATH)

    async with db_connection:
        await db_connection.execute("""
        CREATE TABLE IF NOT EXISTS processed_files (
            filename TEXT PRIMARY KEY
        )
        """)
        await db_connection.commit()

@webserver_app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@webserver_app.get("/api/config/reload")
async def reload_config():
    global global_config, TEXT_FILE_PATH, OUTPUT_FILE_PATH, OLLAMA_MODEL
    global_config = load_config()
    TEXT_FILE_PATH = Path(__file__).parent / global_config["text_file_path"]
    OUTPUT_FILE_PATH = Path(__file__).parent / global_config["output_file_path"]
    OLLAMA_MODEL = global_config["ollama_model"]

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
    """ 현재 처리 중인 파일 목록 및 완료된 파일 개수 반환 """
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT COUNT(*) FROM processed_files") as cursor:
            completed_count = (await cursor.fetchone())[0]
    return JSONResponse({
        "processing": len(processing_files),
        "completed": completed_count,
        "processing_files": processing_files  # 진행 중인 파일 목록 포함
    })

@webserver_app.get("/api/files")
async def get_files():
    """ 현재 input/output 디렉토리의 파일 목록을 반환 """
    input_files = os.listdir(TEXT_FILE_PATH)
    output_files = os.listdir(OUTPUT_FILE_PATH)
    return JSONResponse({"input_files": input_files, "output_files": output_files})

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
