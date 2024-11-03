import os
import json
import asyncio
import aiofiles
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Ollama 라이브러리 사용
import ollama  
from ollama import Client

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# HTML 템플릿 경로 설정
templates = Jinja2Templates(directory="templates")

# 정적 파일 경로 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

# 설정 파일 로드
with open("config.json") as f:
    config = json.load(f)

# 주기적으로 체크할 파일 경로 및 설정값들
TEXT_FILE_PATH = config["text_file_path"]
OUTPUT_FILE_PATH = config["output_file_path"]
OLLAMA_MODEL = config["ollama_model"]
CHECK_DAYS = config["check_days"]
CHECK_TIME = config["check_time"]
DB_PATH = config["db_path"]
OLLAMA_HOST_URL = config["ollama_host_url"]

# SQLite 데이터베이스 연결
db_connection = sqlite3.connect(DB_PATH)
db_cursor = db_connection.cursor()

# ollama client
ollama_client = Client(host=OLLAMA_HOST_URL)

# 이미 처리된 파일 테이블 생성
db_cursor.execute("""
CREATE TABLE IF NOT EXISTS processed_files (
    filename TEXT PRIMARY KEY
)
""")
db_connection.commit()

# 파일이 이미 처리되었는지 확인하는 함수
def is_file_processed(filename):
    db_cursor.execute("SELECT filename FROM processed_files WHERE filename = ?", (filename,))
    return db_cursor.fetchone() is not None

# 파일 처리를 위한 비동기 작업
async def process_files():
    files = await get_files_in_directory(TEXT_FILE_PATH)
    for file in files:
        if not is_file_processed(file):
            text_content = await load_text_file(file)
            response = await query_ollama_model(text_content)
            await save_output(file, response)
            db_cursor.execute("INSERT INTO processed_files (filename) VALUES (?)", (file,))
            db_connection.commit()

# 주기적 작업 설정
scheduler = AsyncIOScheduler()
scheduler.add_job(process_files, 'cron', day_of_week=','.join(CHECK_DAYS), hour=CHECK_TIME.split(':')[0], minute=CHECK_TIME.split(':')[1])
scheduler.start()

# 비동기적으로 디렉토리에서 파일 목록을 가져오기
async def get_files_in_directory(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# 비동기적으로 텍스트 파일 로드하기
async def load_text_file(filepath):
    async with aiofiles.open(filepath, 'r') as file:
        return await file.read()

async def query_ollama_model(in_text_context):
    # 프롬프트 엔지니어링을 통한 지시문 생성
    engineered_prompt = (
        f"Analyze the following text, focusing on identifying key issues, troubleshooting steps, "
        f"and summarizing any relevant code or error details:\n\n{in_text_context}\n\n"
        "Provide a concise summary of the analysis, highlighting critical information."
    )
    # Ollama 모델 호출
    response = ollama.generate(model=OLLAMA_MODEL, prompt=engineered_prompt)
    
    # 응답 텍스트 반환
    return response["text"]

# 비동기적으로 응답 텍스트를 저장하기
async def save_output(input_file, response_text):
    output_filename = os.path.basename(input_file).replace(".txt", "_response.txt")
    output_path = os.path.join(OUTPUT_FILE_PATH, output_filename)
    async with aiofiles.open(output_path, 'w') as file:
        await file.write(response_text)

# 웹 페이지 UI를 표시하는 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def read_root():
    files = await get_files_in_directory(OUTPUT_FILE_PATH)
    return templates.TemplateResponse("index.html", {"request": "request", "files": files})

# 개별 파일의 내용을 반환하는 엔드포인트
@app.get("/file/{filename}", response_class=HTMLResponse)
async def read_file(filename: str):
    file_path = os.path.join(OUTPUT_FILE_PATH, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    async with aiofiles.open(file_path, 'r') as file:
        content = await file.read()
    return f"<html><body><pre>{content}</pre></body></html>"

# 웹서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
