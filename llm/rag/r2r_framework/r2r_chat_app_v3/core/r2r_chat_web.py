from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from r2r import R2RClient
import json
import uvicorn
import asyncio
import os

######################################################################

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(BASE_DIR, "config.json")
static_dir = os.path.join(BASE_DIR, "static")

######################################################################

# config.json에서 설정 불러오기
with open(config_path, "r") as f:
    config = json.load(f)
r2r_server_url = config["r2r_server_url"]
host = config["host"]
port = config["port"]

client = R2RClient(r2r_server_url)

# Query 모델
class Query(BaseModel):
    question: str


# Static files 설정 (index.html 포함 폴더 지정)
app.mount('/static', StaticFiles(directory=static_dir), name="static")

# index.html 반환을 위한 라우팅
@app.get("/")
async def get_home():
    with open(f"{BASE_DIR}/index.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.post("/query")
async def query_r2r(query: Query):
    try:
        res = await client.rag(query.question)
        completion = res.get("results", {}).get("completion", {})
        completion_id = completion.get("id")
        completion_choices = completion.get("choices", [])

        response_data = {
            "completion_id": completion_id,
            "responses": [
                {
                    "finish_reason": choice.get("finish_reason"),
                    "message_content": choice.get("message", {}).get("content")
                }
                for choice in completion_choices
            ]
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        # 예외 발생 시 오류 메시지를 출력하고 500 에러를 반환
        print(f"[LOG] Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
