from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from app.ollama_client import prompt_ollama
from app.prompt_builder import build_prompt
from app.log_helper import g_logger
from app.config_loader import CONFIG

app = FastAPI()

class PromptRequest(BaseModel):
    query: str

@app.post("/prompt")
async def send_prompt(req: PromptRequest):
    """유저 쿼리를 프롬프트화하고 Ollama에 전달"""
    g_logger.info(f"Received user query: {req.query}")

    final_prompt = build_prompt(req.query)
    g_logger.info(f"Final prompt to send: {final_prompt}")

    response = await prompt_ollama(final_prompt)
    return {"response": response}

if __name__ == "__main__":
    g_logger.info("Starting FastAPI server...")
    uvicorn.run(
        "main:app",
        host=CONFIG['server_host'],
        port=CONFIG['server_port'],
        reload=True
    )
