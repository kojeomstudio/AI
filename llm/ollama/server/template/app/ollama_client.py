import httpx
from app.config_loader import CONFIG
from app.log_helper import g_logger

async def prompt_ollama(prompt: str) -> str:
    """Ollama 서버에 비동기로 프롬프트 전달"""
    url = f"{CONFIG['ollama_base_url']}/api/generate"
    payload = {
        "model": CONFIG['model_name'],
        "prompt": prompt
    }

    g_logger.info(f"Sending prompt to Ollama:\n{prompt}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            g_logger.info("Received response from Ollama.")
            return result.get("response", "No response field in Ollama reply.")
        except httpx.RequestError as e:
            g_logger.error(f"Ollama request failed: {str(e)}")
            return f"Error: {str(e)}"
