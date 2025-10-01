import aiohttp
import asyncio
import json

async def send_webhook():
    #url = "http://127.0.0.1:5678/webhook/client_code_review" # production
    url = "http://127.0.0.1:5678/webhook-test/client_code_review" # development
    headers = {
        "Content-Type": "application/json"
    }

    # target_files를 배열로, 각 요소는 {파일명: 변경점 문자열}
    body = {
        "trigger": "kojeomstudio",
        "target_files": [
            {"test_A.cpp": "변경점 없음."},
            {"test_B.cpp": "변경점 없음"}
        ],
        "build_id": 12345
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=body) as response:
                resp_text = await response.text()
                print("=== Webhook Response ===")
                print(resp_text)
        except Exception as e:
            print("Error sending webhook:", e)

if __name__ == "__main__":
    asyncio.run(send_webhook())
