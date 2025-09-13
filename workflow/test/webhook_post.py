import requests
import json

# n8n 웹훅 주소
url = "http://localhost:5678/webhook/test"

# POST body (JSON 스키마)
payload = {
    "trigger": "kuma",
    "target_files": ["a.cpp", "b.py"],
    "build_id": 12345
}

# 요청 헤더 (JSON 전송)
headers = {
    "Content-Type": "application/json"
}

# POST 요청
response = requests.post(url, data=json.dumps(payload), headers=headers)

# 결과 출력
print("Status Code:", response.status_code)
print("Response:", response.text)