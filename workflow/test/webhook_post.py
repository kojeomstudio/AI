import requests
import json

# n8n 웹훅 주소
url = "http://192.168.0.22:5678/webhook-test/review"

# POST body (JSON 스키마)
payload = {
    "trigger": "kojeomstudio",
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