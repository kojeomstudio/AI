import os
import requests
import json

url = os.environ.get("N8N_WEBHOOK_URL", "http://localhost:5678/webhook-test/review")

payload = {
    "trigger": os.environ.get("WEBHOOK_TRIGGER", "local"),
    "target_files": ["a.cpp", "b.py"],
    "build_id": 12345
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, data=json.dumps(payload), headers=headers)

print("Status Code:", response.status_code)
print("Response:", response.text)
