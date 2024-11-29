
from utils.file_path_helper import *

from ollama import *
from ollama import Client

client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)

user_query = f""

response = client.chat(model='qwen2.5-cder:latest', messages=[
  {
    'role': 'user',
    'content': f'{user_query}',
  },
])

print(f"response : {response['message']['content']}")