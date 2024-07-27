import transformers
import torch

from huggingface_hub import login

# my token...
# 
login(token='Youre Huggingface user token')

# huggingface의 경우, 모델 다운로드 기본 경로가 유저/.cache/~
# 필요하면 system var 추가해서 local storage 경로 변경 가능.

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])