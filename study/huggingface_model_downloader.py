import os
import transformers
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)
else:
    print("HF_TOKEN 환경변수가 설정되지 않았습니다. 로그인을 건너뜁니다.")

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")
model = AutoModelForCausalLM.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")
