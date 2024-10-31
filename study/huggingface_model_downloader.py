import transformers
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

# Hugging Face 로그인 (토큰을 사용하여 인증)
login(token='your-token-here')

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")
model = AutoModelForCausalLM.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")
