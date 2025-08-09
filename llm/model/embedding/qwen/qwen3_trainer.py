# pip install "transformers>=4.51.0" peft datasets accelerate faiss-cpu
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model_id = "Qwen/Qwen3-Embedding-0.6B"  # 32~1024차원 가변 지원
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
base_model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16 if device.type!="cpu" else None).to(device)

# LoRA 구성 (주의: Mac에서는 양자화 대신 LoRA 사용을 권장)
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","dense"]  # Qwen3 계열 호환 모듈명
)
model = get_peft_model(base_model, peft_config).to(device)

def masked_mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

class PairDataset(Dataset):
    # 예시: [{"anchor":"...", "positive":"...", "hard_negatives":["...","..."]}, ...]
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch):
    anchors = []
    positives = []
    for b in batch:
        anchors.append(b["anchor"])
        positives.append(b["positive"])
    enc_a = tokenizer(anchors, padding=True, truncation=True, return_tensors="pt", max_length=512)
    enc_p = tokenizer(positives, padding=True, truncation=True, return_tensors="pt", max_length=512)
    return enc_a, enc_p

# 데모용 소규모 데이터 (실전은 수십~수백만 쌍 권장)
train_data = [
    {"anchor": "best laptop for coding", "positive": "A developer laptop buying guide"},
    {"anchor": "python web framework", "positive": "Flask is a lightweight web framework in Python"},
]
train_loader = DataLoader(PairDataset(train_data), batch_size=32, shuffle=True, collate_fn=collate_fn, drop_last=True)

if len(train_loader) == 0:
    raise ValueError("No training data found. Please provide a valid dataset.")

num_epochs = 2
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # LoRA면 lr 크게 가능
num_training_steps = num_epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1*num_training_steps), num_training_steps)

# 학습 가능한 temperature (소배치시 ↑)
temperature = nn.Parameter(torch.tensor(0.1, device=device))

model.train()

last_loss = None
for epoch in range(num_epochs):
    for step, (enc_a, enc_p) in enumerate(train_loader):
        enc_a = {k: v.to(device) for k, v in enc_a.items()}
        enc_p = {k: v.to(device) for k, v in enc_p.items()}

        out_a = model(**enc_a, return_dict=True)
        out_p = model(**enc_p, return_dict=True)

        emb_a = masked_mean_pooling(out_a.last_hidden_state, enc_a["attention_mask"])
        emb_p = masked_mean_pooling(out_p.last_hidden_state, enc_p["attention_mask"])
        emb_a = F.normalize(emb_a, p=2, dim=1)
        emb_p = F.normalize(emb_p, p=2, dim=1)

        logits = (emb_a @ emb_p.T) / temperature.clamp(min=1e-6)
        labels = torch.arange(logits.size(0), device=device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy((emb_p @ emb_a.T) / temperature.clamp(min=1e-6), labels)
        loss = 0.5 * (loss_i2t + loss_t2i)

        last_loss = loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    print(f"epoch {epoch+1} | loss={last_loss:.4f} | temp={temperature.item():.3f}")

model.save_pretrained("./qwen3_embed_0p6b_lora_mps")
tokenizer.save_pretrained("./qwen3_embed_0p6b_lora_mps")