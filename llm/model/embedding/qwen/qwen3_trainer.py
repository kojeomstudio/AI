# pip install "transformers>=4.51.0" peft accelerate
import os
import math
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from pathlib import Path
from train_helpers import make_dataloader, infer_data_paths

# ------------------------ Config ------------------------
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")
HF_TOKEN = os.getenv("HF_TOKEN")  # huggingface-cli login 했다면 없어도 OK
TRUST_REMOTE_CODE = True

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
LR = float(os.getenv("LR", "2e-5"))
EPOCHS = int(os.getenv("EPOCHS", "1"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.06"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
RETURN_NEGATIVES = False  # in-batch negatives 사용

# ------------------------ Paths ------------------------
pairs_path, _ = infer_data_paths(Path(__file__).resolve().parent)
assert pairs_path is not None, "pairs.jsonl 경로를 찾을 수 없습니다."
print(f"[Data] pairs: {pairs_path}")

# ------------------------ Device/Dtype ------------------------
use_mps = torch.backends.mps.is_available()
device = torch.device("mps" if use_mps else "cpu")
dtype = torch.float16 if use_mps else torch.float32  # MPS: fp16 권장, 필요시 fp32

print(f"[Device] {device}, dtype={dtype}")

# ------------------------ DataLoader ------------------------
train_loader = make_dataloader(
    pairs_path=pairs_path,
    tokenizer_name_or_obj=MODEL_ID,
    batch_size=BATCH_SIZE,
    shuffle=True,
    max_length=MAX_LENGTH,
    return_negatives=RETURN_NEGATIVES,
    return_tuple=False,
    device=device,                 # ✅ collator가 토큰을 바로 device로 이동
    hf_token=HF_TOKEN,
    trust_remote_code=TRUST_REMOTE_CODE,
    local_files_only=False,
    pad_to_multiple_of=8 if use_mps else None,
)
steps_per_epoch = max(1, len(train_loader))

# ------------------------ Model ------------------------
tok = AutoTokenizer.from_pretrained(
    MODEL_ID, use_fast=True, token=HF_TOKEN, trust_remote_code=TRUST_REMOTE_CODE
)
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=TRUST_REMOTE_CODE,
    torch_dtype=dtype,
    device_map=None,        # ✅ meta 텐서 방지
)
model.to(device)
model.train()

# ------------------------ Optimizer/Scheduler ------------------------
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
num_train_steps = EPOCHS * steps_per_epoch
num_warmup = max(1, int(num_train_steps * WARMUP_RATIO))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_train_steps)

# ------------------------ Utils ------------------------
def mean_pool(last_hidden_state, attention_mask):
    # attention_mask: [B, L]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, L, 1]
    s = (last_hidden_state * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1e-6)
    return s / d

def get_embeddings(enc_inputs):
    with torch.set_grad_enabled(True):
        out = model(**enc_inputs, return_dict=True)
        # 우선순위: pooler_output → last_hidden_state mean pool
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        else:
            return mean_pool(out.last_hidden_state, enc_inputs["attention_mask"])

def info_nce_in_batch(z_a, z_p, temperature=0.07):
    # z_a, z_p: [B, D]
    z_a = F.normalize(z_a, dim=-1)
    z_p = F.normalize(z_p, dim=-1)
    logits = torch.matmul(z_a, z_p.t()) / temperature  # [B, B]
    labels = torch.arange(z_a.size(0), device=z_a.device)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, acc

# ------------------------ Train ------------------------
global_step = 0
print(f"[Train] steps/epoch={steps_per_epoch}, epochs={EPOCHS}, total_steps={num_train_steps}")
for epoch in range(EPOCHS):
    for step, batch in enumerate(train_loader, start=1):
        enc_a = batch["anchor_inputs"]
        enc_p = batch["positive_inputs"]

        z_a = get_embeddings(enc_a)  # [B, D]
        z_p = get_embeddings(enc_p)  # [B, D]
        loss, acc = info_nce_in_batch(z_a, z_p, temperature=0.07)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        global_step += 1
        if step % 10 == 0 or step == steps_per_epoch:
            print(f"epoch {epoch+1} step {step}/{steps_per_epoch} | loss={loss.item():.4f} | acc={acc.item():.3f}")

print("[OK] Training loop finished.")
