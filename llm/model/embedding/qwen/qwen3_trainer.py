# -*- coding: utf-8 -*-
"""
qwen3_trainer.py
----------------
MPS/CPU 안전한 임베딩 대조학습(InfoNCE) 미니 트레이너.
- 디바이스/dtype 일관성
- 그래디언트 체크포인팅
- 마이크로배치 + 누적학습
- 옵티마이저 선택(AdamW/Adafactor/SGD)
- 구조화 로깅
"""

import os
import math
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from transformers import Adafactor  # optional
    _HAVE_ADAFACTOR = True
except Exception:
    _HAVE_ADAFACTOR = False

from train_helpers import make_dataloader, infer_data_paths

# ------------------------ Logging ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("trainer")

# ------------------------ Config ------------------------
MODEL_ID           = os.getenv("MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")
HF_TOKEN           = os.getenv("HF_TOKEN")           # gated/private면 필요
TRUST_REMOTE_CODE  = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"

BATCH_SIZE         = int(os.getenv("BATCH_SIZE", "4"))     # micro-batch
GRAD_ACC_STEPS     = int(os.getenv("GRAD_ACC_STEPS", "8")) # grad accumulation
MAX_LENGTH         = int(os.getenv("MAX_LENGTH", "64"))
LR                 = float(os.getenv("LR", "2e-5"))
EPOCHS             = int(os.getenv("EPOCHS", "1"))
WARMUP_RATIO       = float(os.getenv("WARMUP_RATIO", "0.06"))
WEIGHT_DECAY       = float(os.getenv("WEIGHT_DECAY", "0.01"))
OPTIM              = os.getenv("OPTIM", "adamw").lower()   # adamw | adafactor | sgd
LOG_EVERY          = int(os.getenv("LOG_EVERY", "10"))
RETURN_NEGATIVES   = os.getenv("RETURN_NEGATIVES", "false").lower() == "true"  # in-batch 기본

# ------------------------ Paths ------------------------
pairs_path, _ = infer_data_paths(Path(__file__).resolve().parent)
assert pairs_path is not None, "pairs.jsonl 경로를 찾을 수 없습니다."
log.info(f"[Data] pairs: {pairs_path}")

# ------------------------ Device/Dtype ------------------------
use_mps = torch.backends.mps.is_available()
device = torch.device("mps" if use_mps else "cpu")
dtype = torch.float16 if use_mps else torch.float32  # MPS: fp16 권장, 문제시 fp32
log.info(f"[Device] {device}, dtype={dtype}")

# ------------------------ DataLoader ------------------------
train_loader = make_dataloader(
    pairs_path=pairs_path,
    tokenizer_name_or_obj=MODEL_ID,
    batch_size=BATCH_SIZE,
    shuffle=True,
    max_length=MAX_LENGTH,
    return_negatives=RETURN_NEGATIVES,
    return_tuple=False,
    device=device,                 # collator가 토큰을 바로 device로 이동
    hf_token=HF_TOKEN,
    trust_remote_code=TRUST_REMOTE_CODE,
    local_files_only=False,
    pad_to_multiple_of=8 if use_mps else None,
)
steps_per_epoch = max(1, len(train_loader))
log.info(f"[Train] steps/epoch={steps_per_epoch}, epochs={EPOCHS}, total_steps={EPOCHS * steps_per_epoch}")

# ------------------------ Model ------------------------
log.info(f"[Model] Loading: {MODEL_ID}")
tok = AutoTokenizer.from_pretrained(
    MODEL_ID, use_fast=True, token=HF_TOKEN, trust_remote_code=TRUST_REMOTE_CODE
)
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=TRUST_REMOTE_CODE,
    torch_dtype=dtype,
    device_map=None,        # meta 텐서 방지
)

# gradient checkpointing + use_cache False
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
    log.info("[Model] gradient checkpointing enabled")
if hasattr(model, "config") and hasattr(model.config, "use_cache"):
    model.config.use_cache = False
    log.info("[Model] model.config.use_cache=False")

model.to(device)
model.train()

# ------------------------ Optimizer/Scheduler ------------------------
def build_optimizer(name: str, params):
    if name == "adamw":
        return AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    if name == "adafactor" and _HAVE_ADAFACTOR:
        return Adafactor(params, scale_parameter=True, relative_step=False, warmup_init=False, weight_decay=WEIGHT_DECAY)
    if name == "sgd":
        return torch.optim.SGD(params, lr=LR, momentum=0.9)
    log.warning(f"[Optim] Unknown or unavailable optimizer '{name}', falling back to AdamW")
    return AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

optimizer = build_optimizer(OPTIM, model.parameters())
num_train_steps = EPOCHS * steps_per_epoch
num_warmup = max(1, int(num_train_steps * WARMUP_RATIO))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_train_steps)
log.info(f"[Optim] {OPTIM} | lr={LR} | weight_decay={WEIGHT_DECAY} | warmup={num_warmup}/{num_train_steps}")

# ------------------------ Utils ------------------------
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, L, 1]
    s = (last_hidden_state * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1e-6)
    return s / d

def get_embeddings(enc_inputs):
    out = model(**enc_inputs, return_dict=True)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    return mean_pool(out.last_hidden_state, enc_inputs["attention_mask"])

def info_nce_in_batch(z_a, z_p, temperature=0.07):
    z_a = F.normalize(z_a, dim=-1)
    z_p = F.normalize(z_p, dim=-1)
    logits = torch.matmul(z_a, z_p.t()) / temperature  # [B, B]
    labels = torch.arange(z_a.size(0), device=z_a.device)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, acc

# ------------------------ Train ------------------------
global_step = 0
accum_loss = 0.0
optimizer.zero_grad(set_to_none=True)

log.info("[Train] Starting training loop")
for epoch in range(EPOCHS):
    for step, batch in enumerate(train_loader, start=1):
        # batch already on correct device
        enc_a = batch["anchor_inputs"]
        enc_p = batch["positive_inputs"]

        # Mixed precision autocast for MPS
        ctx = torch.amp.autocast(device_type="mps", dtype=dtype) if use_mps else torch.no_grad()
        # For CPU, we won't autocast; run in full precision
        if use_mps:
            with ctx:
                z_a = get_embeddings(enc_a)
                z_p = get_embeddings(enc_p)
                loss, acc = info_nce_in_batch(z_a, z_p, temperature=0.07)
                loss = loss / GRAD_ACC_STEPS
        else:
            z_a = get_embeddings(enc_a)
            z_p = get_embeddings(enc_p)
            loss, acc = info_nce_in_batch(z_a, z_p, temperature=0.07)
            loss = loss / GRAD_ACC_STEPS

        loss.backward()
        accum_loss += loss.item()

        if (step % GRAD_ACC_STEPS) == 0:
            try:
                optimizer.step()
            except RuntimeError as e:
                log.error(f"[OOM?] optimizer.step() failed: {e}")
                if torch.backends.mps.is_available():
                    import gc
                    gc.collect()
                    torch.mps.empty_cache()
                raise

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            # optional memory cleanup on MPS
            if torch.backends.mps.is_available():
                import gc
                gc.collect()
                torch.mps.empty_cache()

            if (global_step % LOG_EVERY) == 0 or step == steps_per_epoch:
                avg_loss = accum_loss
                accum_loss = 0.0
                log.info(f"epoch {epoch+1} | step {step}/{steps_per_epoch} | gstep {global_step} | loss={avg_loss:.4f} | acc={acc.item():.3f}")

log.info("[OK] Training loop finished.")
