from __future__ import annotations
"""
train.py
---------
일반화된 임베딩 모델 파인튜너. 설정 파일(config.json)을 읽어
Hugging Face 임베딩 모델을 로드하고, pairs.jsonl 데이터로 InfoNCE 학습을 수행합니다.

주요 흐름
1) 설정/런타임 파라미터 로딩 (모델/토크나이저/디바이스/학습옵션)
2) pairs.jsonl 로더 + DataLoader 구성
3) 모델 로딩 및 mean-pooling 기반 InfoNCE 학습 루프
4) 체크포인트 저장 및 스냅샷 기록
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict
import logging
from datetime import datetime

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

# 안전한 임포트: 패키지/단일 스크립트 실행 모두 지원
if __package__ in (None, ""):
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from utils import (  # type: ignore
        load_json,
        write_json,
        ensure_dir,
        to_abs,
        resolve_device_dtype,
        mean_pool,
        info_nce_in_batch,
    )
    from data_utils import make_dataloader, infer_pairs_path  # type: ignore
else:
    from .utils import (
        load_json,
        write_json,
        ensure_dir,
        to_abs,
        resolve_device_dtype,
        mean_pool,
        info_nce_in_batch,
    )
    from .data_utils import make_dataloader, infer_pairs_path


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("embed_trainer")


def build_optimizer(name: str, params, lr: float, weight_decay: float):
    name = (name or "adamw").lower()
    if name in ("adamw", "adamw_torch"):
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    logging.getLogger(__name__).warning(
        f"Unknown optimizer '{name}', falling back to AdamW"
    )
    return AdamW(params, lr=lr, weight_decay=weight_decay)


def main(cfg_path: Path):
    log = setup_logger()
    base_dir = Path(__file__).resolve().parent
    cfg = load_json(cfg_path)

    # -------------------- HF & runtime opts --------------------
    model_id = cfg.get("model_id")
    tok_id = cfg.get("tokenizer_id") or model_id
    if not model_id:
        raise ValueError("config.model_id is required")

    hf = cfg.get("hf", {})
    hf_token = hf.get("token")
    trust_remote_code = bool(hf.get("trust_remote_code", False))
    local_files_only = bool(hf.get("local_files_only", False))

    device_opt = (cfg.get("device") or "auto").lower()
    dtype_opt = (cfg.get("dtype") or "auto").lower()
    if device_opt == "auto":
        device, dtype, amp_ctx = resolve_device_dtype(prefer_mps=True, dtype_opt=dtype_opt)
    else:
        # explicit device selection
        if device_opt == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device_opt == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # dtype selection
        if dtype_opt == "auto":
            if device.type == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif device.type == "mps":
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
            dtype = mapping.get(dtype_opt.lower(), torch.float32)

        # autocast context
        if device.type == "cuda":
            amp_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
        elif device.type == "mps":
            amp_ctx = torch.amp.autocast(device_type="mps", dtype=dtype)
        else:
            from contextlib import nullcontext
            amp_ctx = nullcontext()

    log.info(f"[Runtime] device={device} dtype={dtype}")

    # -------------------- Data --------------------
    data_cfg = cfg.get("data", {})
    pairs_path = to_abs(data_cfg.get("pairs_path"), cfg_path.parent)
    if pairs_path is None:
        pairs_path = infer_pairs_path(base_dir)
        if pairs_path is None:
            raise FileNotFoundError(
                "pairs.jsonl not found. Set data.pairs_path or place under tools/data or data"
            )
    log.info(f"[Data] pairs: {pairs_path}")

    # -------------------- Model/Tokenizer --------------------
    # 모델/토크나이저 로딩. AutoModel을 사용하여 대부분의 임베딩 백본과 호환
    log.info(f"[Model] loading: {model_id}")
    tok = AutoTokenizer.from_pretrained(
        tok_id,
        token=hf_token,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        use_fast=True,
    )
    model = AutoModel.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        torch_dtype=dtype,
    )
    model.to(device)
    model.train()

    # -------------------- DataLoader --------------------
    tr = cfg.get("train", {})
    batch_size = int(tr.get("batch_size", 8))
    epochs = int(tr.get("epochs", 1))
    max_length = int(tr.get("max_length", 128))
    grad_acc_steps = max(1, int(tr.get("grad_accum_steps", 1)))
    steps_per_epoch = tr.get("steps_per_epoch")
    steps_per_epoch = int(steps_per_epoch) if steps_per_epoch else None
    log_every = int(tr.get("log_every", 50))

    train_loader = make_dataloader(
        tokenizer_name_or_obj=tok,
        pairs_path=pairs_path,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        shuffle=True,
        num_workers=0,
        hf_token=hf_token,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    if steps_per_epoch is None:
        steps_per_epoch = len(train_loader)
        log.info(f"[Train] inferred steps_per_epoch={steps_per_epoch}")

    # -------------------- Optim/Scheduler --------------------
    optim_cfg = cfg.get("optim", {})
    lr = float(optim_cfg.get("lr", 5e-5))
    weight_decay = float(optim_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(optim_cfg.get("warmup_ratio", 0.05))
    optimizer = build_optimizer(optim_cfg.get("name", "adamw"), model.parameters(), lr, weight_decay)
    for g in optimizer.param_groups:
        g.setdefault("initial_lr", g.get("lr", lr))

    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps) if total_steps > 0 else None
    log.info(f"[Optim] lr={lr} wd={weight_decay} warmup={warmup_steps}/{total_steps} grad_acc={grad_acc_steps}")

    # -------------------- Train Loop --------------------
    # 인배치 InfoNCE를 사용하므로 별도 네거티브 샘플링 없이 배치 내에서 대조 학습
    global_step = 0
    accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=total_steps, desc="train", ncols=120)
    for epoch in range(epochs):
        it = iter(train_loader)
        for step in range(1, steps_per_epoch + 1):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)

            enc_a = batch["anchor_inputs"]
            enc_p = batch["positive_inputs"]

            with amp_ctx:
                out_a = model(**enc_a, return_dict=True)
                out_p = model(**enc_p, return_dict=True)
                z_a = (
                    out_a.pooler_output
                    if hasattr(out_a, "pooler_output") and out_a.pooler_output is not None
                    else mean_pool(out_a.last_hidden_state, enc_a["attention_mask"])
                )
                z_p = (
                    out_p.pooler_output
                    if hasattr(out_p, "pooler_output") and out_p.pooler_output is not None
                    else mean_pool(out_p.last_hidden_state, enc_p["attention_mask"])
                )
                loss, acc = info_nce_in_batch(z_a, z_p, temperature=0.07)
                loss = loss / grad_acc_steps

            loss.backward()
            accum_loss += loss.item()

            if (step % grad_acc_steps) == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                pbar.update(1)

                if device.type == "mps":
                    import gc
                    gc.collect()
                    torch.mps.empty_cache()

                if (global_step % log_every) == 0 or step == steps_per_epoch:
                    avg_loss = accum_loss
                    accum_loss = 0.0
                    log.info(
                        f"epoch {epoch+1} | step {step}/{steps_per_epoch} | gstep {global_step} | loss={avg_loss:.4f} | acc={acc.item():.3f}"
                    )
    pbar.close()
    log.info("[OK] training finished")

    # -------------------- Save --------------------
    out_cfg = cfg.get("output", {})
    save_dir = ensure_dir(Path(out_cfg.get("save_dir", "./outputs")).expanduser().resolve())
    save_name = out_cfg.get("save_name", "embed-ft")
    target = ensure_dir(save_dir / save_name)
    log.info(f"[Save] to: {target}")
    model.save_pretrained(target)
    tok.save_pretrained(target)

    if bool(out_cfg.get("save_metrics", True)):
        metrics = {
            "finished_at": datetime.utcnow().isoformat() + "Z",
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "batch_size": batch_size,
            "grad_acc_steps": grad_acc_steps,
            "lr": lr,
            "device": str(device),
            "dtype": str(dtype),
            "model_id": model_id,
        }
        write_json(target / "metrics.json", metrics, "metrics")
        write_json(target / "config.snapshot.json", cfg, "config snapshot")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="General Embedding Fine-tuner")
    ap.add_argument("--config", type=str, default=None, help="config file path (default: ./config.example.json)")
    args = ap.parse_args()

    default_cfg = Path(__file__).parent / "config.example.json"
    cfg_path = Path(args.config) if args.config else default_cfg

    if not cfg_path.exists():
        print(f"[ERROR] config not found: {cfg_path}")
        sys.exit(1)

    main(cfg_path)
