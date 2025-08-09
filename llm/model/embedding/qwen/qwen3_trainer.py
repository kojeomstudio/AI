# -*- coding: utf-8 -*-
"""
qwen3_trainer.py (config.json 기반, 근본 MPS/meta fix + Adafactor 스케줄러 패치)
--------------------------------------------------------------------------------
- config.json 자동 로드(인자 생략 가능)
- MPS/CPU 안전: device/dtype 통일, gradient checkpointing, 메타 텐서 방지/검증
- 마이크로배치 + 누적학습
- 옵티마이저 선택 (AdamW/Adafactor/SGD) + Adafactor LR/스케줄러 안전 패치
- 단계별 로깅
- 모델/토크나이저 저장 + metrics & config snapshot
- (옵션) Ollama Modelfile 생성 (GGUF 경로 제공 시)
"""

import os
import json
import logging
from pathlib import Path
from contextlib import nullcontext
from datetime import datetime

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
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("trainer")

# ------------------------ Config Loader ------------------------
def load_config(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json을 찾지 못했습니다: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    log.info(f"[Config] loaded {cfg_path}")
    return cfg

def resolve_dtype(device_kind: str, cfg_dtype: str):
    if cfg_dtype == "float16":
        return torch.float16
    if cfg_dtype == "float32":
        return torch.float32
    # auto
    if device_kind == "mps":
        return torch.float16
    return torch.float32

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path: Path, payload: dict, title: str):
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"[Save] wrote {title}: {path}")

def maybe_write_ollama_modelfile(cfg: dict, save_dir: Path):
    o = cfg.get("output", {}).get("ollama", {})
    if not o or not o.get("enable", False):
        log.info("[Ollama] disabled")
        return None
    gguf = o.get("gguf_path")
    modelfile_path = Path(o.get("modelfile_path", str(save_dir / "Modelfile")))
    ensure_dir(modelfile_path.parent)
    lines = []
    if gguf:
        lines.append(f"FROM {Path(gguf).expanduser().resolve()}")
    else:
        lines.append("# FROM /path/to/model.gguf  # ← GGUF 변환 후 경로 설정")
    lines.append(f"PARAMETER num_ctx {o.get('parameters', {}).get('num_ctx', 2048)}")
    lines.append('TEMPLATE """')
    lines.append("{{ text }}")
    lines.append('"""')
    modelfile_content = "\n".join(lines) + "\n"
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    log.info(f"[Ollama] Modelfile generated at: {modelfile_path}")
    return modelfile_path

# ------------------------ Utilities ------------------------
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, L, 1]
    s = (last_hidden_state * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1e-6)
    return s / d

def info_nce_in_batch(z_a, z_p, temperature=0.07):
    z_a = F.normalize(z_a, dim=-1)
    z_p = F.normalize(z_p, dim=-1)
    logits = torch.matmul(z_a, z_p.t()) / temperature  # [B, B]
    labels = torch.arange(z_a.size(0), device=z_a.device)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, acc

def _to_device(d, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in d.items()}

def _check_and_fix_allocation(m, expected_device: torch.device):
    meta_params, offdev_params = [], []
    for n, p in m.named_parameters():
        if getattr(p, "is_meta", False) or p.device.type == "meta":
            meta_params.append(n)
        elif p.device.type != expected_device.type:
            offdev_params.append((n, p.device))
    if meta_params or offdev_params:
        log.warning(f"[Model] reallocating: meta={len(meta_params)} offdev={len(offdev_params)}")
        m.to(expected_device)
        # 재검증
        meta2 = [n for n, p in m.named_parameters()
                 if getattr(p, "is_meta", False) or p.device.type == "meta"]
        off2  = [(n, p.device) for n, p in m.named_parameters()
                 if p.device.type != expected_device.type]
        if meta2 or off2:
            raise RuntimeError(f"[Model] params not allocated/on device. meta={meta2[:3]} offdev={off2[:3]}")

# ------------------------ Main ------------------------
def main(cfg_path: str):
    # Load config
    cfg = load_config(Path(cfg_path))

    # Device & dtype
    prefer_mps = cfg.get("device", {}).get("prefer_mps", True)
    use_mps = torch.backends.mps.is_available() and prefer_mps
    device = torch.device("mps" if use_mps else "cpu")
    dtype = resolve_dtype("mps" if use_mps else "cpu", cfg.get("device", {}).get("dtype", "auto"))
    torch.set_default_dtype(torch.float32)  # 안전한 기본 dtype
    log.info(f"[Device] {device}, dtype={dtype}")

    # Data paths
    pairs_path_cfg = cfg.get("data", {}).get("pairs_path")
    if pairs_path_cfg and Path(pairs_path_cfg).expanduser().exists():
        pairs_path = Path(pairs_path_cfg).expanduser().resolve()
    else:
        pairs_path, _ = infer_data_paths(Path(__file__).resolve().parent)
    assert pairs_path is not None and Path(pairs_path).exists(), "pairs.jsonl 경로를 찾을 수 없습니다."
    log.info(f"[Data] pairs: {pairs_path}")

    # DataLoader
    train_loader = make_dataloader(
        pairs_path=pairs_path,
        tokenizer_name_or_obj=cfg.get("model_id"),
        batch_size=int(cfg.get("train", {}).get("batch_size", 4)),
        shuffle=True,
        max_length=int(cfg.get("data", {}).get("max_length", 64)),
        return_negatives=False,
        return_tuple=False,
        device=device,
        hf_token=cfg.get("hf_token"),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        local_files_only=False,
        pad_to_multiple_of=int(cfg.get("data", {}).get("pad_to_multiple_of", 8)) if use_mps else None,
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = int(cfg.get("train", {}).get("epochs", 1)) * steps_per_epoch
    log.info(f"[Train] steps/epoch={steps_per_epoch}, epochs={cfg.get('train', {}).get('epochs', 1)}, total_steps={total_steps}")

    # ------------------------ Model (근본 해결 버전) ------------------------
    log.info(f"[Model] Loading: {cfg.get('model_id')}")

    tok = AutoTokenizer.from_pretrained(
        cfg.get("model_id"),
        use_fast=True,
        token=cfg.get("hf_token"),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
    )

    # 핵심 1) 실물 텐서로 강제 로드: meta 방지
    model = AutoModel.from_pretrained(
        cfg.get("model_id"),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        torch_dtype=dtype,          # 초기 dtype
        device_map=None,            # ❗ meta 방지
        low_cpu_mem_usage=False,    # ❗ meta 방지
    )

    # (선택) 가장 안전: fp32 로딩 후 나중에 fp16 변환
    # if str(dtype) == "torch.float16":
    #     model = model.to(dtype=torch.float32)

    # 핵심 2) 단일 디바이스로 '깨끗하게' 재배치 (CPU→MPS)
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        import gc
        gc.collect()
        torch.mps.empty_cache()
    model.to(device)
    # 필요 시 dtype 변환 (위에서 fp32로 로딩했다면 여기서 fp16로 변경)
    # if str(dtype) == "torch.float16":
    #     model = model.to(dtype=torch.float16)

    # gradient checkpointing + use_cache False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        log.info("[Model] gradient checkpointing enabled")
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
        log.info("[Model] model.config.use_cache=False")

    # 핵심 3) 메타/오프디바이스 검증 & 복구
    _check_and_fix_allocation(model, device)
    model.train()

    # (옵션) 임베딩 계층을 명시적으로 디바이스 고정
    if hasattr(model, "embed_tokens"):
        model.embed_tokens = model.embed_tokens.to(device)

    # ------------------------ Optimizer/Scheduler ------------------------
    opt_name = str(cfg.get("train", {}).get("optimizer", "adamw")).lower()
    lr = float(cfg.get("train", {}).get("lr", 2e-5))
    weight_decay = float(cfg.get("train", {}).get("weight_decay", 0.01))
    warmup_ratio = float(cfg.get("train", {}).get("warmup_ratio", 0.06))
    epochs = int(cfg.get("train", {}).get("epochs", 1))
    grad_acc_steps = int(cfg.get("train", {}).get("grad_acc_steps", 8))
    log_every = int(cfg.get("train", {}).get("log_every", 10))

    def build_optimizer(name: str, params):
        if name == "adamw":
            return AdamW(params, lr=lr, weight_decay=weight_decay)
        if name == "adafactor" and _HAVE_ADAFACTOR:
            # ✅ 고정 LR 모드: lr 명시 + scale_parameter=False + relative_step=False
            return Adafactor(
                params,
                lr=lr,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                weight_decay=weight_decay,
            )
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9)
        log.warning(f"[Optim] Unknown or unavailable optimizer '{name}', falling back to AdamW")
        return AdamW(params, lr=lr, weight_decay=weight_decay)

    optimizer = build_optimizer(opt_name, model.parameters())

    # ✅ 모든 param_group의 lr가 숫자인지 보정 + initial_lr 채움 (스케줄러 안전)
    for g in optimizer.param_groups:
        if g.get("lr", None) is None:
            g["lr"] = lr
        g["initial_lr"] = float(g["lr"])

    num_train_steps = epochs * steps_per_epoch
    num_warmup = max(1, int(num_train_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_train_steps) if num_train_steps > 0 else None

    log.info(f"[Optim] {opt_name} | lr={lr} | weight_decay={weight_decay} | warmup={num_warmup}/{num_train_steps} | grad_acc_steps={grad_acc_steps}")

    # AMP (MPS)
    use_amp_ctx = torch.amp.autocast(device_type="mps", dtype=dtype) if (device.type == "mps") else nullcontext()

    # ------------------------ Train ------------------------
    global_step = 0
    accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    log.info("[Train] Starting training loop")

    for epoch in range(epochs):
        for step, batch in enumerate(train_loader, start=1):
            enc_a = _to_device(batch["anchor_inputs"], device)
            enc_p = _to_device(batch["positive_inputs"], device)

            with use_amp_ctx:
                out_a = model(**enc_a, return_dict=True)
                out_p = model(**enc_p, return_dict=True)
                z_a = out_a.pooler_output if hasattr(out_a, "pooler_output") and out_a.pooler_output is not None else mean_pool(out_a.last_hidden_state, enc_a["attention_mask"])
                z_p = out_p.pooler_output if hasattr(out_p, "pooler_output") and out_p.pooler_output is not None else mean_pool(out_p.last_hidden_state, enc_p["attention_mask"])
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

                if torch.backends.mps.is_available():
                    import gc
                    gc.collect()
                    torch.mps.empty_cache()

                if (global_step % log_every) == 0 or step == steps_per_epoch:
                    avg_loss = accum_loss
                    accum_loss = 0.0
                    log.info(f"epoch {epoch+1} | step {step}/{steps_per_epoch} | gstep {global_step} | loss={avg_loss:.4f} | acc={acc.item():.3f}")

    log.info("[OK] Training loop finished.")

    # ------------------------ Save ------------------------
    out_cfg = cfg.get("output", {})
    save_dir = ensure_dir(Path(out_cfg.get("save_dir", "./outputs")).expanduser().resolve())
    save_name = out_cfg.get("save_name", "model-ft")
    target_dir = ensure_dir(save_dir / save_name)
    log.info(f"[Save] saving model & tokenizer to: {target_dir}")
    model.save_pretrained(target_dir)
    tok.save_pretrained(target_dir)

    # Save metrics/config snapshot
    if bool(out_cfg.get("save_metrics", True)):
        metrics = {
            "finished_at": datetime.utcnow().isoformat() + "Z",
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": num_train_steps,
            "batch_size": int(cfg.get("train", {}).get("batch_size", 4)),
            "grad_acc_steps": grad_acc_steps,
            "lr": lr,
            "optimizer": opt_name,
            "device": str(device),
            "dtype": str(dtype),
            "model_id": cfg.get("model_id"),
        }
        write_json(target_dir / "metrics.json", metrics, "metrics")
        write_json(target_dir / "config.snapshot.json", cfg, "config snapshot")

    # ------------------------ Ollama Modelfile (optional) ------------------------
    modelfile = maybe_write_ollama_modelfile(cfg, target_dir)
    if modelfile:
        log.info(f"[Ollama] To create model: `ollama create {out_cfg.get('ollama',{}).get('template_name','embed-model')} -f {modelfile}`")
        if not out_cfg.get("ollama", {}).get("gguf_path"):
            log.warning("[Ollama] 현재 Qwen3-Embedding 계열의 GGUF 변환/호환성은 제한적일 수 있습니다. llama.cpp/Ollama 지원 여부 확인이 필요합니다.")

if __name__ == "__main__":
    import argparse, os, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="(옵션) config 파일 경로. 미지정 시 ./config.json 자동 사용")
    args = ap.parse_args()

    default_cfg = os.path.join(os.path.dirname(__file__), "config.json")
    cfg_path = args.config or default_cfg

    if not os.path.exists(cfg_path):
        print(f"[ERROR] config 파일을 찾지 못했습니다: {cfg_path}")
        print("       스크립트와 같은 폴더에 config.json을 두거나 --config로 직접 경로를 지정하세요.")
        sys.exit(1)

    main(cfg_path)