from __future__ import annotations
"""
utils.py
--------
런타임/저장/텐서 유틸리티 모음: 디바이스/AMP 결정, 풀링, InfoNCE 등.
"""
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json
import logging
from contextlib import nullcontext

import torch

logger = logging.getLogger(__name__)


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any], title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if title:
        logger.info(f"[write_json] saved {title}: {path}")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_abs(path_str: Optional[str], base_dir: Path) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    return (base_dir / p).resolve() if not p.is_absolute() else p.resolve()


def ensure_exists(path: Path, hint: str = "") -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path} {hint}")
    return path


def resolve_device_dtype(prefer_mps: bool = True, dtype_opt: str = "auto") -> Tuple[torch.device, torch.dtype, Any]:
    """Resolve device and dtype; returns (device, dtype, autocast_ctx)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
    elif prefer_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = True
    else:
        device = torch.device("cpu")
        use_amp = False

    # dtype
    if dtype_opt == "auto":
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif device.type == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = mapping.get(dtype_opt.lower(), torch.float32)

    # autocast ctx
    if use_amp and device.type == "cuda":
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    elif use_amp and device.type == "mps":
        amp_ctx = torch.amp.autocast(device_type="mps", dtype=dtype)
    else:
        amp_ctx = nullcontext()

    return device, dtype, amp_ctx


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings using attention mask.
    Computes in float32 for numerical stability on fp16/bf16 backends, then casts back.
    """
    lhs = last_hidden_state
    orig_dtype = lhs.dtype
    lhs_f = lhs.float()
    mask_f = attention_mask.unsqueeze(-1).expand(lhs.size()).float()
    masked = lhs_f * mask_f
    summed = masked.sum(dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1e-6)
    pooled = summed / counts
    return pooled.to(orig_dtype)


def info_nce_in_batch(z_a: torch.Tensor, z_p: torch.Tensor, temperature: float = 0.07):
    """In-batch InfoNCE. Returns (loss, accuracy).
    Normalizes in float32 with a safe epsilon to avoid NaN on fp16/bf16.
    """
    z_a_f = torch.nn.functional.normalize(z_a.float(), dim=-1, eps=1e-6)
    z_p_f = torch.nn.functional.normalize(z_p.float(), dim=-1, eps=1e-6)
    logits = (z_a_f @ z_p_f.t()) / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, acc
