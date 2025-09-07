from __future__ import annotations
"""
train.py
---------
일반화된 임베딩 모델 파인튜너. 설정 파일(config.json)을 읽어
Hugging Face 임베딩 모델을 로드하고, pairs.jsonl 데이터로 InfoNCE 학습을 수행합니다.

#주요 흐름
1) 설정/런타임 파라미터 로딩 (모델/토크나이저/디바이스/학습옵션)
2) pairs.jsonl 로더 + DataLoader 구성
3) 모델 로딩 및 mean-pooling 기반 InfoNCE 학습 루프
4) 체크포인트 저장 및 스냅샷 기록

#예외처리.
# 동적 모듈 캐시 폴더만 제거
1) rm -rf ~/.cache/huggingface/modules/transformers_modules/jinaai/xlm-roberta-flash-implementation
2) pip install -U "transformers>=4.44" "huggingface_hub>=0.24" accelerate

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
        cls_pool,
        last_token_pool,
        info_nce_in_batch,
    )
    from data_utils import make_dataloader, infer_pairs_path, count_pairs  # type: ignore
else:
    from .utils import (
        load_json,
        write_json,
        ensure_dir,
        to_abs,
        resolve_device_dtype,
        mean_pool,
        cls_pool,
        last_token_pool,
        info_nce_in_batch,
    )
    from .data_utils import make_dataloader, infer_pairs_path, count_pairs


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("embed_trainer")


def _maybe_set_default_task(model: Any, task: str, log: logging.Logger) -> None:
    """Best-effort: set `default_task` attribute on model or known submodules.
    This is useful for models like Jina Embeddings v3 when using remote wrappers.
    """
    if not task:
        return
    try:
        if hasattr(model, "default_task"):
            setattr(model, "default_task", task)
            log.info(f"[Model] set default_task on model: {task}")
            return
        # Try common submodules
        for name in (
            "base_model",
            "model",
            "backbone",
            "encoder",
            "transformer",
            "bert",
            "roberta",
            "deberta",
            "xlm_roberta",
        ):
            sub = getattr(model, name, None)
            if sub is None:
                continue
            if hasattr(sub, "default_task"):
                setattr(sub, "default_task", task)
                log.info(f"[Model] set default_task on submodule '{name}': {task}")
                return
        # Some composite wrappers may be indexable (e.g., SentenceTransformer-like)
        try:
            sub0 = model[0]  # type: ignore[index]
            if hasattr(sub0, "default_task"):
                setattr(sub0, "default_task", task)
                log.info(f"[Model] set default_task on model[0]: {task}")
                return
        except Exception:
            pass
    except Exception as e:
        log.warning(f"[Model] failed to set default_task='{task}': {e}")


def _patch_jina_rotary_alias(log: logging.Logger) -> None:
    """Workaround for some Jina flash impl versions where backward references
    `apply_rotary` while only `apply_rotary_emb` exists. Alias if needed."""
    try:
        patched = 0
        for name, mod in list(sys.modules.items()):
            file = getattr(mod, "__file__", None)
            if not file or "xlm-roberta-flash-implementation" not in str(file):
                continue
            if str(file).endswith("rotary.py"):
                # On non-CUDA setups, their module doesn't define apply_rotary at all.
                # Build a pure-torch fallback that mirrors apply_rotary semantics and
                # supports backward by using the same rotation with sign flip (conjugate).
                if hasattr(mod, "apply_rotary_emb_torch") and not hasattr(mod, "apply_rotary"):
                    torch_impl = getattr(mod, "apply_rotary_emb_torch")

                    def _apply_rotary_fallback(x, cos, sin, *args, interleaved=False, inplace=False, conjugate=False, **kwargs):  # type: ignore[no-redef]
                        try:
                            if conjugate:
                                sin = -sin
                            out = torch_impl(x, cos, sin, interleaved=interleaved)
                            if inplace:
                                x.copy_(out)
                                return x
                            return out
                        except Exception as e:
                            raise RuntimeError(f"apply_rotary fallback failed: {e}")

                    setattr(mod, "apply_rotary", _apply_rotary_fallback)
                    patched += 1
        if patched:
            log.warning(f"[Patch] Installed rotary torch fallback in {patched} module(s) (no CUDA apply_rotary)")
    except Exception as e:
        log.warning(f"[Patch] rotary fallback failed: {e}")


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
    # Respect config for remote code; some models ship custom wrappers that may
    # disable grads (e.g., inference-only). Default True for compatibility but
    # do NOT override user config.
    trust_remote_code = True
    cfg_trust_remote = hf.get("trust_remote_code")
    if cfg_trust_remote is not None:
        trust_remote_code = bool(cfg_trust_remote)
    local_files_only = bool(hf.get("local_files_only", False))
    default_task = hf.get("default_task")  # optional; e.g., "classification"
    loader = (hf.get("loader") or "automodel").lower()  # "automodel" | "sentence_transformer"

    # Do not force-enable remote code for any specific model. If a repo requires
    # remote code, the user can set it explicitly in config.

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

    # MPS OOM mitigation: relax high-watermark unless explicitly set
    if device.type == "mps" and os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") is None:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

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
    log.info(f"[Model] loading: {model_id} (loader={loader})")
    tok = AutoTokenizer.from_pretrained(
        tok_id,
        token=hf_token,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        use_fast=True,
    )

    st_transformer = None
    if loader == "sentence_transformer":
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "hf.loader=sentence_transformer requires 'sentence-transformers' package.\n"
                "Install: pip install -U sentence-transformers"
            ) from e
        st_kwargs = {}
        if isinstance(default_task, str) and default_task:
            st_kwargs["model_kwargs"] = {"default_task": default_task}
        model = SentenceTransformer(
            model_id,
            trust_remote_code=trust_remote_code,
            **st_kwargs,
        )
        # Keep a handle to the first module (Transformer) for token-level outputs
        try:
            st_transformer = model[0]
        except Exception:
            st_transformer = None
    else:
        model = AutoModel.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
    # Ensure device placement and dtype cast for memory control
    try:
        model.to(device, dtype=dtype)
    except Exception:
        model.to(device)
    model.train()
    # If user asked for a specific default_task (for remote wrappers), try to set it.
    if isinstance(default_task, str) and default_task:
        _maybe_set_default_task(model, default_task, log)
    # Apply Jina rotary alias patch if those modules are present
    if trust_remote_code:
        _patch_jina_rotary_alias(log)
    # Ensure all parameters are trainable (some remote wrappers may freeze by default)
    for p in model.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)

    # Some remote models wrap forward with no_grad (e.g., inference-only wrappers).
    # Detect this and try to locate a trainable submodule to use for forward.
    forward_target = model
    try:
        probe_inputs = tok(
            ["__probe__"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8,
        )
        probe_inputs = {k: v.to(device) for k, v in probe_inputs.items()}
        # Prepare a forward that returns an object with last_hidden_state
        class _Out:
            def __init__(self, x):
                self.last_hidden_state = x

        def _call_forward(m, inputs):
            # SentenceTransformer transformer module returns dict with 'token_embeddings'
            if st_transformer is not None:
                # Disable AMP for ST transformer to avoid dtype mismatch inside custom modules
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    out = st_transformer(inputs)
                if isinstance(out, dict) and "token_embeddings" in out:
                    return _Out(out["token_embeddings"])  # type: ignore[index]
                if isinstance(out, dict) and "last_hidden_state" in out:
                    return _Out(out["last_hidden_state"])  # type: ignore[index]
                if hasattr(out, "last_hidden_state"):
                    return _Out(out.last_hidden_state)  # type: ignore[attr-defined]
                raise RuntimeError("Unexpected ST transformer output shape")
            # AutoModel path
            forward_fn = getattr(m, "forward", m)
            return forward_fn(**inputs, return_dict=True)

        with torch.enable_grad():
            with amp_ctx:
                probe_out = _call_forward(model, probe_inputs)
        grad_ok = (
            getattr(probe_out, "last_hidden_state", None) is not None
            and probe_out.last_hidden_state.requires_grad
        )
    except Exception as e:
        # If probing fails, assume not ok so we try submodules next
        grad_ok = False

    if not grad_ok:
        # Try common base attributes to bypass no_grad wrappers
        candidate_attrs = [
            "base_model",
            "model",
            "backbone",
            "encoder",
            "transformer",
            "bert",
            "roberta",
            "deberta",
            "xlm_roberta",
        ]
        for name in candidate_attrs:
            sub = getattr(model, name, None)
            if sub is None:
                continue
            try:
                with torch.enable_grad():
                    with amp_ctx:
                        test_out = _call_forward(sub, probe_inputs)
                if getattr(test_out, "last_hidden_state", None) is not None and test_out.last_hidden_state.requires_grad:
                    forward_target = sub
                    log.warning(f"[Model] detected grad-disabled wrapper; using submodule '{name}' for training forward")
                    break
            except Exception:
                continue
        else:
            log.warning(
                "[Model] Forward appears to disable grads (no grad_fn). Training may fail. "
                "Consider setting hf.trust_remote_code=false in config to load the base architecture."
            )

    # -------------------- DataLoader --------------------
    tr = cfg.get("train", {})
    batch_size = int(tr.get("batch_size", 8))
    epochs = int(tr.get("epochs", 1))
    max_length = int(tr.get("max_length", 128))
    grad_acc_steps = max(1, int(tr.get("grad_accum_steps", 1)))
    steps_per_epoch = tr.get("steps_per_epoch")
    steps_per_epoch = int(steps_per_epoch) if steps_per_epoch else None
    log_every = int(tr.get("log_every", 50))
    num_workers = int(tr.get("num_workers", 0))
    pad_to_multiple_of = tr.get("pad_to_multiple_of")
    pad_to_multiple_of = int(pad_to_multiple_of) if pad_to_multiple_of else None
    pooling_method = (tr.get("pooling", "mean") or "mean").lower()  # mean|cls|last
    temperature = float(tr.get("temperature", 0.07))
    clip_grad_norm = float(tr.get("clip_grad_norm", 0.0))
    freeze_base = bool(tr.get("freeze_base", False))
    proj_cfg = tr.get("projection_head") or {}
    use_proj = bool(proj_cfg.get("enabled", False))
    proj_out = int(proj_cfg.get("out_dim", 0))
    proj_act = (proj_cfg.get("activation", "tanh") or "tanh").lower()

    data_stream = bool(cfg.get("data", {}).get("stream", False))
    use_hard_negs = bool(tr.get("use_hard_negatives", False))
    negs_per_anchor = int(tr.get("negatives_per_anchor", 0))

    train_loader = make_dataloader(
        tokenizer_name_or_obj=tok,
        pairs_path=pairs_path,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        shuffle=True,
        num_workers=num_workers,
        hf_token=hf_token,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        stream=data_stream,
        pad_to_multiple_of=pad_to_multiple_of,
        include_negatives=use_hard_negs,
        negatives_per_anchor=negs_per_anchor,
    )

    if steps_per_epoch is None:
        # If streaming, len(dataloader) may be undefined. Fallback to counting lines.
        try:
            steps_per_epoch = len(train_loader)
        except TypeError:
            # Estimate steps from file length to keep memory small
            n_pairs = count_pairs(pairs_path)
            steps_per_epoch = max(1, n_pairs // batch_size)
        log.info(f"[Train] inferred steps_per_epoch={steps_per_epoch}")

    # -------------------- Optional: Freeze base --------------------
    if freeze_base:
        frozen = 0
        for p in model.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                frozen += 1
        log.info(f"[Train] base model frozen params: {frozen}")

    # -------------------- Optional: Projection head --------------------
    import torch.nn as nn
    proj_head = None
    hidden_size = None
    try:
        # reuse earlier probe to infer hidden size
        # build minimal input
        probe_inputs = tok(["__probe__"], return_tensors="pt", padding=True, truncation=True, max_length=8)
        probe_inputs = {k: v.to(device) for k, v in probe_inputs.items()}
        with torch.no_grad():
            with amp_ctx:
                probe_out = forward_target(**probe_inputs, return_dict=True) if st_transformer is None else None
        if st_transformer is not None:
            with torch.amp.autocast(device_type=device.type, enabled=False):
                out = st_transformer(probe_inputs)
            if isinstance(out, dict) and "token_embeddings" in out:
                probe_lhs = out["token_embeddings"]
            elif isinstance(out, dict) and "last_hidden_state" in out:
                probe_lhs = out["last_hidden_state"]
            else:
                probe_lhs = out.last_hidden_state  # type: ignore[attr-defined]
        else:
            probe_lhs = probe_out.last_hidden_state  # type: ignore[union-attr]
        hidden_size = int(probe_lhs.size(-1))
    except Exception:
        pass

    if use_proj:
        if not hidden_size:
            raise RuntimeError("projection_head enabled but hidden_size inference failed")
        if proj_out <= 0:
            proj_out = hidden_size
        act = nn.Tanh() if proj_act == "tanh" else nn.ReLU()
        class ProjectionHead(nn.Module):
            def __init__(self, in_dim: int, out_dim: int, activation: nn.Module):
                super().__init__()
                self.linear = nn.Linear(in_dim, out_dim)
                self.act = activation
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.act(self.linear(x))
        proj_head = ProjectionHead(hidden_size, proj_out, act).to(device)
        log.info(f"[Train] projection head enabled: {hidden_size}->{proj_out} act={proj_act}")

    # -------------------- Optim/Scheduler --------------------
    optim_cfg = cfg.get("optim", {})
    lr = float(optim_cfg.get("lr", 5e-5))
    weight_decay = float(optim_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(optim_cfg.get("warmup_ratio", 0.05))
    # Optional gradient checkpointing
    use_gc = bool(tr.get("grad_checkpointing", False))
    if use_gc:
        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            # ST path: also try underlying auto_model
            try:
                from sentence_transformers import SentenceTransformer  # noqa: F401
                am = getattr(st_transformer, "auto_model", None) if 'st_transformer' in locals() else None
                if am is not None and hasattr(am, "gradient_checkpointing_enable"):
                    am.gradient_checkpointing_enable()
            except Exception:
                pass
            log.info("[Train] gradient checkpointing enabled")
        except Exception as e:
            log.warning(f"[Train] failed to enable gradient checkpointing: {e}")

    # Collect trainable parameters (include projection head if any)
    params = list(p for p in model.parameters() if p.requires_grad)
    if proj_head is not None:
        params += list(proj_head.parameters())
    optimizer = build_optimizer(optim_cfg.get("name", "adamw"), params, lr, weight_decay)
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

            loss = None
            acc = None

            # Call .forward explicitly to bypass any __call__ with no_grad wrappers
            # Define unified forward for both AutoModel and ST-transformer module
            def _fwd(mod, enc):
                if st_transformer is not None and mod is forward_target:
                    # Disable AMP for ST transformer to avoid dtype mismatch inside custom modules
                    with torch.amp.autocast(device_type=device.type, enabled=False):
                        out = st_transformer(enc)
                    if isinstance(out, dict) and "token_embeddings" in out:
                        return out["token_embeddings"]  # type: ignore[index]
                    if isinstance(out, dict) and "last_hidden_state" in out:
                        return out["last_hidden_state"]  # type: ignore[index]
                    if hasattr(out, "last_hidden_state"):
                        return out.last_hidden_state  # type: ignore[attr-defined]
                    raise RuntimeError("Unexpected ST transformer output shape")
                fwd = getattr(mod, "forward", mod)
                out = fwd(**enc, return_dict=True)
                return out.last_hidden_state
            with torch.enable_grad():
                # Forward under AMP
                with amp_ctx:
                    lhs_a = _fwd(forward_target, enc_a)
                    lhs_p = _fwd(forward_target, enc_p)
                    enc_n = batch.get("negative_inputs") if use_hard_negs else None
                    lhs_n = None
                    if enc_n is not None:
                        lhs_n = _fwd(forward_target, enc_n)

                if not (getattr(lhs_a, "requires_grad", False) and getattr(lhs_p, "requires_grad", False)):
                    raise RuntimeError(
                        "Forward produced non-grad tensors (no grad_fn).\n"
                        f"Hint: If using Jina v3, set hf.loader='sentence_transformer', hf.trust_remote_code=true, hf.default_task='classification'.\n"
                        "Or set hf.trust_remote_code=false to load base AutoModel.\n"
                        "Also try clearing dynamic module cache and upgrading transformers. See README troubleshooting."
                    )

                # Pooling and contrastive loss outside AMP (float32 math for stability)
                if pooling_method == "cls":
                    z_a = cls_pool(lhs_a, enc_a["attention_mask"])  # [B, H]
                    z_p = cls_pool(lhs_p, enc_p["attention_mask"])  # [B, H]
                elif pooling_method == "last":
                    z_a = last_token_pool(lhs_a, enc_a["attention_mask"])  # [B, H]
                    z_p = last_token_pool(lhs_p, enc_p["attention_mask"])  # [B, H]
                else:
                    z_a = mean_pool(lhs_a, enc_a["attention_mask"])  # [B, H]
                    z_p = mean_pool(lhs_p, enc_p["attention_mask"])  # [B, H]

                if proj_head is not None:
                    z_a = proj_head(z_a)
                    z_p = proj_head(z_p)
                # Hard-negative aware loss
                if use_hard_negs and lhs_n is not None and batch.get("neg_ptrs") is not None:
                    # Normalize for cosine similarity in float32
                    z_a_f = torch.nn.functional.normalize(z_a.float(), dim=-1, eps=1e-6)
                    z_p_f = torch.nn.functional.normalize(z_p.float(), dim=-1, eps=1e-6)
                    z_n = mean_pool(lhs_n, enc_n["attention_mask"])  # type: ignore[index]
                    if proj_head is not None:
                        z_n = proj_head(z_n)
                    z_n_f = torch.nn.functional.normalize(z_n.float(), dim=-1, eps=1e-6)

                    neg_ptrs = batch["neg_ptrs"]
                    losses = []
                    hits = 0
                    start_global = 0
                    # Build per-anchor logits: [1 + k]
                    for i, (start, cnt) in enumerate(neg_ptrs):
                        if cnt == 0:
                            # Fall back to in-batch negatives if none provided
                            sim_row = (z_a_f[i] @ z_p_f.t()) / float(temperature)
                            label = torch.tensor(i, device=sim_row.device)
                            losses.append(torch.nn.functional.cross_entropy(sim_row.unsqueeze(0), label.unsqueeze(0)))
                            hits += int(sim_row.argmax().item() == i)
                            continue
                        pos_sim = (z_a_f[i] * z_p_f[i]).sum(0, keepdim=True)  # [1]
                        neg_sims = z_a_f[i].unsqueeze(0) @ z_n_f[start : start + cnt].t()  # [1, k]
                        logits_i = torch.cat([pos_sim, neg_sims.squeeze(0)], dim=0) / float(temperature)  # [1+k]
                        label_i = torch.tensor(0, device=logits_i.device)
                        losses.append(torch.nn.functional.cross_entropy(logits_i.unsqueeze(0), label_i.unsqueeze(0)))
                        hits += int(logits_i.argmax().item() == 0)
                    loss = torch.stack(losses).mean()
                    acc = torch.tensor(hits / max(1, len(neg_ptrs)), device=z_a.device)
                else:
                    loss, acc = info_nce_in_batch(z_a, z_p, temperature=temperature)
                loss = loss / grad_acc_steps

            # Backprop on full-precision scaler-less; AMP is only for forward
            loss.backward()
            accum_loss += loss.item()

            if (step % grad_acc_steps) == 0:
                # Gradient clipping (optional)
                if clip_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=clip_grad_norm)
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

    # Save projection head (if used)
    if proj_head is not None:
        torch.save({
            "state_dict": proj_head.state_dict(),
            "in_dim": hidden_size,
            "out_dim": proj_out,
            "activation": proj_act,
        }, target / "projection_head.pt")

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
            "pooling": pooling_method,
            "temperature": temperature,
            "clip_grad_norm": clip_grad_norm,
            "freeze_base": freeze_base,
            "projection_head": {"enabled": bool(proj_head is not None), "out_dim": proj_out, "activation": proj_act} if proj_head is not None else {"enabled": False},
        }
        write_json(target / "metrics.json", metrics, "metrics")
        write_json(target / "config.snapshot.json", cfg, "config snapshot")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="General Embedding Fine-tuner")
    ap.add_argument("--config", type=str, default=None, help="config file path (default: ./config.json)")
    ap.add_argument("--probe-only", action="store_true", help="Only run a single forward probe and exit")
    args = ap.parse_args()

    default_cfg = Path(__file__).parent / "config.json"
    cfg_path = Path(args.config) if args.config else default_cfg

    if not cfg_path.exists():
        print(f"[ERROR] config not found: {cfg_path}")
        sys.exit(1)

    if args.probe_only:
        # Minimal config-run to check grad availability
        try:
            # Temporarily patch config to run for 1 step and 1 epoch
            cfg = load_json(cfg_path)
            cfg.setdefault("train", {})
            cfg["train"].update({"epochs": 1, "steps_per_epoch": 1, "log_every": 1})
            tmp_cfg = Path(".tmp.probe.config.json")
            write_json(tmp_cfg, cfg)
            print("[probe] Running a single forward step to verify grads...")
            main(tmp_cfg)
        finally:
            try:
                Path(".tmp.probe.config.json").unlink(missing_ok=True)  # type: ignore[call-arg]
            except Exception:
                pass
    else:
        main(cfg_path)
