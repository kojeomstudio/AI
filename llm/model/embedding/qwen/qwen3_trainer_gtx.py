# -*- coding: utf-8 -*-
"""
qwen3_trainer_gtx.py - GTX GPU 최적화 버전
--------------------------------------------------------------------------------
- CUDA GPU (GTX 시리즈) 호환성 개선
- CUDA OOM 방지: 동적 배치 크기 조정, 메모리 최적화
- Mixed Precision Training (AMP) 지원
- GPU 메모리 모니터링 및 자동 정리
- 그래디언트 체크포인팅으로 메모리 절약
- 더 나은 오류 처리 및 복구 메커니즘
"""

import os
import json
import logging
from pathlib import Path
from contextlib import nullcontext
from datetime import datetime
import gc

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast

try:
    from transformers import Adafactor
    _HAVE_ADAFACTOR = True
except Exception:
    _HAVE_ADAFACTOR = False

from train_helpers import make_dataloader, infer_data_paths

# ------------------------ Logging ------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("trainer_gtx")

# ------------------------ GPU Memory Management ------------------------
def get_gpu_memory_info():
    """GPU 메모리 사용량 정보 반환"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        cached = torch.cuda.memory_reserved() / (1024**3)  # GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        return {
            'allocated': allocated,
            'cached': cached,
            'total': total,
            'free': total - allocated
        }
    return None

def clear_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def log_gpu_memory(prefix=""):
    """GPU 메모리 사용량 로깅"""
    info = get_gpu_memory_info()
    if info:
        log.info(f"{prefix}GPU Memory - Allocated: {info['allocated']:.2f}GB, "
                f"Cached: {info['cached']:.2f}GB, Free: {info['free']:.2f}GB")

def auto_adjust_batch_size(initial_batch_size, model, tokenizer, max_length, device):
    """메모리 상황에 따라 자동으로 배치 크기 조정"""
    batch_size = initial_batch_size
    while batch_size >= 1:
        try:
            # 테스트 배치 생성
            test_inputs = {
                'input_ids': torch.randint(1, 1000, (batch_size, max_length)).to(device),
                'attention_mask': torch.ones((batch_size, max_length)).to(device)
            }
            
            # 포워드 패스 테스트
            with autocast():
                model(**test_inputs)
            
            clear_gpu_memory()
            log.info(f"[Memory] Optimal batch size determined: {batch_size}")
            return batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                batch_size = max(1, batch_size // 2)
                clear_gpu_memory()
                log.warning(f"[Memory] OOM detected, reducing batch size to: {batch_size}")
                continue
            else:
                raise e
    
    return 1

# ------------------------ Config & Utilities ------------------------
def load_config(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json을 찾지 못했습니다: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    log.info(f"[Config] loaded {cfg_path}")
    return cfg

def resolve_device_and_dtype(cfg: dict):
    """디바이스와 dtype 결정 (GTX GPU 우선)"""
    prefer_cuda = cfg.get("device", {}).get("prefer_cuda", True)
    use_mixed_precision = cfg.get("device", {}).get("use_mixed_precision", True)
    
    if torch.cuda.is_available() and prefer_cuda:
        device = torch.device("cuda")
        # GTX 시리즈는 일반적으로 FP16 지원이 제한적이므로 FP32 기본 사용
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "gtx" in gpu_name and ("10" in gpu_name or "9" in gpu_name):
            # GTX 10xx, 9xx 시리즈는 FP16 지원 제한
            dtype = torch.float32
            use_mixed_precision = False
            log.warning(f"[Device] {gpu_name} detected - using FP32 for compatibility")
        else:
            dtype = torch.float16 if use_mixed_precision else torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        use_mixed_precision = False
    
    log.info(f"[Device] {device}, dtype={dtype}, mixed_precision={use_mixed_precision}")
    return device, dtype, use_mixed_precision

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path: Path, payload: dict, title: str):
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"[Save] wrote {title}: {path}")

# ------------------------ Model Utilities ------------------------
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    s = (last_hidden_state * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1e-6)
    return s / d

def info_nce_in_batch(z_a, z_p, temperature=0.07):
    z_a = F.normalize(z_a, dim=-1)
    z_p = F.normalize(z_p, dim=-1)
    logits = torch.matmul(z_a, z_p.t()) / temperature
    labels = torch.arange(z_a.size(0), device=z_a.device)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, acc

def _to_device(d, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in d.items()}

def setup_model_for_training(model, device, enable_gradient_checkpointing=True):
    """모델을 훈련용으로 설정"""
    model.to(device)
    model.train()
    
    # 그래디언트 체크포인팅 활성화 (메모리 절약)
    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        log.info("[Model] gradient checkpointing enabled")
    
    # 캐시 비활성화
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
        log.info("[Model] model.config.use_cache=False")
    
    return model

# ------------------------ Main Training Loop ------------------------
def main(cfg_path: str):
    # Load config
    cfg = load_config(Path(cfg_path))
    
    # Device & dtype resolution
    device, dtype, use_mixed_precision = resolve_device_and_dtype(cfg)
    
    # GPU 메모리 정보 로깅
    if device.type == "cuda":
        log_gpu_memory("[Init] ")
    
    # Data paths
    pairs_path_cfg = cfg.get("data", {}).get("pairs_path")
    if pairs_path_cfg and Path(pairs_path_cfg).expanduser().exists():
        pairs_path = Path(pairs_path_cfg).expanduser().resolve()
    else:
        pairs_path, _ = infer_data_paths(Path(__file__).resolve().parent)
    assert pairs_path is not None and Path(pairs_path).exists(), "pairs.jsonl 경로를 찾을 수 없습니다."
    log.info(f"[Data] pairs: {pairs_path}")
    
    # Model & Tokenizer
    log.info(f"[Model] Loading: {cfg.get('model_id')}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.get("model_id"),
        use_fast=True,
        token=cfg.get("hf_token"),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
    )
    
    # 메모리 효율적인 모델 로딩
    model = AutoModel.from_pretrained(
        cfg.get("model_id"),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,  # 메모리 효율성을 위해 True로 변경
    )
    
    # 모델 설정
    model = setup_model_for_training(model, device, enable_gradient_checkpointing=True)
    
    # 메모리 정리
    clear_gpu_memory()
    if device.type == "cuda":
        log_gpu_memory("[Model Loaded] ")
    
    # 자동 배치 크기 조정
    initial_batch_size = int(cfg.get("train", {}).get("batch_size", 4))
    if device.type == "cuda":
        max_length = int(cfg.get("data", {}).get("max_length", 64))
        optimal_batch_size = auto_adjust_batch_size(
            initial_batch_size, model, tokenizer, max_length, device
        )
    else:
        optimal_batch_size = initial_batch_size
    
    # DataLoader with optimal batch size
    train_loader = make_dataloader(
        pairs_path=pairs_path,
        tokenizer_name_or_obj=cfg.get("model_id"),
        batch_size=optimal_batch_size,
        shuffle=True,
        max_length=int(cfg.get("data", {}).get("max_length", 64)),
        return_negatives=False,
        return_tuple=False,
        device=device,
        hf_token=cfg.get("hf_token"),
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        local_files_only=False,
    )
    
    steps_per_epoch = max(1, len(train_loader))
    total_steps = int(cfg.get("train", {}).get("epochs", 1)) * steps_per_epoch
    log.info(f"[Train] batch_size={optimal_batch_size}, steps/epoch={steps_per_epoch}, "
             f"epochs={cfg.get('train', {}).get('epochs', 1)}, total_steps={total_steps}")
    
    # Optimizer & Scheduler
    opt_name = str(cfg.get("train", {}).get("optimizer", "adamw")).lower()
    lr = float(cfg.get("train", {}).get("lr", 2e-5))
    weight_decay = float(cfg.get("train", {}).get("weight_decay", 0.01))
    warmup_ratio = float(cfg.get("train", {}).get("warmup_ratio", 0.06))
    epochs = int(cfg.get("train", {}).get("epochs", 1))
    grad_acc_steps = int(cfg.get("train", {}).get("grad_acc_steps", 8))
    log_every = int(cfg.get("train", {}).get("log_every", 10))
    
    # GTX GPU에서는 AdamW가 더 안정적
    if device.type == "cuda" and "gtx" in torch.cuda.get_device_name(0).lower():
        if opt_name == "adafactor":
            log.warning("[Optim] GTX GPU detected - using AdamW for better stability")
            opt_name = "adamw"
    
    def build_optimizer(name: str, params):
        if name == "adamw":
            return AdamW(params, lr=lr, weight_decay=weight_decay, eps=1e-8)
        if name == "adafactor" and _HAVE_ADAFACTOR:
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
        log.warning(f"[Optim] Unknown optimizer '{name}', using AdamW")
        return AdamW(params, lr=lr, weight_decay=weight_decay, eps=1e-8)
    
    optimizer = build_optimizer(opt_name, model.parameters())
    
    # Scheduler setup
    for g in optimizer.param_groups:
        if g.get("lr", None) is None:
            g["lr"] = lr
        g["initial_lr"] = float(g["lr"])
    
    num_train_steps = epochs * steps_per_epoch
    num_warmup = max(1, int(num_train_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, num_train_steps)
    
    # Mixed Precision Scaler
    scaler = GradScaler() if use_mixed_precision and device.type == "cuda" else None
    
    log.info(f"[Optim] {opt_name} | lr={lr} | weight_decay={weight_decay} | "
             f"warmup={num_warmup}/{num_train_steps} | grad_acc_steps={grad_acc_steps}")
    
    # ------------------------ Training Loop ------------------------
    global_step = 0
    accum_loss = 0.0
    best_loss = float('inf')
    optimizer.zero_grad(set_to_none=True)
    
    log.info("[Train] Starting training loop")
    
    try:
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            
            for step, batch in enumerate(train_loader, start=1):
                try:
                    enc_a = _to_device(batch["anchor_inputs"], device)
                    enc_p = _to_device(batch["positive_inputs"], device)
                    
                    # Mixed precision forward pass
                    if use_mixed_precision and scaler is not None:
                        with autocast():
                            out_a = model(**enc_a, return_dict=True)
                            out_p = model(**enc_p, return_dict=True)
                            
                            # Embedding extraction
                            z_a = (out_a.pooler_output if hasattr(out_a, "pooler_output") and out_a.pooler_output is not None 
                                   else mean_pool(out_a.last_hidden_state, enc_a["attention_mask"]))
                            z_p = (out_p.pooler_output if hasattr(out_p, "pooler_output") and out_p.pooler_output is not None 
                                   else mean_pool(out_p.last_hidden_state, enc_p["attention_mask"]))
                            
                            loss, acc = info_nce_in_batch(z_a, z_p, temperature=0.07)
                            loss = loss / grad_acc_steps
                        
                        # Backward pass with scaling
                        scaler.scale(loss).backward()
                        
                    else:
                        # Standard precision
                        out_a = model(**enc_a, return_dict=True)
                        out_p = model(**enc_p, return_dict=True)
                        
                        z_a = (out_a.pooler_output if hasattr(out_a, "pooler_output") and out_a.pooler_output is not None 
                               else mean_pool(out_a.last_hidden_state, enc_a["attention_mask"]))
                        z_p = (out_p.pooler_output if hasattr(out_p, "pooler_output") and out_p.pooler_output is not None 
                               else mean_pool(out_p.last_hidden_state, enc_p["attention_mask"]))
                        
                        loss, acc = info_nce_in_batch(z_a, z_p, temperature=0.07)
                        loss = loss / grad_acc_steps
                        loss.backward()
                    
                    accum_loss += loss.item()
                    epoch_loss += loss.item()
                    epoch_acc += acc.item()
                    
                    # Gradient accumulation step
                    if (step % grad_acc_steps) == 0:
                        if use_mixed_precision and scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        
                        # Memory cleanup
                        if device.type == "cuda" and global_step % 10 == 0:
                            clear_gpu_memory()
                        
                        # Logging
                        if (global_step % log_every) == 0 or step == steps_per_epoch:
                            avg_loss = accum_loss
                            current_lr = scheduler.get_last_lr()[0] if scheduler else lr
                            
                            log.info(f"epoch {epoch+1}/{epochs} | step {step}/{steps_per_epoch} | "
                                   f"gstep {global_step} | loss={avg_loss:.4f} | acc={acc.item():.3f} | "
                                   f"lr={current_lr:.2e}")
                            
                            if device.type == "cuda" and global_step % (log_every * 2) == 0:
                                log_gpu_memory(f"[Step {global_step}] ")
                            
                            accum_loss = 0.0
                            
                            # Best model tracking
                            if avg_loss < best_loss:
                                best_loss = avg_loss
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        log.error(f"[OOM] Step {step}: {str(e)}")
                        clear_gpu_memory()
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    else:
                        raise e
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_epoch_acc = epoch_acc / len(train_loader)
            log.info(f"[Epoch {epoch+1}] Completed - avg_loss={avg_epoch_loss:.4f}, avg_acc={avg_epoch_acc:.3f}")
        
        log.info("[OK] Training completed successfully!")
        
    except KeyboardInterrupt:
        log.info("[Interrupted] Training stopped by user")
    except Exception as e:
        log.error(f"[Error] Training failed: {str(e)}")
        raise
    
    # ------------------------ Save Model ------------------------
    out_cfg = cfg.get("output", {})
    save_dir = ensure_dir(Path(out_cfg.get("save_dir", "./outputs")).expanduser().resolve())
    save_name = out_cfg.get("save_name", "model-ft-gtx")
    target_dir = ensure_dir(save_dir / save_name)
    
    log.info(f"[Save] Saving model & tokenizer to: {target_dir}")
    
    try:
        # CPU로 이동 후 저장 (메모리 절약)
        model.cpu()
        clear_gpu_memory()
        
        model.save_pretrained(target_dir)
        tokenizer.save_pretrained(target_dir)
        
        # Save training metrics
        if bool(out_cfg.get("save_metrics", True)):
            gpu_info = get_gpu_memory_info()
            metrics = {
                "finished_at": datetime.utcnow().isoformat() + "Z",
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": num_train_steps,
                "batch_size": optimal_batch_size,
                "original_batch_size": initial_batch_size,
                "grad_acc_steps": grad_acc_steps,
                "lr": lr,
                "optimizer": opt_name,
                "device": str(device),
                "dtype": str(dtype),
                "use_mixed_precision": use_mixed_precision,
                "model_id": cfg.get("model_id"),
                "best_loss": best_loss,
                "gpu_info": gpu_info,
            }
            write_json(target_dir / "metrics.json", metrics, "metrics")
            write_json(target_dir / "config.snapshot.json", cfg, "config snapshot")
        
        log.info("[Save] Model and metrics saved successfully!")
        
    except Exception as e:
        log.error(f"[Save Error] Failed to save model: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse, sys
    
    ap = argparse.ArgumentParser(description="Qwen3 Embedding Training - GTX GPU Optimized")
    ap.add_argument("--config", type=str, default=None, 
                    help="Config file path (default: ./config_gtx.json)")
    args = ap.parse_args()
    
    default_cfg = os.path.join(os.path.dirname(__file__), "config_gtx.json")
    cfg_path = args.config or default_cfg
    
    if not os.path.exists(cfg_path):
        print(f"[ERROR] Config file not found: {cfg_path}")
        print("       Please create config_gtx.json or specify path with --config")
        sys.exit(1)
    
    main(cfg_path)