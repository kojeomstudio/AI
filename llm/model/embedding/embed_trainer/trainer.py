"""
trainer.py
----------
- Trainer class that encapsulates the entire training process.
- Manages model, tokenizer, data loaders, optimizer, and the main training loop.
"""
import logging
import torch
from torch.utils.data import DataLoader
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import InputExample
from tqdm import tqdm
from pathlib import Path

from .model import EmbeddingModel
from .utils import ensure_dir

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config: dict, train_samples: list[InputExample]):
        self.config = config
        
        train_cfg = self.config.get("train", {})
        device_cfg = self.config.get("device", {})

        self.device, self.dtype, self.amp_ctx = self._setup_device_dtype(device_cfg)
        self.fp16_enabled = self.dtype == torch.float16 and train_cfg.get("fp16", True)

        self.model = EmbeddingModel(
            model_id=config.get("model_id"),
            hf_config=config.get("hf", {}),
            pooling_method=train_cfg.get("pooling", "mean"),
            device=self.device,
            dtype=self.dtype
        )
        
        self.train_dataloader = self._create_dataloader(train_samples, train_cfg)
        self.loss_fn = self._create_loss_function()

    def _setup_device_dtype(self, device_cfg: dict):
        # Device selection
        device_str = device_cfg.get("device", "auto").lower()
        if device_str == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif device_str == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Dtype selection
        dtype_str = device_cfg.get("dtype", "auto").lower()
        if dtype_str == "float32":
            dtype = torch.float32
        elif dtype_str == "float16":
            dtype = torch.float16
        elif dtype_str == "bfloat16":
            dtype = torch.bfloat16
        else: # auto
            if device.type == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif device.type == "cuda":
                dtype = torch.float16
            elif device.type == "mps":
                dtype = torch.float16
            else:
                dtype = torch.float32
        
        # Autocast context
        if device.type == "cuda":
            amp_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
        elif device.type == "mps":
            amp_ctx = torch.amp.autocast(device_type="mps", dtype=dtype)
        else:
            from contextlib import nullcontext
            amp_ctx = nullcontext()

        logger.info(f"[Runtime] device={device} dtype={dtype}")
        return device, dtype, amp_ctx

    def _create_dataloader(self, train_samples: list[InputExample], train_cfg: dict) -> DataLoader:
        batch_size = train_cfg.get("batch_size", 32)
        num_workers = train_cfg.get("num_workers", 0)
        return DataLoader(
            train_samples,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda" and num_workers > 0)
        )

    def _create_loss_function(self):
        return MultipleNegativesRankingLoss(self.model.st_model)

    def train(self):
        train_cfg = self.config.get("train", {})
        optim_cfg = self.config.get("optim", {})
        output_cfg = self.config.get("output", {})
        
        epochs = train_cfg.get("epochs", 1)
        lr = optim_cfg.get("lr", 2e-5)
        grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
        warmup_ratio = optim_cfg.get("warmup_ratio", 0.1)
        
        save_dir = Path(output_cfg.get("save_dir", "./outputs")).expanduser().resolve()
        save_name = output_cfg.get("save_name", "embed-ft")
        output_path = ensure_dir(save_dir / save_name)
        checkpoint_path = ensure_dir(output_path / "checkpoints")

        # Calculate total steps for scheduler and warmup
        total_steps = len(self.train_dataloader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)

        self.model.st_model.to(self.device)
        
        logger.info(f"[Train] Starting training for {epochs} epochs, total steps: {total_steps}")
        logger.info(f"[Train] Effective batch size: {train_cfg.get("batch_size", 32) * grad_accum_steps}")

        self.model.st_model.fit(
            train_objectives=[(self.train_dataloader, self.loss_fn)],
            epochs=epochs,
            optimizer_params={'lr': lr},
            warmup_steps=warmup_steps,
            output_path=str(output_path),
            checkpoint_path=str(checkpoint_path),
            show_progress_bar=True,
            callback=None, # Optional: Add a callback for custom logging/metrics

        )
        
        logger.info("Training complete.")