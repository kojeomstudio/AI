"""
trainer.py
----------
- Trainer class that encapsulates the entire training process.
- Manages model, tokenizer, data loaders, optimizer, and the main training loop.
"""
import logging
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import InputExample
from tqdm import tqdm

from .model import EmbeddingModel

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config: dict, train_samples: list[InputExample]):
        self.config = config
        self.train_samples = train_samples
        
        train_cfg = self.config.get("train", {})
        self.device, self.dtype, self.amp_ctx = self._setup_device_dtype()
        
        self.model = EmbeddingModel(
            model_id=config.get("model_id"),
            hf_config=config.get("hf", {}),
            pooling_method=train_cfg.get("pooling", "mean"),
            device=self.device,
            dtype=self.dtype
        )
        
        self.train_dataloader = self._create_dataloader()
        self.loss_fn = self._create_loss_function()

    def _setup_device_dtype(self):
        # Simplified device and dtype setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        amp_ctx = torch.amp.autocast(device_type=device.type, dtype=dtype)
        logger.info(f"Device: {device}, DType: {dtype}")
        return device, dtype, amp_ctx

    def _create_dataloader(self) -> DataLoader:
        train_cfg = self.config.get("train", {})
        batch_size = train_cfg.get("batch_size", 32)
        return DataLoader(self.train_samples, shuffle=True, batch_size=batch_size)

    def _create_loss_function(self):
        # Using a robust, standard loss function from sentence-transformers
        return MultipleNegativesRankingLoss(self.model.st_model)

    def train(self):
        train_cfg = self.config.get("train", {})
        optim_cfg = self.config.get("optim", {})
        
        epochs = train_cfg.get("epochs", 1)
        lr = optim_cfg.get("lr", 2e-5)
        
        self.model.st_model.to(self.device)
        
        # sentence-transformers recommends its own training loop structure
        self.model.st_model.fit(
            train_objectives=[(self.train_dataloader, self.loss_fn)],
            epochs=epochs,
            optimizer_params={'lr': lr},
            warmup_steps=int(len(self.train_dataloader) * 0.1),
            output_path=self.config.get("output", {}).get("save_dir", "./outputs/embed-ft-new"),
            show_progress_bar=True,
            checkpoint_save_steps=train_cfg.get("log_every", 100),
            checkpoint_path=f'{self.config.get("output", {}).get("save_dir", "./outputs/embed-ft-new")}_checkpoints'
        )
        
        logger.info("Training complete.")
