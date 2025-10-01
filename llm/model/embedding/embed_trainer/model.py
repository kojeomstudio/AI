"""
model.py
--------
- Defines the EmbeddingModel class, a wrapper around sentence-transformers.
- Handles model and tokenizer loading, and provides a consistent interface for encoding.
"""
import logging
from typing import Optional
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_id: str, hf_config: dict, pooling_method: str, device: torch.device, dtype: torch.dtype):
        self.model_id = model_id
        self.hf_config = hf_config
        self.pooling_method = pooling_method
        self.device = device
        self.dtype = dtype

        self.st_model = self._load_model()
        self._configure_pooling()

    def _load_model(self) -> SentenceTransformer:
        logger.info(f"Loading model: {self.model_id}")
        
        # SentenceTransformer handles remote code and other complexities internally
        model = SentenceTransformer(
            self.model_id,
            device=self.device,
            trust_remote_code=self.hf_config.get("trust_remote_code", True)
        )
        model.to(self.dtype)
        return model

    def _configure_pooling(self):
        # The pooling is typically configured in the sentence-transformers model card
        # but we can log what it's set to.
        pooling_module = self.st_model._last_module()
        if pooling_module.__class__.__name__ == 'Pooling':
            logger.info(f"Using pooling mode: {pooling_module.get_pooling_mode_str()}")
        else:
            logger.warning("Could not determine pooling mode from the model's last module.")

    def get_sentence_transformer_model(self) -> SentenceTransformer:
        return self.st_model
