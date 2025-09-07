"""
Public API shortcuts for the embed_trainer package.
Expose commonly used entrypoints for easier imports in user code.
"""

from .train import main as train_main  # noqa: F401
from .eval_utils import (  # noqa: F401
    load_model_and_tokenizer,
    embed_texts,
    compute_retrieval_metrics,
    evaluate_pairs_with_model,
)

__all__ = [
    "train_main",
    "load_model_and_tokenizer",
    "embed_texts",
    "compute_retrieval_metrics",
    "evaluate_pairs_with_model",
]
