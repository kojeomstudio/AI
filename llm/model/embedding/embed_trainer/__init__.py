"""
Public API shortcuts for the embed_trainer package.
Expose commonly used entrypoints for easier imports in user code.
"""

from .train import main as train_main  # noqa: F401

__all__ = [
    "train_main",
]