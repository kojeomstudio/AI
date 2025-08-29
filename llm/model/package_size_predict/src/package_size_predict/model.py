from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    """Configuration for the linear regression model."""

    in_features: int = 1
    out_features: int = 1


class LinearRegressor(nn.Module):
    """A minimal linear regression model y = Wx + b.

    This uses a single `nn.Linear` layer without activation.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.linear = nn.Linear(cfg.in_features, cfg.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Compute forward pass returning predicted values."""
        return self.linear(x)

