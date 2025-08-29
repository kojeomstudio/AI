from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from .data import DatasetPaths, ensure_sample_excel, load_excel_as_xy
from .model import LinearRegressor, ModelConfig


# -------------------------- Logging Utilities -------------------------- #


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure console and file logging.

    Args:
        log_dir: Directory to store log files.

    Returns:
        Configured root logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("package_size_predict")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"training_{ts}.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# --------------------------- Core Train Logic -------------------------- #


@dataclass
class TrainConfig:
    """Training configuration values."""

    epochs: int = 300
    batch_size: int = 32
    lr: float = 1e-2
    val_split: float = 0.2
    seed: int = 42


def prepare_dataloaders(X: torch.Tensor, y: torch.Tensor, cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train/validation DataLoaders from tensors."""
    dataset = TensorDataset(X, y)
    val_len = int(len(dataset) * cfg.val_split)
    train_len = len(dataset) - val_len

    g = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=g)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    """Train for a single epoch and return average loss."""
    model.train()
    total, count = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
        count += xb.size(0)
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> float:
    """Evaluate model and return average loss on the given loader."""
    model.eval()
    total, count = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total += loss.item() * xb.size(0)
        count += xb.size(0)
    return total / max(count, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package size predictor (linear, PyTorch)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory to find/create Excel dataset")
    parser.add_argument("--excel-name", type=str, default="sample.xlsx", help="Excel filename under data directory")
    parser.add_argument("--logs", type=Path, default=Path("logs"), help="Directory to write training logs")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=TrainConfig.lr, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=TrainConfig.val_split, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed, help="Random seed")
    parser.add_argument("--generate-only", action="store_true", help="Only generate sample Excel and exit")

    args = parser.parse_args(argv)

    # Prepare logging first
    logger = setup_logging(args.logs)
    logger.info("Starting training run")

    # Paths and data preparation
    ds_paths = DatasetPaths(root=args.data_dir, excel=args.data_dir / args.excel_name)
    ensure_sample_excel(ds_paths)
    logger.info("Dataset ready at: %s", ds_paths.excel)

    # If only data generation is requested, exit early
    if args.generate_only:
        logger.info("Data generation requested only. Exiting before training.")
        return 0

    # Load data and encode features
    X_series, y_series = load_excel_as_xy(ds_paths)
    n_samples = len(X_series)
    logger.info("Loaded %d rows", n_samples)

    # Convert to tensors with shape [N, 1]
    X = torch.tensor(X_series.to_numpy(), dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y_series.to_numpy(), dtype=torch.float32).unsqueeze(1)

    # Reproducibility
    torch.manual_seed(args.seed)

    # Device (CPU is sufficient for linear model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # DataLoaders
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
    )
    train_loader, val_loader = prepare_dataloaders(X, y, cfg)

    # Model, loss, optimizer
    model = LinearRegressor(ModelConfig()).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Training loop
    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        if val_loss < best_val:
            best_val = val_loss
        if epoch % 10 == 0 or epoch == 1 or epoch == cfg.epochs:
            logger.info(
                "Epoch %4d | train_loss=%.6f | val_loss=%.6f | best_val=%.6f",
                epoch,
                train_loss,
                val_loss,
                best_val,
            )

    # Final log summary
    # Estimate slope and intercept from trained layer for interpretability
    weight = model.linear.weight.detach().cpu().item()
    bias = model.linear.bias.detach().cpu().item()
    logger.info("Training complete. Learned relationship: size ≈ %.4f * day_index + %.4f", weight, bias)

    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution entry
    raise SystemExit(main())
