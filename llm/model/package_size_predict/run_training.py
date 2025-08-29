"""
Convenience entrypoint to train the package size predictor.

Usage (from repo root):
    python run_training.py --epochs 300 --lr 0.01

This will create a sample Excel dataset under `data/sample.xlsx` if it does
not exist, train a linear PyTorch model, log progress to console and write a
timestamped log file under `logs/`.
"""

from pathlib import Path

from src.package_size_predict.train import main


if __name__ == "__main__":
    raise SystemExit(main())

