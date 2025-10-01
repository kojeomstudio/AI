"""
train.py
---------
Main entry point for the embedding fine-tuner.

This script reads a configuration file, loads the training data, and
invokes the Trainer class to perform the actual training.
"""
import argparse
import logging
import sys
from pathlib import Path

from utils import load_json
from data_utils import load_contrastive_pairs, infer_pairs_path
from trainer import Trainer

def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("embed_trainer")

def main(cfg_path: Path):
    log = setup_logger()
    base_dir = Path(__file__).resolve().parent
    
    log.info(f"Loading configuration from: {cfg_path}")
    cfg = load_json(cfg_path)

    # --- Load Data ---
    data_cfg = cfg.get("data", {})
    pairs_path_str = data_cfg.get("pairs_path")
    
    if pairs_path_str:
        pairs_path = Path(pairs_path_str)
    else:
        pairs_path = infer_pairs_path(base_dir)

    if not pairs_path or not pairs_path.exists():
        log.error("Could not find pairs.jsonl. Please specify `data.pairs_path` in your config.")
        sys.exit(1)

    log.info(f"Loading data from: {pairs_path}")
    train_samples = load_contrastive_pairs(pairs_path)

    if not train_samples:
        log.error("No training samples were loaded. Aborting.")
        sys.exit(1)

    # --- Initialize and Run Trainer ---
    trainer = Trainer(config=cfg, train_samples=train_samples)
    trainer.train()

    log.info("Fine-tuning process completed successfully.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="General Embedding Fine-tuner")
    ap.add_argument("--config", type=str, default="config.json", help="Path to the configuration file (default: config.json)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    main(cfg_path)