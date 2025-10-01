"""
data_utils.py
-------------
- Loads pairs.jsonl data for contrastive learning.
- Converts data into sentence_transformers.InputExample format.
"""
import json
import logging
from pathlib import Path
from typing import List
from sentence_transformers import InputExample

logger = logging.getLogger(__name__)

def load_contrastive_pairs(path: Path) -> List[InputExample]:
    """Loads a JSONL file and converts it to a list of InputExample objects.

    Each line in the JSONL should be a dictionary with 'anchor' and 'positive' keys.
    'hard_negatives' are ignored as MultipleNegativesRankingLoss handles negatives implicitly.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            anchor = data.get("anchor")
            positive = data.get("positive")
            if anchor and positive:
                samples.append(InputExample(texts=[anchor, positive]))
            else:
                logger.warning(f"Skipping invalid line: {line.strip()}")

    if not samples:
        logger.warning(f"No valid samples found in {path}. The model will not be trained.")

    return samples

def infer_pairs_path(base_dir: Path) -> Path | None:
    """Infers the path to pairs.jsonl by checking common locations."""
    candidates = [
        base_dir / "tools" / "data" / "pairs.jsonl",
        base_dir / "data" / "pairs.jsonl",
        base_dir.parent / "tools" / "data" / "pairs.jsonl",
        base_dir.parent / "data" / "pairs.jsonl",
        Path("./tools/data/pairs.jsonl"), # Relative path from where script is run
        Path("./data/pairs.jsonl"),
    ]
    for p in candidates:
        if p.exists():
            logger.info(f"Found data file at: {p}")
            return p
    return None