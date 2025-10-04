"""
utils.py
--------
Basic utility functions for file I/O.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

def load_json(path: Path) -> Dict[str, Any]:
    """Reads and parses a JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load JSON from {path}: {e}")
        raise

def write_json(path: Path, data: Dict[str, Any], log_label: str = "data") -> None:
    """Serializes a dictionary to a JSON file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"[Save] {log_label} saved to: {path}")
    except IOError as e:
        logger.error(f"Failed to write JSON to {path}: {e}")
        raise

def ensure_dir(p: Path) -> Path:
    """Ensures that the directory for the given path exists."""
    p.mkdir(parents=True, exist_ok=True)
    return p