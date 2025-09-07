from __future__ import annotations
"""
data_utils.py
-------------
pairs.jsonl(contrastive) 데이터셋 로딩 및 토크나이즈 콜레이터/로더 제공.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Iterator
from pathlib import Path
import json
import logging

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

logger = logging.getLogger(__name__)


def load_pairs_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"pairs.jsonl not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows:
        logger.warning("[data] pairs is empty")
    return rows


class PairJsonlDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], use_negatives: bool = True):
        self.rows = rows
        self.use_negatives = use_negatives

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        out = {"anchor": r["anchor"], "positive": r["positive"]}
        if self.use_negatives:
            out["negatives"] = r.get("hard_negatives") or []
        return out


class IterablePairsJsonlDataset(IterableDataset):
    """Streaming-style JSONL reader to avoid loading all pairs in memory.
    Each item is parsed on demand. Suitable for very large datasets.
    """
    def __init__(self, path: Path, use_negatives: bool = True):
        super().__init__()
        self.path = path
        self.use_negatives = use_negatives

    def parse_line(self, line: str) -> Dict[str, Any]:
        r = json.loads(line)
        out = {"anchor": r["anchor"], "positive": r["positive"]}
        if self.use_negatives:
            out["negatives"] = r.get("hard_negatives") or []
        return out

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield self.parse_line(line)


@dataclass
class ContrastiveCollator:
    tokenizer_name_or_obj: Any
    max_length: int = 128
    device: Optional[torch.device] = None
    hf_token: Optional[str] = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    pad_to_multiple_of: Optional[int] = None
    include_negatives: bool = False
    negatives_per_anchor: int = 0

    def __post_init__(self):
        if isinstance(self.tokenizer_name_or_obj, str):
            if AutoTokenizer is None:
                raise RuntimeError("transformers is required for tokenization")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_obj,
                token=self.hf_token,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
                use_fast=True,
            )
        else:
            self.tokenizer = self.tokenizer_name_or_obj

    def _tok(self, texts: List[str]):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

    def __call__(self, batch: List[Dict[str, Any]]):
        anchors = [b["anchor"] for b in batch]
        positives = [b["positive"] for b in batch]
        enc_a = self._tok(anchors)
        enc_p = self._tok(positives)
        if self.device is not None:
            enc_a = {k: v.to(self.device, non_blocking=True) for k, v in enc_a.items()}
            enc_p = {k: v.to(self.device, non_blocking=True) for k, v in enc_p.items()}
        out = {"anchor_inputs": enc_a, "positive_inputs": enc_p}

        # Optional hard negatives per anchor
        if self.include_negatives and self.negatives_per_anchor > 0:
            neg_ptrs = []  # list of (start, count) for each anchor
            neg_texts: List[str] = []
            for r in batch:
                negs = r.get("negatives") or []
                # 샘플링: 앞에서부터 k개 사용(간단). 필요시 무작위 샘플링으로 확장 가능.
                k = min(self.negatives_per_anchor, len(negs))
                start = len(neg_texts)
                neg_texts.extend(negs[:k])
                neg_ptrs.append((start, k))

            if neg_texts:
                enc_n = self._tok(neg_texts)
                if self.device is not None:
                    enc_n = {k: v.to(self.device, non_blocking=True) for k, v in enc_n.items()}
                out["negative_inputs"] = enc_n
                out["neg_ptrs"] = neg_ptrs
            else:
                out["negative_inputs"] = None
                out["neg_ptrs"] = [(0, 0)] * len(batch)

        return out


def infer_pairs_path(base_dir: Path) -> Optional[Path]:
    candidates = [
        base_dir / "tools" / "data" / "pairs.jsonl",
        base_dir / "data" / "pairs.jsonl",
        base_dir.parent / "tools" / "data" / "pairs.jsonl",
        base_dir.parent / "data" / "pairs.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def make_dataloader(
    tokenizer_name_or_obj: Any,
    pairs_path: Path,
    batch_size: int,
    max_length: int,
    device: Optional[torch.device] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    stream: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    include_negatives: bool = False,
    negatives_per_anchor: int = 0,
):
    # Choose dataset mode: in-memory or streaming
    if stream:
        ds = IterablePairsJsonlDataset(pairs_path)
    else:
        rows = load_pairs_jsonl(pairs_path)
        ds = PairJsonlDataset(rows)
    collator = ContrastiveCollator(
        tokenizer_name_or_obj=tokenizer_name_or_obj,
        max_length=max_length,
        device=device,
        hf_token=hf_token,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        pad_to_multiple_of=pad_to_multiple_of,
        include_negatives=include_negatives,
        negatives_per_anchor=negatives_per_anchor,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device is not None and device.type == "cuda"),
        collate_fn=collator,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    return dl


def count_pairs(pairs_path: Path) -> int:
    """Count number of non-empty lines (pairs) in JSONL without loading into memory."""
    n = 0
    with pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n
