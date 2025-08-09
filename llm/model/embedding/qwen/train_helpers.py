# -*- coding: utf-8 -*-
"""
train_helpers.py (MPS/CUDA/CPU 안전, 토크나이저 토큰/옵션 지원)

기능
- load_pairs_jsonl(path)
- infer_data_paths(start_path)
- PairJsonlDataset(rows)
- ContrastiveCollator(...): device로 텐서 이동, tuple/dict 모드 지원
- make_dataloader(...): 안전 가드, 헬퍼 인자 전달
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    raise RuntimeError("PyTorch가 필요합니다. pip install torch") from e

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


# --------------------------- I/O ---------------------------
def load_pairs_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"pairs 파일을 찾을 수 없습니다: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def infer_data_paths(any_project_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    candidates = []
    for rel in ["tools/data", "data", "../tools/data", "../data", "../../tools/data"]:
        p = (any_project_path / rel).resolve()
        if p.exists():
            candidates.append(p)
    root = any_project_path.resolve()
    for parent in [root, *root.parents]:
        td = parent / "tools" / "data"
        if td.exists():
            candidates.append(td)
        d = parent / "data"
        if d.exists():
            candidates.append(d)
    seen = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
    pairs_path = None
    csv_path = None
    for c in seen:
        if (c / "pairs.jsonl").exists():
            pairs_path = c / "pairs.jsonl"
        if (c / "game_glossary.csv").exists():
            csv_path = c / "game_glossary.csv"
    return pairs_path, csv_path


# --------------------------- Dataset ---------------------------
class PairJsonlDataset(Dataset):
    """
    JSONL 예:
    {"anchor": str, "positive": str, "hard_negatives": [str, ...]?}
    """
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


# --------------------------- Collator ---------------------------
@dataclass
class ContrastiveCollator:
    tokenizer_name_or_obj: Any
    max_length: int = 128
    return_negatives: bool = True
    return_tuple: bool = False
    device: Optional["torch.device"] = None
    hf_token: Optional[str] = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.tokenizer_name_or_obj, str):
            if AutoTokenizer is None:
                raise RuntimeError("transformers가 필요합니다.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_obj,
                use_fast=True,
                token=self.hf_token,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
            )
        else:
            self.tokenizer = self.tokenizer_name_or_obj

    def _to_device(self, batch_encoding):
        if self.device is None:
            return batch_encoding
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch_encoding.items()}

    def __call__(self, batch: List[Dict[str, Any]]):
        anchors = [b["anchor"] for b in batch]
        positives = [b["positive"] for b in batch]

        negatives = []
        if self.return_negatives:
            for b in batch:
                negs = b.get("negatives") or []
                if negs:
                    negatives.append(negs[0])
                else:
                    # in-batch fallback
                    fallback = positives[0]
                    for cand in positives:
                        if cand != b["positive"]:
                            fallback = cand
                            break
                    negatives.append(fallback)

        tok_args = dict(padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        if self.pad_to_multiple_of is not None:
            tok_args["pad_to_multiple_of"] = self.pad_to_multiple_of

        tok_anchor = self.tokenizer(anchors, **tok_args)
        tok_positive = self.tokenizer(positives, **tok_args)
        tok_anchor = self._to_device(tok_anchor)
        tok_positive = self._to_device(tok_positive)

        if self.return_tuple:
            if self.return_negatives:
                tok_negative = self._to_device(self.tokenizer(negatives, **tok_args))
                return (tok_anchor, tok_positive, tok_negative)
            return (tok_anchor, tok_positive)

        out = {
            "anchor_inputs": tok_anchor,
            "positive_inputs": tok_positive,
            "raw_texts": {"anchors": anchors, "positives": positives},
        }
        if self.return_negatives:
            tok_negative = self._to_device(self.tokenizer(negatives, **tok_args))
            out["negative_inputs"] = tok_negative
            out["raw_texts"]["negatives"] = negatives
        return out


# --------------------------- DataLoader ---------------------------
def make_dataloader(
    pairs_path: Path,
    tokenizer_name_or_obj: Any,
    batch_size: int = 32,
    shuffle: bool = True,
    max_length: int = 128,
    return_negatives: bool = True,
    return_tuple: bool = False,
    device: Optional["torch.device"] = None,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    pad_to_multiple_of: Optional[int] = None,
) -> "DataLoader":
    rows = load_pairs_jsonl(pairs_path)
    effective_bs = min(batch_size, max(1, len(rows)))
    ds = PairJsonlDataset(rows, use_negatives=return_negatives)
    collator = ContrastiveCollator(
        tokenizer_name_or_obj=tokenizer_name_or_obj,
        max_length=max_length,
        return_negatives=return_negatives,
        return_tuple=return_tuple,
        device=device,
        hf_token=hf_token,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    loader = DataLoader(ds, batch_size=effective_bs, shuffle=shuffle, collate_fn=collator, drop_last=False)
    return loader


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    pairs_path, _ = infer_data_paths(here)
    if not pairs_path:
        raise SystemExit("pairs.jsonl을 찾지 못했습니다.")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    loader = make_dataloader(
        pairs_path,
        tokenizer_name_or_obj="Qwen/Qwen3-Embedding-0.6B",
        batch_size=8,
        device=device,
        trust_remote_code=True,
    )
    first = next(iter(loader))
    print("Batch keys:", list(first.keys()))
