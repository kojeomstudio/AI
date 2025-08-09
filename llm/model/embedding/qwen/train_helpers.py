# -*- coding: utf-8 -*-
"""
train_helpers.py
- tools/data/{pairs.jsonl, game_glossary.csv}를 참조해 학습에 바로 넣을 수 있는 헬퍼
- Qwen/Qwen3-Embedding-0.6 등 임베딩 모델용 토크나이저를 사용

기능
1) load_pairs_jsonl(path)               : JSONL 로드
2) PairJsonlDataset(rows)               : torch Dataset
3) ContrastiveCollator(tokenizer, ...)  : 앵커/포지티브(+네거티브) 토크나이즈
4) make_dataloader(... )                : 안전 가드 포함 DataLoader 생성
5) infer_data_paths(start_path)         : 상단/하단 경로 탐색(스크린샷 구조 대응)
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
    raise RuntimeError("PyTorch가 필요합니다. pip install torch 로 설치하세요.") from e

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None  # 토크나이저가 꼭 필요하지 않은 경우도 있어 optional 처리


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
    """
    현재 경로를 기준으로 tools/data 또는 data 폴더를 찾아
    pairs.jsonl, game_glossary.csv 경로를 반환.
    """
    candidates = []
    for rel in [
        "tools/data",
        "data",
        "../tools/data",
        "../data",
        "../../tools/data",
    ]:
        p = (any_project_path / rel).resolve()
        if p.exists():
            candidates.append(p)

    for parent in [any_project_path.resolve(), *any_project_path.resolve().parents]:
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
    JSONL 구조 예:
    { "anchor": str, "positive": str, "hard_negatives": [str, ...]? }
    """
    def __init__(self, rows: List[Dict[str, Any]], use_negatives: bool = True):
        self.rows = rows
        self.use_negatives = use_negatives

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        item = {
            "anchor": row["anchor"],
            "positive": row["positive"],
        }
        if self.use_negatives:
            item["negatives"] = row.get("hard_negatives") or []
        return item


# --------------------------- Collator ---------------------------
@dataclass
class ContrastiveCollator:
    tokenizer_name_or_obj: Any
    max_length: int = 128
    return_negatives: bool = True

    def __post_init__(self):
        if isinstance(self.tokenizer_name_or_obj, str):
            if AutoTokenizer is None:
                raise RuntimeError("transformers 라이브러리가 필요합니다.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_obj, use_fast=True)
        else:
            self.tokenizer = self.tokenizer_name_or_obj

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        anchors = [b["anchor"] for b in batch]
        positives = [b["positive"] for b in batch]

        negatives = []
        if self.return_negatives:
            for b in batch:
                negs = b.get("negatives") or []
                if negs:
                    negatives.append(negs[0])
                else:
                    # in-batch negative fallback
                    fallback = positives[0]
                    for cand in positives:
                        if cand != b["positive"]:
                            fallback = cand
                            break
                    negatives.append(fallback)

        tok_anchor = self.tokenizer(anchors, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt")
        tok_positive = self.tokenizer(positives, padding=True, truncation=True,
                                      max_length=self.max_length, return_tensors="pt")

        out = {
            "anchor_inputs": tok_anchor,
            "positive_inputs": tok_positive,
            "raw_texts": {"anchors": anchors, "positives": positives},
        }
        if self.return_negatives and negatives:
            tok_negative = self.tokenizer(negatives, padding=True, truncation=True,
                                          max_length=self.max_length, return_tensors="pt")
            out["negative_inputs"] = tok_negative
            out["raw_texts"]["negatives"] = negatives
        return out


# --------------------------- DataLoader ---------------------------
def make_dataloader(
    pairs_path: Path,
    tokenizer_name_or_obj: Any = "Qwen/Qwen3-Embedding-0.6",
    batch_size: int = 32,
    shuffle: bool = True,
    max_length: int = 128,
    return_negatives: bool = True,
) -> "DataLoader":
    rows = load_pairs_jsonl(pairs_path)
    effective_bs = min(batch_size, max(1, len(rows)))
    ds = PairJsonlDataset(rows, use_negatives=return_negatives)
    collator = ContrastiveCollator(tokenizer_name_or_obj, max_length=max_length, return_negatives=return_negatives)
    loader = DataLoader(ds, batch_size=effective_bs, shuffle=shuffle, collate_fn=collator, drop_last=False)
    return loader


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    pairs_path, csv_path = infer_data_paths(here)
    if not pairs_path:
        raise SystemExit("pairs.jsonl을 찾지 못했습니다.")
    print(f"[OK] Using pairs: {pairs_path}")
    loader = make_dataloader(pairs_path, batch_size=8)
    first = next(iter(loader))
    print({k: list(v.keys()) if hasattr(v, 'keys') else type(v).__name__ for k, v in first.items()})