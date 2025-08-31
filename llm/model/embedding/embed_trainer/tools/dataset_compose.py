#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset composer: converts CSV/TSV lists into pairs.jsonl for embedding training.

Supported inputs
- Glossary-like CSV with columns: term,definition[,lang,source]
- Two-column CSV/TSV: anchor,positive

Outputs
- JSONL pairs: {anchor,positive,hard_negatives?}
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict


def read_table(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        # auto sniff delimiter
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        rdr = csv.DictReader(f, dialect=dialect)
        for r in rdr:
            rows.append({k.strip(): (v or "").strip() for k, v in r.items()})
    return rows


def to_pairs(rows: List[Dict], hard_k: int = 3) -> List[Dict]:
    pairs: List[Dict] = []
    if not rows:
        return pairs

    # Try glossary format first
    if set([c.lower() for c in rows[0].keys()]) >= {"term", "definition"}:
        defs = [r["definition"] for r in rows]
        for i, r in enumerate(rows):
            anchor = r["term"]
            pos = r["definition"]
            negs = [defs[(i + j) % len(defs)] for j in range(1, hard_k + 1)]
            pairs.append({"anchor": anchor, "positive": pos, "hard_negatives": negs})
        return pairs

    # Fallback: two-column
    # normalize common header variants
    header = [h.lower() for h in rows[0].keys()]
    a_key = "anchor" if "anchor" in header else list(rows[0].keys())[0]
    p_key = "positive" if "positive" in header else list(rows[0].keys())[1]
    positives = [r[p_key] for r in rows]
    for i, r in enumerate(rows):
        anchor = r[a_key]
        pos = r[p_key]
        negs = [positives[(i + j) % len(positives)] for j in range(1, hard_k + 1)]
        pairs.append({"anchor": anchor, "positive": pos, "hard_negatives": negs})
    return pairs


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compose pairs.jsonl from CSV/TSV")
    p.add_argument("--input", type=Path, required=True, help="Input CSV/TSV path")
    p.add_argument("--output", type=Path, required=True, help="Output pairs.jsonl")
    p.add_argument("--hard-k", type=int, default=3, help="# of hard negatives per pair")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_table(args.input)
    pairs = to_pairs(rows, hard_k=args.hard_k)
    write_jsonl(args.output, pairs)
    print(f"[OK] wrote pairs: {args.output} ({len(pairs)})")


if __name__ == "__main__":
    main()

