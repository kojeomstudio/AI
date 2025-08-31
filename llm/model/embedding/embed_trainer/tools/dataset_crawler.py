#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General dataset crawler for embedding training.

Sources
- wikipedia_glossary: Video game glossary page (demo offline fallback)
- custom_urls: Arbitrary pages (best-effort title/definition extraction)

Outputs
- CSV with columns: term,definition,lang,source
- JSONL pairs: {anchor,positive,hard_negatives?,lang,source}

Note: Ensure site terms/robots/license compliance when crawling.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import time
from typing import List, Dict, Iterable, Optional
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    requests = None
    BeautifulSoup = None


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
WIKI_GLOSSARY_URL = "https://en.wikipedia.org/wiki/Glossary_of_video_game_terms"


# ------------------------- I/O -------------------------
def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["term", "definition", "lang", "source"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "term": r.get("term", ""),
                "definition": r.get("definition", ""),
                "lang": r.get("lang", "auto"),
                "source": r.get("source", "unknown"),
            })


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------- Utils -------------------------
def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def unique_by_key(rows: Iterable[Dict], key: str) -> List[Dict]:
    seen = set()
    out = []
    for r in rows:
        k = r.get(key, "")
        if k and k not in seen:
            seen.add(k)
            out.append(r)
    return out


def dedup_and_clean(rows: List[Dict]) -> List[Dict]:
    rows = [
        {
            "term": normalize_text(r.get("term", "")),
            "definition": normalize_text(r.get("definition", "")),
            "lang": r.get("lang", "auto"),
            "source": r.get("source", "unknown"),
        }
        for r in rows
        if r.get("term") and r.get("definition")
    ]
    rows = unique_by_key(rows, key="term")
    return rows


# ------------------------- Crawlers -------------------------
def http_get(url: str, timeout: int = 15) -> Optional[str]:
    if requests is None:
        return None
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "embed-trainer-crawler/1.0"})
        if resp.status_code == 200:
            return resp.text
        return None
    except Exception:
        return None


def crawl_wikipedia_glossary(limit: int = 1000) -> List[Dict]:
    html = http_get(WIKI_GLOSSARY_URL)
    if not html or BeautifulSoup is None:
        # Offline demo sample
        return [
            {"term": "cooldown", "definition": "A period of time a player must wait before using an ability again.", "lang": "en", "source": "wikipedia_glossary"},
            {"term": "hitbox", "definition": "An invisible shape used for detecting collisions between objects.", "lang": "en", "source": "wikipedia_glossary"},
            {"term": "gacha", "definition": "A monetization mechanic where players spend currency for random rewards.", "lang": "en", "source": "wikipedia_glossary"},
        ]
    soup = BeautifulSoup(html, "html.parser")
    out: List[Dict] = []
    for dl in soup.select("dl"):
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            term = normalize_text(dt.get_text(" ", strip=True))
            definition = normalize_text(dd.get_text(" ", strip=True))
            if term and definition and len(definition) > 10:
                out.append({"term": term, "definition": definition, "lang": "en", "source": "wikipedia_glossary"})
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break
    # Fallback pattern
    if len(out) < max(50, limit // 4):
        for li in soup.select("li"):
            strong = li.find("b") or li.find("strong")
            if strong:
                term = normalize_text(strong.get_text(" ", strip=True))
                text = normalize_text(li.get_text(" ", strip=True))
                definition = normalize_text(text.replace(term, "", 1))
                if term and len(definition) > 10:
                    out.append({"term": term, "definition": definition, "lang": "en", "source": "wikipedia_glossary"})
                if len(out) >= limit:
                    break
    return out[:limit]


def crawl_custom_urls(urls: List[str], limit_per_source: int = 500) -> List[Dict]:
    rows: List[Dict] = []
    if BeautifulSoup is None or requests is None:
        # Offline demo
        demo = [
            ("Critical hit", "An attack that deals bonus damage with a given probability."),
            ("Pull", "A tactic to lure one or more enemies to a desired location."),
            ("Cooldown", "A delay before a skill can be used again."),
        ]
        return [
            {"term": t, "definition": d, "lang": "en", "source": "custom_demo"}
            for t, d in demo
        ]

    for url in urls:
        html = http_get(url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        # Heuristic: dt/dd and h2/h3 + p
        for dl in soup.select("dl"):
            dts = dl.find_all("dt")
            dds = dl.find_all("dd")
            for dt, dd in zip(dts, dds):
                term = normalize_text(dt.get_text(" ", strip=True))
                defn = normalize_text(dd.get_text(" ", strip=True))
                if term and len(defn) > 10:
                    rows.append({"term": term, "definition": defn, "lang": "auto", "source": url})
                if len(rows) >= limit_per_source:
                    break
            if len(rows) >= limit_per_source:
                break
        if len(rows) < limit_per_source:
            heads = soup.select("h2, h3")
            for h in heads:
                nxt = h.find_next_sibling(["p", "ul", "ol"]) or h.find_next("p")
                if nxt:
                    term = normalize_text(h.get_text(" ", strip=True))
                    defn = normalize_text(nxt.get_text(" ", strip=True))
                    if term and len(defn) > 10:
                        rows.append({"term": term, "definition": defn, "lang": "auto", "source": url})
                if len(rows) >= limit_per_source:
                    break
    return rows


# ------------------------- Build Pairs -------------------------
def build_pairs_from_glossary(rows: List[Dict], hard_k: int = 3) -> List[Dict]:
    pairs: List[Dict] = []
    defs = [r["definition"] for r in rows]
    for i, r in enumerate(rows):
        anchor = r["term"]
        pos = r["definition"]
        # naive hard negatives: other random definitions nearby
        negs = []
        # pick next few definitions cyclically
        for j in range(1, hard_k + 1):
            negs.append(defs[(i + j) % len(defs)])
        pairs.append({
            "anchor": anchor,
            "positive": pos,
            "hard_negatives": negs,
            "lang": r.get("lang", "auto"),
            "source": r.get("source", "unknown"),
        })
    return pairs


# ------------------------- CLI -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="General dataset crawler for embedding training")
    p.add_argument("--sources", nargs="+", default=["wikipedia_glossary"],
                   help="wikipedia_glossary | custom_urls")
    p.add_argument("--custom_urls", nargs="*", default=[],
                   help="URLs to crawl when using --sources custom_urls")
    p.add_argument("--limit_per_source", type=int, default=500)
    p.add_argument("--output_csv", type=Path, default=DATA_DIR / "glossary.csv")
    p.add_argument("--output_pairs", type=Path, default=DATA_DIR / "pairs.jsonl")
    p.add_argument("--dry-run", action="store_true", help="Offline demo without network")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict] = []

    if args.dry_run or "wikipedia_glossary" in args.sources:
        rows += crawl_wikipedia_glossary(limit=args.limit_per_source)
    if "custom_urls" in args.sources and args.custom_urls:
        rows += crawl_custom_urls(args.custom_urls, limit_per_source=args.limit_per_source)

    rows = dedup_and_clean(rows)
    write_csv(args.output_csv, rows)
    print(f"[OK] wrote CSV: {args.output_csv} ({len(rows)})")

    pairs = build_pairs_from_glossary(rows, hard_k=3)
    write_jsonl(args.output_pairs, pairs)
    print(f"[OK] wrote pairs: {args.output_pairs} ({len(pairs)})")


if __name__ == "__main__":
    main()

