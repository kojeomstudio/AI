#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
게임/모바일 도메인 용어 크롤링 → 정규화 → CSV/JSONL 저장 파이프라인

- 소스 예시:
  1) Wikipedia: Glossary of video game terms (영문)
  2) Fandom(LoL 위키) Terminology 등 (옵션; URL 지정)
- 라이선스/robots/약관 준수는 사용자 책임.
- 네트워크가 불가한 환경에서는 --dry-run 으로 데모 샘플 생성 가능.

출력(기본):
  data/game_glossary.csv
  data/pairs.jsonl (anchor, positive, hard_negatives)

사용 예:
  python dataset_crawler.py --sources wikipedia_glossary --limit_per_source 500
  python dataset_crawler.py --sources custom_urls --custom_urls https://example.com/page1 https://example.com/page2
  python dataset_crawler.py --dry-run
"""

import argparse
import csv
import json
import re
import time
import hashlib
from typing import List, Dict, Iterable, Optional, Tuple
from pathlib import Path

# 외부 모듈
try:
    import requests
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    requests = None
    BeautifulSoup = None

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ------------------------- 유틸 -------------------------
def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)

def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\s+", " ", s)
    return s

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def unique_by_key(rows: Iterable[Dict], key: str) -> List[Dict]:
    seen = set()
    out = []
    for r in rows:
        k = r.get(key, "")
        if k and k not in seen:
            seen.add(k)
            out.append(r)
    return out

# ------------------------- 크롤러 -------------------------
WIKI_GLOSSARY_URL = "https://en.wikipedia.org/wiki/Glossary_of_video_game_terms"

def http_get(url: str, timeout: int = 15) -> Optional[str]:
    if requests is None:
        return None
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "dataset-crawler/1.0"})
        if resp.status_code == 200:
            return resp.text
        return None
    except Exception:
        return None

def crawl_wikipedia_glossary(limit: int = 1000, sleep_sec: float = 0.5) -> List[Dict]:
    """위키피디아 비디오 게임 용어집에서 용어-정의 추출"""
    html = http_get(WIKI_GLOSSARY_URL)
    if not html or BeautifulSoup is None:
        # 오프라인/모듈 없음: 데모 샘플
        return [
            {"term": "cooldown", "definition": "A period of time a player must wait before using an ability again.", "lang":"en", "source":"wikipedia_glossary"},
            {"term": "gacha", "definition": "A monetization mechanic where players spend currency for random rewards.", "lang":"en", "source":"wikipedia_glossary"},
            {"term": "reroll", "definition": "To restart an account or process to obtain a desired outcome.", "lang":"en", "source":"wikipedia_glossary"},
        ]
    soup = BeautifulSoup(html, "html.parser")
    out: List[Dict] = []
    # 위키의 Glossary 페이지는 dl/dt/dd 조합 또는 li 구조가 혼재할 수 있음
    # 가장 일반적인 패턴: <dl><dt>Term</dt><dd>Definition...</dd></dl>
    for dl in soup.select("dl"):
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            term = normalize_text(dt.get_text(" ", strip=True))
            definition = normalize_text(dd.get_text(" ", strip=True))
            if term and definition and len(definition) > 10:
                out.append({"term": term, "definition": definition, "lang":"en", "source":"wikipedia_glossary"})
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break
    # 보조 패턴: li strong + text
    if len(out) < max(50, limit//4):
        for li in soup.select("li"):
            strong = li.find("b") or li.find("strong")
            if strong:
                term = normalize_text(strong.get_text(" ", strip=True))
                text = normalize_text(li.get_text(" ", strip=True))
                # strong 용어를 제외한 정의
                definition = normalize_text(text.replace(term, "", 1))
                if term and len(definition) > 10:
                    out.append({"term": term, "definition": definition, "lang":"en", "source":"wikipedia_glossary"})
                if len(out) >= limit:
                    break
    return out[:limit]

def crawl_custom_urls(urls: List[str], limit_per_source: int = 500, sleep_sec: float = 0.5) -> List[Dict]:
    """사용자 지정 문서(FAQ/용어집/공지 등)에서 제목/정의 후보를 추출"""
    rows: List[Dict] = []
    if BeautifulSoup is None or requests is None:
        # 오프라인 데모
        demo_pages = [
            ("What is reroll?", "Reroll is restarting an account to get a better initial draw in gacha games."),
            ("Drop rate", "The probability of obtaining an item from loot sources."),
            ("Critical chance", "The probability that an attack deals bonus damage."),
        ]
        for title, body in demo_pages:
            rows.append({"term": title, "definition": body, "lang":"en", "source":"custom_urls"})
        return rows
    for url in urls:
        html = http_get(url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        # 간단한 룰: h1/h2를 term, 그 다음 문단을 definition 후보로
        for header in soup.select("h1, h2, h3"):
            term = normalize_text(header.get_text(" ", strip=True))
            sib = header.find_next_sibling()
            if not term or not sib:
                continue
            paragraph = sib.get_text(" ", strip=True) if hasattr(sib, "get_text") else ""
            definition = normalize_text(paragraph)[:600]
            if len(definition) < 20:
                continue
            rows.append({"term": term, "definition": definition, "lang":"auto", "source": url})
            if len(rows) >= limit_per_source:
                break
        time.sleep(sleep_sec)
    return rows

# ------------------------- 정규화/중복 -------------------------
def dedup_and_clean(rows: List[Dict]) -> List[Dict]:
    cleaned: List[Dict] = []
    for r in rows:
        term = normalize_text(r.get("term",""))
        definition = normalize_text(r.get("definition",""))
        if not term or not definition:
            continue
        # 너무 짧은/긴 정의 필터
        if not (5 <= len(term) <= 100 and 10 <= len(definition) <= 1000):
            continue
        cleaned.append({
            "term": term,
            "definition": definition,
            "lang": r.get("lang","auto"),
            "source": r.get("source","unknown"),
            "uid": sha1(term + "|" + definition)
        })
    # term 기준 1차 중복 제거
    cleaned = unique_by_key(cleaned, "uid")
    return cleaned

def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["term","definition","lang","source"])
        w.writeheader()
        for r in rows:
            w.writerow({k:r[k] for k in ["term","definition","lang","source"]})

def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ------------------------- 페어 생성 -------------------------
def build_pairs_from_glossary(glossary_rows: List[Dict], hard_k: int = 3) -> List[Dict]:
    """
    anchor: term
    positive: definition
    hard_negatives: 동일 도메인 정의 중 랜덤 샘플(간이). 실전은 BM25/ANN 사용.
    """
    import random
    positives = [r["definition"] for r in glossary_rows]
    out: List[Dict] = []
    for r in glossary_rows:
        negs = []
        if len(positives) > 1:
            pool = [p for p in positives if p != r["definition"]]
            random.shuffle(pool)
            negs = pool[:hard_k]
        out.append({
            "anchor": r["term"],
            "positive": r["definition"],
            "hard_negatives": negs,
            "lang": r.get("lang","auto"),
            "source": r.get("source","unknown"),
        })
    return out

# ------------------------- CLI -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", default=["wikipedia_glossary"],
                   help="wikipedia_glossary | custom_urls")
    p.add_argument("--custom_urls", nargs="*", default=[],
                   help="--sources custom_urls 일 때 대상 URL 목록")
    p.add_argument("--limit_per_source", type=int, default=500)
    p.add_argument("--output_csv", type=Path, default=DATA_DIR / "game_glossary.csv")
    p.add_argument("--output_pairs", type=Path, default=DATA_DIR / "pairs.jsonl")
    p.add_argument("--dry-run", action="store_true", help="네트워크 없이 데모 데이터 생성")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    ensure_dirs()

    rows: List[Dict] = []
    if args.dry_run or "wikipedia_glossary" in args.sources:
        rows += crawl_wikipedia_glossary(limit=args.limit_per_source)
    if "custom_urls" in args.sources and args.custom_urls:
        rows += crawl_custom_urls(args.custom_urls, limit_per_source=args.limit_per_source)

    rows = dedup_and_clean(rows)
    write_csv(args.output_csv, rows)
    print(f"[OK] wrote CSV: {args.output_csv} ({len(rows)} rows)")

    pairs = build_pairs_from_glossary(rows, hard_k=3)
    write_jsonl(args.output_pairs, pairs)
    print(f"[OK] wrote pairs: {args.output_pairs} ({len(pairs)} rows)")

if __name__ == "__main__":
    main()
