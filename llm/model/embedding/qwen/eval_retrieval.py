# -*- coding: utf-8 -*-
"""
eval_retrieval.py (리팩터링 버전: 실행 인자 없이 eval_config.json 자동 로드)
--------------------------------------------------------------------------
- 같은 폴더의 eval_config.json을 자동 로드하여 평가 실행
- 모든 경로는 "eval_config.json 파일이 있는 폴더" 기준으로 상대경로 해석
- 모델/토크나이저 로드 → anchors/positives 임베딩 → Top-1 & Recall@K 출력
- (옵션) query 테스트, TSV 저장
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("eval")

# ---------------- Utils ----------------
def load_json(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def to_abs(path_str: str, base_dir: Path) -> Path | None:
    if path_str is None or str(path_str).strip() == "":
        return None
    p = Path(str(path_str).strip()).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p

def ensure_exists(path: Path, hint: str):
    if path is None:
        raise ValueError(f"{hint} 경로가 비어 있습니다.")
    if not path.exists():
        parent = path.parent if path else None
        ls = []
        try:
            if parent and parent.exists():
                ls = [x.name for x in sorted(parent.iterdir())][:30]
        except Exception:
            pass
        msg = f"{hint}를 찾을 수 없습니다: {path}"
        if ls:
            msg += f"\n- 참고: 상위 폴더 목록({parent}): {ls}"
        raise FileNotFoundError(msg)

def resolve_device_dtype(prefer_mps: bool = True, dtype_opt: str = "auto"):
    use_mps = torch.backends.mps.is_available() and prefer_mps
    device = torch.device("mps" if use_mps else "cpu")
    if dtype_opt == "float16":
        dtype = torch.float16
    elif dtype_opt == "float32":
        dtype = torch.float32
    else:
        dtype = torch.float16 if device.type == "mps" else torch.float32
    return device, dtype

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    s = (last_hidden_state * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1e-6)
    return s / d

@torch.inference_mode()
def encode(texts, tok, model, device, max_length=64, batch_size=16):
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, return_dict=True)
        if getattr(out, "pooler_output", None) is not None:
            emb = out.pooler_output
        else:
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        outs.append(torch.nn.functional.normalize(emb, dim=-1))
    return torch.cat(outs, dim=0) if outs else torch.empty(0, model.config.hidden_size, device=device)

def recall_at_k(sim, k):
    topk = sim.topk(k=k, dim=-1).indices
    m = torch.arange(sim.size(0), device=sim.device).unsqueeze(-1)
    return (topk == m).any(dim=-1).float().mean().item()

# ---------------- Main ----------------
def main():
    # 0) 설정 파일 경로 결정 (이 스크립트와 같은 폴더의 eval_config.json)
    cfg_path = Path(__file__).with_name("eval_config.json").resolve()
    cfg = load_json(cfg_path)
    base_dir = cfg_path.parent
    log.info(f"[Config] {cfg_path}")
    log.info(f"[BaseDir] {base_dir}")

    # 1) 경로 해석
    model_dir = to_abs(cfg.get("model_dir"), base_dir)
    pairs_path = to_abs(cfg.get("pairs_path"), base_dir)

    # model_dir이 비어있다면 train_config에서 추론
    if model_dir is None:
        fb = cfg.get("fallback", {}) or {}
        train_cfg_path = to_abs(fb.get("train_config_path"), base_dir)
        ensure_exists(train_cfg_path, "fallback.train_config_path")
        train_cfg = load_json(train_cfg_path)
        out = train_cfg.get("output", {})
        save_dir = to_abs(out.get("save_dir", "./outputs"), train_cfg_path.parent)
        save_name = out.get("save_name", "model-ft")
        model_dir = (save_dir / save_name).resolve()
        log.info(f"[ModelDir] inferred from train_config: {model_dir}")

    ensure_exists(model_dir, "model_dir")
    ensure_exists(pairs_path, "pairs_path")

    # 2) 장치/정밀도
    device, dtype = resolve_device_dtype(
        cfg.get("device", {}).get("prefer_mps", True),
        cfg.get("device", {}).get("dtype", "auto"),
    )
    log.info(f"[Device] {device}, dtype={dtype}")
    log.info(f"[ModelDir] {model_dir}")
    log.info(f"[Pairs] {pairs_path}")

    # 3) 모델 로드
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModel.from_pretrained(str(model_dir), torch_dtype=dtype, device_map=None).to(device).eval()

    # 4) 데이터 로드
    import json as _json
    rows = []
    max_samples = int(cfg.get("eval", {}).get("max_samples", 500))
    with pairs_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            rows.append(_json.loads(line))
    anchors  = [r["anchor"] for r in rows]
    positives= [r["positive"] for r in rows]
    log.info(f"[Eval] samples={len(anchors)} (max={max_samples})")

    # 5) max_length 확보 (train_config에서 읽거나 기본 64)
    max_length = 64
    fb = cfg.get("fallback", {}) or {}
    if fb.get("train_config_path"):
        try:
            train_cfg_path = to_abs(fb.get("train_config_path"), base_dir)
            train_cfg = load_json(train_cfg_path)
            max_length = int(train_cfg.get("data", {}).get("max_length", 64))
        except Exception as e:
            log.warning(f"[Warn] train_config에서 max_length를 읽지 못했습니다. 기본 64 사용. ({e})")

    # 6) 임베딩 & 지표
    bs = int(cfg.get("eval", {}).get("batch_size", 16))
    A = encode(anchors, tok, model, device, max_length=max_length, batch_size=bs)
    P = encode(positives, tok, model, device, max_length=max_length, batch_size=bs)

    sim = A @ P.T
    topk = int(cfg.get("eval", {}).get("topk", 10))
    acc1 = (sim.argmax(dim=-1) == torch.arange(sim.size(0), device=sim.device)).float().mean().item()
    r_at_k = recall_at_k(sim, topk)
    log.info(f"[Metrics] Top-1 acc={acc1:.4f}, Recall@{topk}={r_at_k:.4f}")

    # 7) (옵션) 단일 쿼리 테스트
    qtext = cfg.get("eval", {}).get("query")
    if qtext:
        q = encode([qtext], tok, model, device, max_length=max_length, batch_size=bs)
        scores = (q @ P.T).squeeze(0)
        k = min(topk, scores.numel())
        idx = scores.topk(k=k).indices.tolist()
        log.info(f"[Query] \"{qtext}\" → top-{k} positives")
        for rank, i in enumerate(idx, 1):
            log.info(f"  {rank:>2}. {scores[i].item():.3f} | {positives[i]}")

    # 8) (옵션) TSV 저장
    export_tsv = cfg.get("eval", {}).get("export_tsv")
    if export_tsv:
        outp = to_abs(export_tsv, base_dir)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            f.write("anchor\tbest_positive\tscore\n")
            argmax = sim.argmax(dim=-1).tolist()
            for i, a in enumerate(anchors):
                j = argmax[i]
                f.write(f"{a}\t{positives[j]}\t{sim[i, j].item():.6f}\n")
        log.info(f"[Export] wrote TSV: {outp}")

if __name__ == "__main__":
    main()