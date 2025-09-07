from __future__ import annotations
"""
eval_utils.py
--------------
임베딩 파인튜닝 결과를 간단히 평가하는 도우미. 쿼리(anchors)→코퍼스(positives)
최근접 검색 정확도와 MRR 등을 계산합니다.

설계 원칙
- 학습 시 사용한 풀링/정규화와 일관성 유지(mean-pool + cosine 유사도)
- 배치/AMP/디바이스를 유연하게 처리하여 대용량 데이터에도 안전
- SentenceTransformer 경로를 지원하되, 단순하고 유지보수 가능한 형태 유지
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import sys
from pathlib import Path as _Path

import torch
from transformers import AutoTokenizer, AutoModel

# 안전한 임포트: 패키지/단일 스크립트 실행 모두 지원
if __package__ in (None, ""):
    CURRENT_DIR = _Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from utils import resolve_device_dtype, mean_pool  # type: ignore
    from data_utils import load_pairs_jsonl  # type: ignore
else:
    from .utils import resolve_device_dtype, mean_pool
    from .data_utils import load_pairs_jsonl


def _detect_st_transformer(model: Any):
    """SentenceTransformer 사용 시 내부 Transformer 모듈을 찾아 반환.
    실패하면 None.
    """
    try:
        # ST는 모듈 인덱싱으로 접근 가능한 경우가 많음
        sub0 = model[0]  # type: ignore[index]
        return sub0
    except Exception:
        return None


def _forward_hidden_states(model_or_sub: Any, inputs: Dict[str, torch.Tensor], device_type: str, st_transformer: Optional[Any] = None):
    """AutoModel 또는 ST-transformer에서 token-level 은닉 상태를 얻는다."""
    if st_transformer is not None:
        # ST transformer는 종종 dtype/AMP 특이점이 있어 AMP 비활성 경로를 사용
        with torch.amp.autocast(device_type=device_type, enabled=False):
            out = st_transformer(inputs)
        if isinstance(out, dict) and "token_embeddings" in out:
            return out["token_embeddings"]
        if isinstance(out, dict) and "last_hidden_state" in out:
            return out["last_hidden_state"]
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state  # type: ignore[attr-defined]
        raise RuntimeError("Unexpected ST transformer output shape")

    fwd = getattr(model_or_sub, "forward", model_or_sub)
    out = fwd(**inputs, return_dict=True)
    return out.last_hidden_state


@torch.no_grad()
def embed_texts(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 32,
    max_length: int = 128,
) -> torch.Tensor:
    """텍스트 리스트를 배치로 임베딩(mean-pool)하여 하나의 텐서로 반환."""
    st_transformer = _detect_st_transformer(model)
    device_type = device.type
    all_vecs: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        # forward는 AMP 사용 가능. 풀링은 내부에서 float32로 처리됨.
        with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=(device_type != "cpu")):
            lhs = _forward_hidden_states(model, enc, device_type=device_type, st_transformer=st_transformer)
        vec = mean_pool(lhs, enc["attention_mask"])  # returns lhs dtype
        all_vecs.append(vec)
    return torch.cat(all_vecs, dim=0)


def compute_retrieval_metrics(q: torch.Tensor, c: torch.Tensor, k_values: Tuple[int, ...] = (1, 5)) -> Dict[str, float]:
    """쿼리 임베딩(q)과 코퍼스 임베딩(c)으로 R@k, MRR@10을 계산.
    q, c는 같은 순서(동일 인덱스 쌍이 긍정)라고 가정.
    """
    # cosine 유사도: float32에서 정규화 후 행렬 곱
    qn = torch.nn.functional.normalize(q.float(), dim=-1, eps=1e-6)
    cn = torch.nn.functional.normalize(c.float(), dim=-1, eps=1e-6)
    sims = qn @ cn.t()
    labels = torch.arange(qn.size(0), device=qn.device)
    # ranks: 각 쿼리의 정답 인덱스 순위(0이 best)
    sorted_idx = sims.argsort(dim=-1, descending=True)
    ranks = (sorted_idx == labels.unsqueeze(1)).nonzero()[:, 1]

    metrics: Dict[str, float] = {}
    for k in k_values:
        r_at_k = (ranks < k).float().mean().item()
        metrics[f"recall@{k}"] = r_at_k
    # MRR@10
    cutoff = 10
    rr = (1.0 / (ranks.float() + 1.0)) * (ranks < cutoff).float()
    metrics["mrr@10"] = rr.mean().item()
    return metrics


def evaluate_pairs_with_model(
    model: Any,
    tokenizer: Any,
    pairs_path: Path,
    device: Optional[torch.device] = None,
    dtype_opt: str = "auto",
    batch_size: int = 32,
    max_length: int = 128,
) -> Dict[str, float]:
    """저장된 모델/토크나이저를 사용해 pairs.jsonl의 R@k/MRR을 계산."""
    # imported at module level with safe path above

    rows = load_pairs_jsonl(pairs_path)
    anchors = [r["anchor"] for r in rows]
    positives = [r["positive"] for r in rows]

    if device is None:
        device, dtype, _ = resolve_device_dtype()
    else:
        # dtype 결정
        if dtype_opt == "auto":
            if device.type == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif device.type == "mps":
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
            dtype = mapping.get(dtype_opt.lower(), torch.float32)

    model.to(device)
    model.eval()

    emb_q = embed_texts(model, tokenizer, anchors, device=device, dtype=dtype, batch_size=batch_size, max_length=max_length)
    emb_c = embed_texts(model, tokenizer, positives, device=device, dtype=dtype, batch_size=batch_size, max_length=max_length)
    return compute_retrieval_metrics(emb_q, emb_c)


def load_model_and_tokenizer(model_dir: Path, trust_remote_code: bool = False):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code, use_fast=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    return model, tok
