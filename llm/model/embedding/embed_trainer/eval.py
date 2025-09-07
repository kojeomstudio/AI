from __future__ import annotations
"""
CLI evaluator: 저장된 파인튜닝 결과 디렉터리와 pairs.jsonl을 받아
recall@1/5 및 mrr@10을 출력합니다.
"""
import argparse
from pathlib import Path

from .eval_utils import load_model_and_tokenizer, evaluate_pairs_with_model


def main():
    ap = argparse.ArgumentParser(description="Evaluate fine-tuned embedding model on pairs.jsonl")
    ap.add_argument("--model-dir", type=str, required=True, help="path to saved model directory (from train output)")
    ap.add_argument("--pairs", type=str, required=True, help="path to pairs.jsonl")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--dtype", type=str, default="auto", help="auto|float32|float16|bfloat16")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    pairs_path = Path(args.pairs)

    model, tok = load_model_and_tokenizer(model_dir, trust_remote_code=args.trust_remote_code)

    # device handling: keep simple for CLI; detailed rules are inside evaluator when device is None
    device = None
    import torch
    if args.device != "auto":
        if args.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif args.device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    metrics = evaluate_pairs_with_model(
        model,
        tok,
        pairs_path,
        device=device,
        dtype_opt=args.dtype,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()

