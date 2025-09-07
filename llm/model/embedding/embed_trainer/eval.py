from __future__ import annotations
"""
CLI evaluator: 저장된 파인튜닝 결과 디렉터리와 pairs.jsonl을 받아
recall@1/5 및 mrr@10을 출력합니다.
"""
import argparse
from pathlib import Path
import sys
from pathlib import Path as _Path

# 안전 임포트: 패키지로 실행하거나 단일 스크립트로 실행 모두 지원
if __package__ in (None, ""):
    CURRENT_DIR = _Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from eval_utils import load_model_and_tokenizer, evaluate_pairs_with_model  # type: ignore
else:
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

    # Try to load optional projection head saved during training
    proj_head = None
    ph_path = model_dir / "projection_head.pt"
    if ph_path.exists():
        import torch
        import torch.nn as nn
        blob = torch.load(ph_path, map_location="cpu")
        in_dim = int(blob.get("in_dim"))
        out_dim = int(blob.get("out_dim", in_dim))
        act_name = str(blob.get("activation", "tanh")).lower()
        act = nn.Tanh() if act_name == "tanh" else nn.ReLU()
        class ProjectionHead(nn.Module):
            def __init__(self, in_dim: int, out_dim: int, activation: nn.Module):
                super().__init__()
                self.linear = nn.Linear(in_dim, out_dim)
                self.act = activation
            def forward(self, x):
                return self.act(self.linear(x))
        proj_head = ProjectionHead(in_dim, out_dim, act)
        proj_head.load_state_dict(blob["state_dict"])
        proj_head.to(model.device if hasattr(model, "device") else "cpu")

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
        proj_head=proj_head,
    )
    # Pretty print summary with counts and percentages for clarity
    n = int(metrics.get("n_queries", 0))
    r1 = float(metrics.get("recall@1", 0.0))
    r5 = float(metrics.get("recall@5", 0.0))
    mrr10 = float(metrics.get("mrr@10", 0.0))
    h1 = int(metrics.get("hits@1", round(r1 * n)))
    h5 = int(metrics.get("hits@5", round(r5 * n)))

    print(f"queries: {n}")
    print(f"recall@1: {r1*100:.2f}% ({h1}/{n})")
    print(f"recall@5: {r5*100:.2f}% ({h5}/{n})")
    print(f"mrr@10:  {mrr10*100:.2f}%")


if __name__ == "__main__":
    main()
