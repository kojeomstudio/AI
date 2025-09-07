import json
from pathlib import Path

import shutil


def make_pairs(path: Path, n: int = 8):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            ex = {
                "anchor": f"hello world {i}",
                "positive": f"hello world {i}",
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def test_train_end_to_end(tmp_path: Path):
    # Late import to ensure test environment paths are set
    import train as tr

    data_file = tmp_path / "pairs.jsonl"
    out_dir = tmp_path / "outputs"
    make_pairs(data_file, n=8)

    cfg = {
        "model_id": "hf-internal-testing/tiny-random-bert",
        "hf": {
            "trust_remote_code": False,
            "local_files_only": False,
            "loader": "automodel",
        },
        "device": "cpu",
        "dtype": "float32",
        "train": {
            "batch_size": 2,
            "epochs": 1,
            "max_length": 16,
            "grad_accum_steps": 1,
            "steps_per_epoch": 2,
            "log_every": 1,
            "num_workers": 0,
            "pad_to_multiple_of": 8,
        },
        "optim": {
            "name": "adamw",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
        },
        "data": {
            "pairs_path": str(data_file),
            "stream": False,
        },
        "output": {
            "save_dir": str(out_dir),
            "save_name": "smoke",
            "save_metrics": True,
        },
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    tr.main(cfg_path)

    target = out_dir / "smoke"
    assert target.exists()
    metrics = json.loads((target / "metrics.json").read_text(encoding="utf-8"))
    assert metrics.get("total_steps") == 2

    # Evaluate saved model on the same small dataset
    from eval_utils import load_model_and_tokenizer, evaluate_pairs_with_model

    model, tok = load_model_and_tokenizer(target)
    eval_metrics = evaluate_pairs_with_model(
        model,
        tok,
        pairs_path=data_file,
        device=None,
        dtype_opt="float32",
        batch_size=4,
        max_length=16,
    )
    # Basic sanity: metrics present and in range
    for key in ("recall@1", "recall@5", "mrr@10"):
        assert key in eval_metrics
        assert 0.0 <= eval_metrics[key] <= 1.0


def test_streaming_loader(tmp_path: Path):
    # Validate streaming dataloader produces batches without loading entire file
    import data_utils as du
    from transformers import AutoTokenizer

    data_file = tmp_path / "pairs.jsonl"
    make_pairs(data_file, n=5)

    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")

    dl = du.make_dataloader(
        tokenizer_name_or_obj=tok,
        pairs_path=data_file,
        batch_size=2,
        max_length=16,
        device=None,
        shuffle=False,
        num_workers=0,
        stream=True,
        pad_to_multiple_of=8,
    )

    it = iter(dl)
    batch1 = next(it)
    assert set(batch1.keys()) == {"anchor_inputs", "positive_inputs"}
    # Should be batched to size 2
    assert batch1["anchor_inputs"]["input_ids"].shape[0] == 2
