"""VQA accuracy evaluation: Base model vs Base + trained plugin.

Usage
-----
python src/eval/eval_vqa.py --config configs/qwen2_5_frozenlake.yaml

Optional overrides:
  --checkpoint /path/to/best_plugin.pt
  --num_samples 500
  --output_dir /tmp/eval_out
"""

from __future__ import annotations
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.common.extractors import extract_digit
from src.eval.engine import (
    RaptTracker,
    base_parser,
    load_cfg_with_overrides,
    run_eval_pipeline,
    _resolve_checkpoint,
)


def load_split(cfg: dict, max_samples):
    """Load the held-out 20% split that matches training's random_state=42."""
    ds = load_dataset("parquet", data_files={"train": cfg["data_path"]})["train"]
    indices = list(range(len(ds)))
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    if max_samples:
        test_idx = test_idx[:max_samples]
    return ds.select(test_idx)


def main():
    args = base_parser("Eval: Base vs Base+Plugin (VQA)").parse_args()
    cfg = load_cfg_with_overrides(args, "max_samples")

    print(f"\n[2] Loading dataset: {cfg['data_path']}")
    test_ds = load_split(cfg, cfg.get("max_samples"))
    print(f"    Test samples: {len(test_ds)}")

    samples = [
        {"image": r["image"], "prompt": r.get("prompt", ""), "label": str(r["label"])}
        for r in test_ds
    ]

    run_eval_pipeline(
        cfg, samples, extract_digit,
        tag="vqa",
        checkpoint_path=_resolve_checkpoint(args.checkpoint, cfg),
        max_new_tokens=args.max_new_tokens or cfg.get("max_new_tokens", 10),
        compute_rapt=True,
        make_hooks=lambda: [RaptTracker()],
    )


if __name__ == "__main__":
    main()
