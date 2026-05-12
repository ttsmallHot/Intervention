"""MMStar evaluation (multiple-choice VQA): Base model vs Base + trained plugin.

Usage
-----
python src/eval/eval_mmstar.py --config configs/qwen2_5_mmstar_eval.yaml

Optional overrides:
    --checkpoint /path/to/best_plugin.pt
    --max_samples 500
    --split val
    --output_dir /tmp/eval_out
"""

from __future__ import annotations
import os
import sys
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from datasets import load_dataset

from src.common.extractors import extract_option
from src.eval.engine import (
    ParseFailTracker,
    base_parser,
    load_cfg_with_overrides,
    run_eval_pipeline,
    _resolve_checkpoint,
)


def _get_field(item: dict, *candidates, default=None):
    for key in candidates:
        if key in item and item[key] is not None:
            return item[key]
    return default


def load_mmstar(dataset_id: str, split: str, max_samples: Optional[int]):
    """Load MMStar split from HuggingFace. Returns list of {image, prompt, label}."""
    print(f"  Loading {dataset_id}  split={split} ...")
    ds = load_dataset(dataset_id, split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    samples = []
    for item in ds:
        image    = _get_field(item, "image", "img", "pixel_values")
        question = _get_field(item, "question", "prompt", "text")
        answer   = _get_field(item, "answer", "label")
        if image is None or question is None or answer is None:
            continue
        samples.append({
            "image":  image,
            "prompt": f"{question}\nAnswer with the option letter only.",
            "label":  str(answer).strip().upper(),
        })

    print(f"  Loaded {len(samples)} samples.")
    return samples


def main():
    parser = base_parser("Eval: Base vs Base+Plugin (MMStar)")
    parser.add_argument("--split", default=None, help="Override dataset_split")
    args = parser.parse_args()

    cfg = load_cfg_with_overrides(args, "max_samples")
    if args.split:
        cfg["dataset_split"] = args.split

    print("\n[2] Loading dataset...")
    samples = load_mmstar(
        dataset_id  = cfg.get("dataset_id", "Lin-Chen/MMStar"),
        split       = cfg.get("dataset_split", "val"),
        max_samples = cfg.get("max_samples"),
    )

    run_eval_pipeline(
        cfg, samples, extract_option,
        tag="mmstar",
        checkpoint_path=_resolve_checkpoint(args.checkpoint, cfg),
        max_new_tokens=args.max_new_tokens or cfg.get("max_new_tokens", 10),
        compute_rapt=False,
        make_hooks=lambda: [ParseFailTracker()],
    )


if __name__ == "__main__":
    main()
