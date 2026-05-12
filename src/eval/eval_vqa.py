"""
VQA accuracy evaluation: Base model vs Base + trained plugin.

Supports: qwen2_5vl | qwen3vl | llava | internvl | gemma3

Usage
-----
python src/eval/eval_vqa.py --config configs/qwen2_5_frozenlake.yaml

Optional overrides:
  --checkpoint /path/to/best_plugin.pt
  --num_samples 500
  --output_dir /tmp/eval_out
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from datetime import datetime

import yaml
import torch
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoProcessor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from src.model import build_plugin
from src.train.utils import save_checkpoint, load_checkpoint, VQADataset
from src.eval.utils import load_model, infer_one


# ---------------------------------------------------------------------------
# Evaluate one mode
# ---------------------------------------------------------------------------

def extract_digit(text: str) -> str:
    digits = re.findall(r"\d+", text)
    return digits[0] if digits else ""


# ---------------------------------------------------------------------------
# Evaluate one mode
# ---------------------------------------------------------------------------

def evaluate_mode(
    mode_name: str,
    model,
    processor,
    dataset,
    plugin=None,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Mode: {mode_name}")
    print(f"{'='*60}")

    correct = 0
    rapt_imgs, rapt_txts = [], []

    for item in tqdm(dataset, desc=f"  {mode_name}"):
        gt = str(item["label"]).strip()
        pred_text, rapt = infer_one(
            model, processor, item["image"], item.get("prompt", ""), 
            plugin, max_new_tokens=10, compute_rapt=True
        )
        pred = extract_digit(pred_text)

        if pred == gt:
            correct += 1
        rapt_imgs.append(rapt["image"])
        rapt_txts.append(rapt["text"])

    n = len(dataset)
    acc = correct / n if n > 0 else 0.0
    result = {
        "mode":       mode_name,
        "accuracy":   acc,
        "correct":    correct,
        "total":      n,
        "rapt_image": float(np.mean(rapt_imgs)),
        "rapt_text":  float(np.mean(rapt_txts)),
    }
    print(f"  Accuracy  : {acc:.2%} ({correct}/{n})")
    print(f"  RAPT img  : {result['rapt_image']:.4f}")
    print(f"  RAPT text : {result['rapt_text']:.4f}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Eval: Base vs Base+Plugin (VQA)")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  default=None, help="Override checkpoint path")
    parser.add_argument("--max_samples", type=int, default=None, help="Override max_samples")
    parser.add_argument("--output_dir",  default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    # Priority: CLI --checkpoint > yaml 'checkpoint' key > output_dir/best_plugin.pt
    checkpoint_path = (
        args.checkpoint
        or cfg.get("checkpoint")
        or os.path.join(cfg.get("output_dir", ""), "best_plugin.pt")
    )
    output_dir = cfg.get("output_dir", "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("\n[1] Loading model...")
    model, processor = load_model(cfg)
    for p in model.parameters():
        p.requires_grad = False

    # Load dataset (test split)
    print(f"\n[2] Loading dataset: {cfg['data_path']}")
    ds = load_dataset("parquet", data_files={"train": cfg["data_path"]})["train"]
    indices = list(range(len(ds)))
    # Fixed split to match training: must not be changed
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    max_samples = args.max_samples or cfg.get("max_samples", None)
    if max_samples:
        test_idx = test_idx[:max_samples]
    test_ds = ds.select(test_idx)
    print(f"    Test samples: {len(test_ds)}")

    results = []

    # Create a base plugin (not applied) — used only for build_prompt() and
    # compute_rapt() so that both Base and Plugin modes use consistent formatting.
    base_plugin = build_plugin(
        cfg["model_type"], model,
        boost_strength = 0.0,
        mode           = cfg.get("mode", "image"),
        learnable      = False,
    )
    # NOTE: base_plugin.apply() is intentionally NOT called here.

    # Mode 1: Base (no plugin attention bias)
    results.append(evaluate_mode("Base (No Plugin)", model, processor, test_ds, plugin=base_plugin))

    # Mode 2: Trained plugin (if checkpoint exists) OR fixed-strength plugin (fallback)
    if os.path.exists(checkpoint_path):
        print(f"\n[3] Checkpoint found — loading trained plugin: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        plugin = build_plugin(
            cfg["model_type"], model,
            boost_strength = 0.0,
            mode           = ckpt.get("mode", cfg.get("mode", "image")),
            layer_range    = ckpt.get("layer_range", None),
            learnable      = True,
            free_train     = ckpt.get("free_train", cfg.get("free_train", True)),
        )
        load_checkpoint(plugin, checkpoint_path)
        plugin.apply()
        ft = ckpt.get('free_train', True)
        print(f"    epoch      : {ckpt.get('epoch')}")
        print(f"    val_acc    : {ckpt.get('val_acc', 0):.4f}")
        print(f"    mode       : {ckpt.get('mode')}")
        print(f"    layer_range: {plugin.layer_range[0]}-{plugin.layer_range[-1]} ({len(plugin.layer_range)} layers)")
        print(f"    free_train : {ft}  ({'per-layer' if ft else 'unified'})")
        print(f"    strengths  : {plugin.boost_strength.data.cpu().numpy().round(4)}")
        results.append(evaluate_mode("Base + Trained Plugin", model, processor, test_ds, plugin=plugin))
        plugin.disable()
    else:
        fixed_strength = cfg.get("fixed_strength", 1.0)
        print(f"\n[3] No checkpoint found — using fixed plugin (strength={fixed_strength})")
        plugin = build_plugin(
            cfg["model_type"], model,
            boost_strength = fixed_strength,
            mode           = cfg.get("mode", "image"),
            learnable      = False,
        )
        plugin.apply()
        results.append(evaluate_mode(f"Base + Fixed Plugin (s={fixed_strength})", model, processor, test_ds, plugin=plugin))
        plugin.disable()

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  {'Mode':<25} {'Acc':>8}  {'RAPT-img':>10}  {'RAPT-txt':>10}")
    print(f"  {'-'*25} {'-'*8}  {'-'*10}  {'-'*10}")
    for r in results:
        print(f"  {r['mode']:<25} {r['accuracy']:>7.2%}  {r['rapt_image']:>10.4f}  {r['rapt_text']:>10.4f}")
    if len(results) >= 2:
        delta = results[-1]["accuracy"] - results[0]["accuracy"]
        print(f"\n  Plugin effect: {'+' if delta>=0 else ''}{delta:.2%}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"eval_vqa_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
