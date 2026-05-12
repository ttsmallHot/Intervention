"""Shared evaluation pipeline used by eval_vqa.py and eval_mmstar.py.

Both scripts only differ in three things:
  - how the dataset is loaded (parquet split vs HF MMStar)
  - which extractor parses the model output
  - which per-sample metrics get tracked (RAPT, parse_fail, ...)

Everything else (model loading, checkpoint resolution, base plugin, JSON dump,
summary printing) lives here.
"""

from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.common.checkpoint import load_checkpoint
from src.eval.utils import load_model, infer_one
from src.model import build_plugin


# ---------------------------------------------------------------------------
# Per-sample metric hooks
# ---------------------------------------------------------------------------

class RaptTracker:
    """Accumulates mean RAPT-image / RAPT-text. Use with compute_rapt=True."""

    def __init__(self):
        self.imgs, self.txts = [], []

    def __call__(self, pred, gt, rapt):
        if rapt is not None:
            self.imgs.append(rapt["image"])
            self.txts.append(rapt["text"])

    def summary(self) -> dict:
        return {
            "rapt_image": float(np.mean(self.imgs)) if self.imgs else 0.0,
            "rapt_text":  float(np.mean(self.txts)) if self.txts else 0.0,
        }


class ParseFailTracker:
    """Counts samples where the extractor produced an empty string."""

    def __init__(self):
        self.n = 0

    def __call__(self, pred, gt, rapt):
        if not pred:
            self.n += 1

    def summary(self) -> dict:
        return {"parse_fail": self.n}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_mode(
    mode_name: str,
    model,
    processor,
    samples,
    plugin,
    extract_fn: Callable[[str], str],
    *,
    max_new_tokens: int = 10,
    compute_rapt: bool = False,
    hooks: Optional[list] = None,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Mode: {mode_name}")
    print(f"{'='*60}")

    hooks = hooks or []
    correct = 0

    for item in tqdm(samples, desc=f"  {mode_name}"):
        pred_text, rapt = infer_one(
            model, processor, item["image"], item.get("prompt", ""),
            plugin, max_new_tokens=max_new_tokens, compute_rapt=compute_rapt,
        )
        pred = extract_fn(pred_text)
        gt = str(item["label"]).strip()
        if pred == gt:
            correct += 1
        for hook in hooks:
            hook(pred, gt, rapt)

    n = len(samples)
    acc = correct / n if n > 0 else 0.0
    result = {"mode": mode_name, "accuracy": acc, "correct": correct, "total": n}
    for hook in hooks:
        result.update(hook.summary())

    print(f"  Accuracy  : {acc:.2%} ({correct}/{n})")
    for hook in hooks:
        for k, v in hook.summary().items():
            print(f"  {k:<10}: {v:.4f}" if isinstance(v, float) else f"  {k:<10}: {v}")
    return result


# ---------------------------------------------------------------------------
# Argparse + pipeline
# ---------------------------------------------------------------------------

def base_parser(description: str) -> argparse.ArgumentParser:
    """Common CLI flags shared by every eval script."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config",         required=True)
    parser.add_argument("--checkpoint",     default=None, help="Override checkpoint path")
    parser.add_argument("--max_samples",    type=int, default=None)
    parser.add_argument("--output_dir",     default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    return parser


def _resolve_checkpoint(cli_path: Optional[str], cfg: dict) -> str:
    return (
        cli_path
        or cfg.get("checkpoint")
        or os.path.join(cfg.get("output_dir", ""), "best_plugin.pt")
    )


def _print_summary(results: list, mode_col_width: int = 30) -> None:
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  {'Mode':<{mode_col_width}} {'Acc':>8}")
    print(f"  {'-'*mode_col_width} {'-'*8}")
    for r in results:
        print(f"  {r['mode']:<{mode_col_width}} {r['accuracy']:>7.2%}")
    if len(results) >= 2:
        delta = results[-1]["accuracy"] - results[0]["accuracy"]
        sign = "+" if delta >= 0 else ""
        print(f"\n  Plugin effect: {sign}{delta:.2%}")


def _save_results(results: list, output_dir: str, tag: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"eval_{tag}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {out_path}")
    return out_path


def run_eval_pipeline(
    cfg: dict,
    samples,
    extract_fn: Callable[[str], str],
    *,
    tag: str,
    checkpoint_path: Optional[str] = None,
    max_new_tokens: int = 10,
    compute_rapt: bool = False,
    make_hooks: Optional[Callable[[], list]] = None,
) -> list:
    """Run Base vs Base+Plugin evaluation and dump results.

    Args:
        samples         : list of dicts {image, prompt, label}
        extract_fn      : output-text → predicted-answer parser
        tag             : filename suffix for the result JSON
        checkpoint_path : if file exists, load trained plugin; else fall back
                          to fixed-strength plugin from cfg["fixed_strength"]
        make_hooks      : factory returning per-mode metric hook instances
    """
    output_dir = cfg.get("output_dir", "eval_results")

    print("\n[1] Loading model...")
    model, processor = load_model(cfg)
    for p in model.parameters():
        p.requires_grad = False

    base_plugin = build_plugin(
        cfg["model_type"], model,
        boost_strength=0.0,
        mode=cfg.get("mode", "image"),
        learnable=False,
    )

    results = []

    results.append(evaluate_mode(
        "Base (No Plugin)", model, processor, samples,
        plugin=base_plugin, extract_fn=extract_fn,
        max_new_tokens=max_new_tokens, compute_rapt=compute_rapt,
        hooks=make_hooks() if make_hooks else None,
    ))

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n[3] Checkpoint found — loading trained plugin: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        plugin = build_plugin(
            cfg["model_type"], model,
            boost_strength=0.0,
            mode=ckpt.get("mode", cfg.get("mode", "image")),
            layer_range=ckpt.get("layer_range", None),
            learnable=True,
            free_train=ckpt.get("free_train", cfg.get("free_train", True)),
        )
        load_checkpoint(plugin, checkpoint_path)
        plugin.apply()
        ft = ckpt.get("free_train", True)
        print(f"    epoch      : {ckpt.get('epoch')}")
        print(f"    val_acc    : {ckpt.get('val_acc', 0):.4f}")
        print(f"    mode       : {ckpt.get('mode')}")
        print(f"    layer_range: {plugin.layer_range[0]}-{plugin.layer_range[-1]} ({len(plugin.layer_range)} layers)")
        print(f"    free_train : {ft}  ({'per-layer' if ft else 'unified'})")
        print(f"    strengths  : {plugin.boost_strength.data.cpu().numpy().round(4)}")
        mode_name = "Base + Trained Plugin"
    else:
        strength = cfg.get("fixed_strength", 1.0)
        print(f"\n[3] No checkpoint — using fixed plugin (strength={strength})")
        plugin = build_plugin(
            cfg["model_type"], model,
            boost_strength=strength,
            mode=cfg.get("mode", "image"),
            learnable=False,
        )
        plugin.apply()
        mode_name = f"Base + Fixed Plugin (s={strength})"

    results.append(evaluate_mode(
        mode_name, model, processor, samples,
        plugin=plugin, extract_fn=extract_fn,
        max_new_tokens=max_new_tokens, compute_rapt=compute_rapt,
        hooks=make_hooks() if make_hooks else None,
    ))
    plugin.disable()

    _print_summary(results)
    _save_results(results, output_dir, tag=tag)
    return results


def load_cfg_with_overrides(args, *extra_keys: str) -> dict:
    """Load YAML and apply --output_dir / extra CLI overrides."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    for key in extra_keys:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    return cfg
