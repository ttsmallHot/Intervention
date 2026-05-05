"""
RefCOCO evaluation: Base model vs Base + trained plugin.
Metric: Acc@0.5 (IoU > 0.5 between predicted and ground-truth bounding box).

Qwen2.5-VL outputs bounding boxes as:
    <|box_start|>(x1,y1),(x2,y2)<|box_end|>
where coordinates are normalised to [0, 1000].

Usage
-----
python src/eval/eval_refcoco.py --config configs/qwen2_5_refcoco_eval.yaml

Optional overrides:
  --checkpoint /path/to/best_plugin.pt
  --max_samples 500
  --split testA
  --output_dir /tmp/eval_out
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Optional

import yaml
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from src.model import build_plugin
from src.train.utils import load_checkpoint


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(cfg: dict):
    model_path = cfg["model_path"]
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


# ---------------------------------------------------------------------------
# Dataset loading — handles multiple HF dataset schemas
# ---------------------------------------------------------------------------

def _get_field(item: dict, *candidates, default=None):
    for key in candidates:
        if key in item and item[key] is not None:
            return item[key]
    return default


def load_refcoco(dataset_id: str, split: str, max_samples: Optional[int]):
    """
    Load RefCOCO split from HuggingFace.

    Returns a list of dicts with keys:
        image      : PIL.Image
        expression : str
        bbox       : [x1, y1, x2, y2]  (pixel coords)
    """
    print(f"  Loading {dataset_id}  split={split} ...")
    ds = load_dataset(dataset_id, split=split, trust_remote_code=True)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    samples = []
    for item in ds:
        image = _get_field(item, "image", "img", "pixel_values")

        # Expression / referring text
        expr = _get_field(item, "sent", "expression", "question",
                          "referring_expression", "text")
        if expr is None:
            sents = item.get("sentences") or item.get("refs") or []
            if sents:
                first = sents[0]
                expr = first.get("sent") or first.get("raw") or str(first)

        # Ground-truth bbox [x, y, w, h] -> [x1, y1, x2, y2]
        raw_bbox = _get_field(item, "bbox", "gt_bbox", "box", "bounding_box")
        if raw_bbox is None:
            continue
        raw_bbox = list(raw_bbox)
        if len(raw_bbox) == 4:
            x, y, w, h = raw_bbox
            # Detect if already x2/y2 format (w > x means it's a coordinate, not width)
            if w > x + 1 or h > y + 1:
                bbox = [x, y, w, h]
            else:
                bbox = [x, y, x + w, y + h]
        else:
            continue

        if image is None or expr is None:
            continue
        samples.append({"image": image, "expression": str(expr), "bbox": bbox})

    print(f"  Loaded {len(samples)} samples.")
    return samples


# ---------------------------------------------------------------------------
# Grounding prompt & output parsing
# ---------------------------------------------------------------------------

GROUNDING_PROMPT = (
    "Please provide the bounding box coordinate of the region "
    "this sentence describes: {expression}"
)


def build_inputs(processor, image, expression: str):
    prompt_text = GROUNDING_PROMPT.format(expression=expression)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt_text},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    )
    return inputs


def parse_bbox(text: str) -> Optional[list]:
    """
    Extract [x1, y1, x2, y2] from model output.
    Handles:
      <|box_start|>(x1,y1),(x2,y2)<|box_end|>  — Qwen special tokens
      [x1, y1, x2, y2] or (x1, y1, x2, y2)
    """
    # Qwen paired-point format: (x1,y1),(x2,y2)
    m = re.search(
        r"\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\),\s*"
        r"\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)",
        text,
    )
    if m:
        return [float(m.group(i)) for i in range(1, 5)]

    # Fallback: first 4 numbers in the string
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if len(nums) >= 4:
        return [float(nums[i]) for i in range(4)]
    return None


def compute_iou(pred: list, gt: list, img_w: int, img_h: int,
                pred_normalised: bool = True) -> float:
    """
    Compute IoU.
    pred : [x1, y1, x2, y2] in [0,1000] if pred_normalised else pixels
    gt   : [x1, y1, x2, y2] in pixel coords
    """
    if pred_normalised:
        px1 = pred[0] / 1000.0 * img_w
        py1 = pred[1] / 1000.0 * img_h
        px2 = pred[2] / 1000.0 * img_w
        py2 = pred[3] / 1000.0 * img_h
    else:
        px1, py1, px2, py2 = pred

    gx1, gy1, gx2, gy2 = gt

    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_p = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    area_g = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = area_p + area_g - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluate one mode
# ---------------------------------------------------------------------------

def evaluate_mode(mode_name: str, model, processor, samples: list,
                  plugin=None) -> dict:
    print(f"\n{'='*60}")
    print(f"  Mode: {mode_name}")
    print(f"{'='*60}")

    correct = 0
    parse_fail = 0
    ious = []

    for item in tqdm(samples, desc=f"  {mode_name}"):
        image  = item["image"]
        expr   = item["expression"]
        gt_box = item["bbox"]

        img_w = getattr(image, "width",  640)
        img_h = getattr(image, "height", 480)

        inputs = build_inputs(processor, image, expr)
        inputs = inputs.to(model.device)

        if plugin is not None:
            plugin.update_masks(
                inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                image_sizes=inputs.get("image_sizes"),
            )

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=50)
            gen_ids = gen_ids[0][inputs["input_ids"].shape[1]:]
            pred_text = processor.decode(gen_ids, skip_special_tokens=False)

        pred_box = parse_bbox(pred_text)
        if pred_box is None:
            parse_fail += 1
            ious.append(0.0)
            continue

        iou = compute_iou(pred_box, gt_box, img_w, img_h, pred_normalised=True)
        ious.append(iou)
        if iou >= 0.5:
            correct += 1

    n = len(samples)
    acc = correct / n if n > 0 else 0.0
    result = {
        "mode":       mode_name,
        "acc_at_0_5": acc,
        "correct":    correct,
        "total":      n,
        "parse_fail": parse_fail,
        "mean_iou":   float(np.mean(ious)),
    }
    print(f"  Acc@0.5   : {acc:.2%} ({correct}/{n})")
    print(f"  Mean IoU  : {result['mean_iou']:.4f}")
    print(f"  Parse fail: {parse_fail}/{n}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Eval: Base vs Base+Plugin (RefCOCO Acc@0.5)"
    )
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split",       default=None, help="Override dataset_split")
    parser.add_argument("--output_dir",  default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.split:
        cfg["dataset_split"] = args.split

    checkpoint_path = (
        args.checkpoint
        or cfg.get("checkpoint")
        or os.path.join(cfg.get("output_dir", ""), "best_plugin.pt")
    )
    output_dir = cfg.get("output_dir", "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1] Loading model...")
    model, processor = load_model(cfg)
    for p in model.parameters():
        p.requires_grad = False

    print("\n[2] Loading dataset...")
    samples = load_refcoco(
        dataset_id  = cfg.get("dataset_id", "lmms-lab/RefCOCO"),
        split       = cfg.get("dataset_split", "testA"),
        max_samples = args.max_samples or cfg.get("max_samples"),
    )

    results = []

    # Shared base plugin — used for consistent build_prompt/compute_rapt,
    # but NOT applied (boost_strength=0 keeps attention unmodified).
    base_plugin = build_plugin(
        cfg["model_type"], model,
        boost_strength=0.0,
        mode=cfg.get("mode", "image"),
        learnable=False,
    )

    # Mode 1: Base
    results.append(
        evaluate_mode("Base (No Plugin)", model, processor,
                      samples, plugin=base_plugin)
    )

    # Mode 2: Trained or fixed-strength plugin
    if os.path.exists(checkpoint_path):
        print(f"\n[3] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        plugin = build_plugin(
            cfg["model_type"], model,
            boost_strength=0.0,
            mode=ckpt.get("mode", cfg.get("mode", "image")),
            layer_range=ckpt.get("layer_range"),
            learnable=True,
            free_train=ckpt.get("free_train", cfg.get("free_train", True)),
        )
        load_checkpoint(plugin, checkpoint_path)
        plugin.apply()
        print(f"  epoch   : {ckpt.get('epoch')}")
        print(f"  val_acc : {ckpt.get('val_acc', 0):.4f}")
        results.append(
            evaluate_mode("Base + Trained Plugin", model, processor,
                          samples, plugin=plugin)
        )
        plugin.disable()
    else:
        strength = cfg.get("fixed_strength", 1.0)
        print(f"\n[3] No checkpoint — fixed plugin (strength={strength})")
        plugin = build_plugin(
            cfg["model_type"], model,
            boost_strength=strength,
            mode=cfg.get("mode", "image"),
            learnable=False,
        )
        plugin.apply()
        results.append(
            evaluate_mode(f"Base + Fixed Plugin (s={strength})",
                          model, processor, samples, plugin=plugin)
        )
        plugin.disable()

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  {'Mode':<30} {'Acc@0.5':>8}  {'Mean IoU':>10}")
    print(f"  {'-'*30} {'-'*8}  {'-'*10}")
    for r in results:
        print(f"  {r['mode']:<30} {r['acc_at_0_5']:>7.2%}  {r['mean_iou']:>10.4f}")
    if len(results) >= 2:
        delta = results[-1]["acc_at_0_5"] - results[0]["acc_at_0_5"]
        sign = "+" if delta >= 0 else ""
        print(f"\n  Plugin effect: {sign}{delta:.2%}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"eval_refcoco_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
