"""
MMStar evaluation (multiple-choice VQA): Base model vs Base + trained plugin.
Metric: accuracy on answer option (A/B/C/D).

Usage
-----
python src/eval/eval_refcoco.py --config configs/qwen2_5_mmstar_eval.yaml

Optional overrides:
    --checkpoint /path/to/best_plugin.pt
    --max_samples 500
    --split val
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
    model_type = cfg["model_type"]
    model_path = cfg["model_path"]

    if model_type == "qwen2_5vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager",
        )
    elif model_type == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager",
        )
    elif model_type == "llava":
        from transformers import LlavaNextForConditionalGeneration
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager",
        )
    elif model_type == "internvl":
        import transformers
        model = transformers.AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto",
        )
    elif model_type == "gemma3":
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager",
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _get_field(item: dict, *candidates, default=None):
    for key in candidates:
        if key in item and item[key] is not None:
            return item[key]
    return default


def load_mmstar(dataset_id: str, split: str, max_samples: Optional[int]):
    """
    Load MMStar split from HuggingFace.

    Returns a list of dicts with keys:
        image  : PIL.Image
        prompt : str
        answer : str  (A/B/C/D)
    """
    print(f"  Loading {dataset_id}  split={split} ...")
    ds = load_dataset(dataset_id, split=split)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    samples = []
    for item in ds:
        image = _get_field(item, "image", "img", "pixel_values")
        question = _get_field(item, "question", "prompt", "text")
        answer = _get_field(item, "answer", "label")

        if image is None or question is None or answer is None:
            continue

        # Prevent OOM from extremely large images in MMStar
        if hasattr(image, 'width') and hasattr(image, 'height'):
            if image.width > 1536 or image.height > 1536:
                image.thumbnail((1536, 1536))

        prompt = f"{question}\nAnswer with the option letter only."
        samples.append({"image": image, "prompt": prompt, "answer": str(answer)})

    print(f"  Loaded {len(samples)} samples.")
    return samples


def extract_option(text: str) -> str:
    text = text.upper()
    match = re.search(r"\b([A-D])\b", text)
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------
# Evaluate one mode
# ---------------------------------------------------------------------------

def evaluate_mode(mode_name: str, model, processor, samples: list,
                  plugin=None, max_new_tokens: int = 10) -> dict:
    print(f"\n{'='*60}")
    print(f"  Mode: {mode_name}")
    print(f"{'='*60}")

    correct = 0
    parse_fail = 0

    for item in tqdm(samples, desc=f"  {mode_name}"):
        inputs, _ = plugin.build_prompt(
            processor, item["image"], item.get("prompt", "")
        )
        inputs = inputs.to(model.device)

        plugin.update_masks(
            inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            image_sizes=inputs.get("image_sizes"),
        )

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            gen_ids_sliced = gen_ids[0][inputs["input_ids"].shape[1]:]
            pred_text = processor.decode(gen_ids_sliced, skip_special_tokens=True)

        inputs = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        # Clear GPU memory aggressively to prevent loop OOM
        del inputs
        del gen_ids
        del gen_ids_sliced
        torch.cuda.empty_cache()

        pred = extract_option(pred_text)
        if not pred:
            parse_fail += 1
            continue
        gt = str(item["answer"]).strip().upper()
        if pred == gt:
            correct += 1

    n = len(samples)
    acc = correct / n if n > 0 else 0.0
    result = {
        "mode":       mode_name,
        "accuracy":   acc,
        "correct":    correct,
        "total":      n,
        "parse_fail": parse_fail,
    }
    print(f"  Accuracy  : {acc:.2%} ({correct}/{n})")
    print(f"  Parse fail: {parse_fail}/{n}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Eval: Base vs Base+Plugin (MMStar)"
    )
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split",       default=None, help="Override dataset_split")
    parser.add_argument("--output_dir",  default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
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
    samples = load_mmstar(
        dataset_id  = cfg.get("dataset_id", "Lin-Chen/MMStar"),
        split       = cfg.get("dataset_split", "val"),
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
                      samples, plugin=base_plugin,
                      max_new_tokens=args.max_new_tokens or cfg.get("max_new_tokens", 10))
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
                          samples, plugin=plugin,
                          max_new_tokens=args.max_new_tokens or cfg.get("max_new_tokens", 10))
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
                          model, processor, samples, plugin=plugin,
                          max_new_tokens=args.max_new_tokens or cfg.get("max_new_tokens", 10))
        )
        plugin.disable()

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  {'Mode':<30} {'Acc':>8}")
    print(f"  {'-'*30} {'-'*8}")
    for r in results:
        print(f"  {r['mode']:<30} {r['accuracy']:>7.2%}")
    if len(results) >= 2:
        delta = results[-1]["accuracy"] - results[0]["accuracy"]
        sign = "+" if delta >= 0 else ""
        print(f"\n  Plugin effect: {sign}{delta:.2%}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"eval_mmstar_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
