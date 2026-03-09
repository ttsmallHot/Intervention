"""
Unified training script for Attention Intervention plugins.

Supports: qwen2_5vl | qwen3vl | llava | internvl
Dataset : VQA-style parquet (image, prompt, label)

Usage
-----
python src/train/train.py \
    --config configs/qwen2_5_frozenlake.yaml

Or override any field directly:
python src/train/train.py \
    --config configs/qwen2_5_frozenlake.yaml \
    --num_epochs 30 \
    --batch_size 16
"""

from __future__ import annotations
import argparse
import os
import sys
import re

import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Make sure project root is on sys.path when running as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from src.model import build_plugin
from src.train.utils import (
    VQADataset, collate_qwen, collate_llava,
    compute_accuracy, save_checkpoint
)

# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_model_and_processor(cfg: dict):
    model_type  = cfg["model_type"]
    model_path  = cfg["model_path"]

    if model_type in ("qwen2_5vl",):
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager"
        )
    elif model_type == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager"
        )
    elif model_type == "llava":
        from transformers import LlavaNextForConditionalGeneration
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager"
        )
    elif model_type == "internvl":
        import transformers
        model = transformers.AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto"
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def get_collate_fn(model_type: str, processor):
    if model_type in ("qwen2_5vl", "qwen3vl"):
        return lambda batch: collate_qwen(batch, processor)
    elif model_type == "llava":
        return lambda batch: collate_llava(batch, processor)
    elif model_type == "internvl":
        from src.train.utils import collate_internvl
        return lambda batch: collate_internvl(batch, processor)
    raise ValueError(f"No collate_fn for model_type: {model_type}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate(model, processor, plugin, val_loader, model_type: str, print_samples: int = 4) -> float:
    model.eval()
    plugin.eval()
    all_outputs, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="  Validation", leave=False):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            plugin.update_masks(
                inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                image_sizes=inputs.get("image_sizes"),
            )

            gen_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            generated_ids = model.generate(**gen_inputs, max_new_tokens=10)
            generated_ids = [
                out[len(inp):]
                for inp, out in zip(inputs["input_ids"], generated_ids)
            ]
            texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            all_outputs.extend(texts)
            all_labels.extend(labels)

    acc = compute_accuracy(all_outputs, all_labels)

    # Print a few samples so we can see what the model is actually generating
    if print_samples > 0:
        print(f"\n  {'label':<8} {'output':<40} {'correct'}")
        print(f"  {'-'*8} {'-'*40} {'-'*7}")
        for i in range(min(print_samples, len(all_labels))):
            out = all_outputs[i].strip().replace("\n", " ")[:40]
            lbl = all_labels[i]
            nums = re.findall(r'\d+', all_outputs[i])
            ok = "✓" if nums and nums[0] == lbl else "✗"
            print(f"  {lbl:<8} {out:<40} {ok}")
        print()

    return acc


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict):
    print("=" * 65)
    print(f"  Attention Intervention Training")
    print(f"  model_type : {cfg['model_type']}")
    print(f"  model_path : {cfg['model_path']}")
    print(f"  data_path  : {cfg['data_path']}")
    print(f"  output_dir : {cfg['output_dir']}")
    print("=" * 65)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # 1. Load model
    print("\n[1] Loading model...")
    model, processor = load_model_and_processor(cfg)
    for param in model.parameters():
        param.requires_grad = False
    print(f"    Loaded ({model.config.model_type})")

    # 2. Build plugin
    print("\n[2] Initialising plugin...")
    plugin = build_plugin(
        cfg["model_type"],
        model,
        boost_strength = cfg.get("boost_strength_init", 0.0),
        mode           = cfg.get("mode", "image"),
        layer_range    = cfg.get("layer_range", None),
        learnable      = True,
    )
    plugin.apply()

    # 3. Load dataset
    print(f"\n[3] Loading dataset: {cfg['data_path']}")
    ds = load_dataset("parquet", data_files={"train": cfg["data_path"]})["train"]
    max_samples = cfg.get("max_samples", None)
    indices = list(range(len(ds)))[:max_samples] if max_samples else list(range(len(ds)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42  # fixed split, must match eval
    )

    collate_fn = get_collate_fn(cfg["model_type"], processor)

    train_loader = DataLoader(
        VQADataset(ds.select(train_idx), processor, mode="train"),
        batch_size = cfg.get("batch_size", 32),
        shuffle    = True,
        collate_fn = collate_fn,
    )
    val_loader = DataLoader(
        VQADataset(ds.select(val_idx), processor, mode="inference"),
        batch_size = cfg.get("batch_size", 32),
        shuffle    = False,
        collate_fn = collate_fn,
    )
    print(f"    train={len(train_idx)}, val={len(val_idx)}")

    # 4. Optimizer
    optimizer = torch.optim.AdamW(
        plugin.parameters(), lr=cfg.get("learning_rate", 1e-2)
    )

    # 5. Initial baseline
    print("\n[4] Baseline validation (before training)...")
    best_val_acc = evaluate(model, processor, plugin, val_loader, cfg["model_type"])
    print(f"    Baseline val acc: {best_val_acc:.2%}")

    eval_every  = cfg.get("eval_every", 5)
    num_epochs  = cfg.get("num_epochs", 50)

    # 6. Training loop
    print(f"\n[5] Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        plugin.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, (inputs, _) in enumerate(pbar):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            plugin.update_masks(
                inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                image_sizes=inputs.get("image_sizes"),
            )

            outputs = model(**inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{total_loss/(step+1):.4f}",
                strength=f"{plugin.boost_strength.mean().item():.3f}",
            )

        # Save latest checkpoint each epoch
        save_checkpoint(plugin, epoch, 0.0, cfg["output_dir"], tag="latest")

        # Validation
        if (epoch + 1) % eval_every == 0 or (epoch + 1) == num_epochs:
            val_acc = evaluate(model, processor, plugin, val_loader, cfg["model_type"])
            avg_loss = total_loss / len(train_loader)

            print(f"\nEpoch {epoch+1:3d} | "
                  f"loss={avg_loss:.4f} | val_acc={val_acc:.2%} | "
                  f"strengths={plugin.boost_strength.data.cpu().numpy().round(3)}")

            save_checkpoint(plugin, epoch, val_acc, cfg["output_dir"], tag="latest")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                path = save_checkpoint(plugin, epoch, val_acc, cfg["output_dir"], tag="best")
                print(f"  ✅ New best ({best_val_acc:.2%}) saved -> {path}")

    print(f"\n[Done] Best val acc: {best_val_acc:.2%}")
    print(f"       Checkpoints in: {cfg['output_dir']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Attention Intervention plugin")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    # Allow any config key to be overridden from CLI
    parser.add_argument("--model_type",   default=None)
    parser.add_argument("--model_path",   default=None)
    parser.add_argument("--data_path",    default=None)
    parser.add_argument("--output_dir",   default=None)
    parser.add_argument("--num_epochs",   type=int,   default=None)
    parser.add_argument("--batch_size",   type=int,   default=None)
    parser.add_argument("--learning_rate",type=float, default=None)
    parser.add_argument("--max_samples",  type=int,   default=None)
    parser.add_argument("--mode",         default=None, help="image|text|both|oppose")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    for key in ("model_type", "model_path", "data_path", "output_dir",
                "num_epochs", "batch_size", "learning_rate", "max_samples", "mode"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    train(cfg)


if __name__ == "__main__":
    main()
