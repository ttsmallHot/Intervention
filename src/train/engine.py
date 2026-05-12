import os
import argparse
import yaml
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.checkpoint import save_checkpoint
from src.common.models import load_model_and_processor
from src.model import build_plugin
from src.train.utils import (
    VQADataset, collate_qwen, collate_gemma3, collate_llava, collate_internvl,
)


# ---------------------------------------------------------------------------
# Collate registry
# ---------------------------------------------------------------------------

COLLATE_FNS = {
    "qwen2_5vl": collate_qwen,
    "qwen3vl":   collate_qwen,
    "gemma3":    collate_gemma3,
    "llava":     collate_llava,
    "internvl":  collate_internvl,
}


def get_collate_fn(model_type: str, processor):
    if model_type not in COLLATE_FNS:
        raise ValueError(f"No collate_fn for model_type: {model_type}")
    fn = COLLATE_FNS[model_type]
    return lambda batch: fn(batch, processor)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate(model, processor, plugin, val_loader, extract_fn, print_samples: int = 4) -> float:
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

    correct = 0
    if print_samples > 0:
        print(f"\n  {'label':<8} {'output':<40} {'correct'}")
        print(f"  {'-'*8} {'-'*40} {'-'*7}")

    for i, (out, lbl) in enumerate(zip(all_outputs, all_labels)):
        out_clean = out.strip().replace("\n", " ")
        lbl_clean = lbl.strip()
        pred = extract_fn(out_clean)
        is_ok = pred == lbl_clean
        if is_ok:
            correct += 1
        if i < print_samples:
            ok_str = "✓" if is_ok else "✗"
            print(f"  {lbl_clean:<8} {out_clean[:40]:<40} {ok_str}")

    if print_samples > 0:
        print()

    return correct / len(all_labels) if all_labels else 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_train_pipeline(cfg: dict, extract_fn):
    print("=" * 65)
    print(f"  Attention Intervention Training")
    print(f"  model_type : {cfg['model_type']}")
    print(f"  model_path : {cfg['model_path']}")
    if "train_data_path" in cfg:
        print(f"  train_data : {cfg['train_data_path']}")
        print(f"  val_data   : {cfg['val_data_path']}")
    else:
        print(f"  data_path  : {cfg.get('data_path', 'Not provided')}")
    print(f"  output_dir : {cfg['output_dir']}")
    print("=" * 65)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("\n[1] Loading model...")
    model, processor = load_model_and_processor(cfg)
    for param in model.parameters():
        param.requires_grad = False
    print(f"    Loaded ({model.config.model_type})")

    print("\n[2] Initialising plugin...")
    plugin = build_plugin(
        cfg["model_type"],
        model,
        boost_strength = cfg.get("boost_strength_init", 0.0),
        mode           = cfg.get("mode", "image"),
        layer_range    = cfg.get("layer_range", None),
        learnable      = True,
        free_train     = cfg.get("free_train", True),
    )
    plugin.apply()

    print(f"\n[3] Loading dataset...")
    if "train_data_path" in cfg and "val_data_path" in cfg:
        train_ds = load_dataset("parquet", data_files={"train": cfg["train_data_path"]})["train"]
        val_ds   = load_dataset("parquet", data_files={"train": cfg["val_data_path"]})["train"]
    else:
        ds = load_dataset("parquet", data_files={"train": cfg["data_path"]})["train"]
        max_samples = cfg.get("max_samples", None)
        indices = list(range(len(ds)))[:max_samples] if max_samples else list(range(len(ds)))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        train_ds = ds.select(train_idx)
        val_ds   = ds.select(val_idx)

    collate_fn = get_collate_fn(cfg["model_type"], processor)
    batch_size = cfg.get("batch_size", 32)
    train_loader = DataLoader(
        VQADataset(train_ds, processor, mode="train"),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        VQADataset(val_ds, processor, mode="inference"),
        batch_size=max(1, batch_size // 4), shuffle=False, collate_fn=collate_fn,
    )
    print(f"    train={len(train_ds)}, val={len(val_ds)}")

    optimizer = torch.optim.AdamW(plugin.parameters(), lr=cfg.get("learning_rate", 1e-2))

    print("\n[4] Baseline validation (before training)...")
    best_val_acc = evaluate(model, processor, plugin, val_loader, extract_fn=extract_fn)
    print(f"    Baseline val acc: {best_val_acc:.2%}")

    eval_every = cfg.get("eval_every", 5)
    num_epochs = cfg.get("num_epochs", 50)

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
            grad_clip = cfg.get("grad_clip", None)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(plugin.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{total_loss/(step+1):.4f}",
                strength=f"{plugin.boost_strength.mean().item():.3f}",
            )

        save_checkpoint(plugin, epoch, 0.0, cfg["output_dir"], tag="latest")

        if (epoch + 1) % eval_every == 0 or (epoch + 1) == num_epochs:
            val_acc = evaluate(model, processor, plugin, val_loader, extract_fn=extract_fn)
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
# CLI
# ---------------------------------------------------------------------------

# Single source of truth: each entry is (cli_flag, type). Adding an override
# here automatically wires both argparse and the YAML-merge step.
CLI_OVERRIDES = [
    ("model_type",    str),
    ("model_path",    str),
    ("data_path",     str),
    ("output_dir",    str),
    ("num_epochs",    int),
    ("batch_size",    int),
    ("learning_rate", float),
    ("max_samples",   int),
    ("mode",          str),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Attention Intervention plugin")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    for name, typ in CLI_OVERRIDES:
        parser.add_argument(f"--{name}", default=None, type=typ)
    return parser.parse_args()


def run_main(extract_fn):
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for name, _ in CLI_OVERRIDES:
        val = getattr(args, name, None)
        if val is not None:
            cfg[name] = val
    run_train_pipeline(cfg, extract_fn)
