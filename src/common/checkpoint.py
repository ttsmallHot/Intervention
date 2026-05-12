"""Checkpoint helpers shared by train and eval phases."""

import os

import torch


def save_checkpoint(plugin, epoch: int, val_acc: float, output_dir: str, tag: str = "latest"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{tag}_plugin.pt")
    torch.save({
        "epoch":          epoch,
        "boost_strength": plugin.boost_strength.data.cpu(),
        "layer_range":    plugin.layer_range,
        "mode":           plugin.mode,
        "free_train":     plugin.free_train,
        "val_acc":        val_acc,
    }, path)
    return path


def load_checkpoint(plugin, checkpoint_path: str, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved = ckpt["boost_strength"].to(plugin.boost_strength.device)
    if saved.shape != plugin.boost_strength.shape:
        raise RuntimeError(
            f"Checkpoint boost_strength shape {tuple(saved.shape)} "
            f"!= plugin shape {tuple(plugin.boost_strength.shape)}. "
            f"Checkpoint was trained with free_train="
            f"{ckpt.get('free_train', 'unknown')}."
        )
    with torch.no_grad():
        plugin.boost_strength.data.copy_(saved)
    return ckpt
