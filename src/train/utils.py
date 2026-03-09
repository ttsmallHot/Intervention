"""
Training utilities: dataset, collate function, checkpoint helpers.

Works with any VQA-style parquet dataset where each row has:
  - 'image'  : PIL.Image or bytes
  - 'prompt' : str
  - 'label'  : int | str  (ground truth answer)
"""

from __future__ import annotations
import os
import re
from typing import List, Callable, Optional

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VQADataset(Dataset):
    def __init__(self, hf_dataset, processor, mode: str = "train"):
        """
        Args:
            hf_dataset : HuggingFace Dataset object (already split)
            processor  : model-specific processor / tokenizer
            mode       : "train" (returns inputs + labels tensor) |
                         "inference" (returns inputs only for generate)
        """
        self.dataset = hf_dataset
        self.processor = processor
        self.mode = mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "image":  item["image"],
            "prompt": item.get("prompt", ""),
            "label":  str(item["label"]),
            "mode":   self.mode,
        }


# ---------------------------------------------------------------------------
# Collate functions (one per model family)
# ---------------------------------------------------------------------------

def collate_qwen(batch: list, processor) -> tuple:
    """Collate for Qwen2.5-VL and Qwen3-VL (share the same processor API).

    Uses the full_real_len - prompt_real_len diff method (same as collate_llava)
    so that label token count is always accurate regardless of pad_token_id or
    padding side.
    """
    images      = [item["image"]  for item in batch]
    prompts     = [item["prompt"] for item in batch]
    labels_text = [item["label"]  for item in batch]
    mode        = batch[0]["mode"]

    # Build messages with real images (not placeholder)
    all_messages = []
    for img, prompt in zip(images, prompts):
        all_messages.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": prompt},
            ],
        }])

    texts_prompt = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in all_messages
    ]

    if mode == "train":
        texts_full = [tp + lb for tp, lb in zip(texts_prompt, labels_text)]

        # Tokenize prompt-only and full sequence.
        # label_token_count = full_real_len - prompt_real_len is accurate in any
        # tokenizer context (no standalone-tokenization space-prefix issue,
        # no hardcoded pad_token_id dependency).
        inp_prompt = processor(
            text=texts_prompt, images=images, padding=True, return_tensors="pt"
        )
        inputs = processor(
            text=texts_full, images=images, padding=True, return_tensors="pt"
        )

        labels = torch.full_like(inputs["input_ids"], -100)
        for i in range(len(batch)):
            prompt_real_len   = inp_prompt["attention_mask"][i].sum().item()
            full_real_len     = inputs["attention_mask"][i].sum().item()
            label_token_count = full_real_len - prompt_real_len
            if label_token_count <= 0:
                continue
            real_pos = inputs["attention_mask"][i].nonzero(as_tuple=True)[0]
            label_positions = real_pos[-label_token_count:]
            labels[i][label_positions] = inputs["input_ids"][i][label_positions]

        inputs["labels"] = labels
        return inputs, labels_text

    else:  # inference
        inputs = processor(
            text=texts_prompt, images=images, padding=True, return_tensors="pt"
        )
        return inputs, labels_text


def collate_gemma3(batch: list, processor) -> tuple:
    """Collate for Gemma-3 multimodal processor.

    Gemma3Processor requires images as a nested list [[img1], [img2], ...]
    where each inner list corresponds to images for one text sample.
    Passing a flat list causes "inconsistently sized batches of images/text".

    Otherwise identical logic to collate_qwen (diff method for label masking).
    """
    images      = [item["image"]  for item in batch]
    prompts     = [item["prompt"] for item in batch]
    labels_text = [item["label"]  for item in batch]
    mode        = batch[0]["mode"]

    # Gemma3 processor needs nested image list
    images_nested = [[img] for img in images]

    all_messages = []
    for img, prompt in zip(images, prompts):
        all_messages.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": prompt},
            ],
        }])

    texts_prompt = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in all_messages
    ]

    if mode == "train":
        texts_full = [tp + lb for tp, lb in zip(texts_prompt, labels_text)]

        inp_prompt = processor(
            text=texts_prompt, images=images_nested, padding=True, return_tensors="pt"
        )
        inputs = processor(
            text=texts_full, images=images_nested, padding=True, return_tensors="pt"
        )

        labels = torch.full_like(inputs["input_ids"], -100)
        for i in range(len(batch)):
            prompt_real_len   = inp_prompt["attention_mask"][i].sum().item()
            full_real_len     = inputs["attention_mask"][i].sum().item()
            label_token_count = full_real_len - prompt_real_len
            if label_token_count <= 0:
                continue
            real_pos = inputs["attention_mask"][i].nonzero(as_tuple=True)[0]
            label_positions = real_pos[-label_token_count:]
            labels[i][label_positions] = inputs["input_ids"][i][label_positions]

        inputs["labels"] = labels
        return inputs, labels_text

    else:  # inference
        inputs = processor(
            text=texts_prompt, images=images_nested, padding=True, return_tensors="pt"
        )
        return inputs, labels_text


def collate_llava(batch: list, processor) -> tuple:
    """
    Collate for LLaVA-Next processors.

    LLaVA processor uses LEFT padding by default (needed for generate).
    With left-padding + batch_size > 1, the prompt_len-from-left approach
    masks PAD tokens instead of actual prompt tokens, leaking the full
    prompt into the CE loss.

    Fix: tokenize each label independently to get its exact token length,
    then supervise ONLY the last label_len real tokens per sample.
    """
    images      = [item["image"]  for item in batch]
    prompts     = [item["prompt"] for item in batch]
    labels_text = [item["label"]  for item in batch]
    mode        = batch[0]["mode"]

    # Ensure tokenizer has a pad token (LLaVA-Mistral uses eos as pad)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    all_messages = []
    for prompt in prompts:
        all_messages.append([{
            "role": "user",
            "content": [
                {"type": "image"},   # no PIL object here; images passed separately
                {"type": "text", "text": prompt},
            ],
        }])

    texts_prompt = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in all_messages
    ]

    if mode == "train":
        texts_full = [tp + lb for tp, lb in zip(texts_prompt, labels_text)]

        # Tokenize prompt-only and full sequence.
        # The number of label tokens in context = full_real_len - prompt_real_len.
        # This avoids the standalone-tokenization space-prefix bug
        # (e.g. "2" → [▁, 2] alone vs [2] after [/INST]).
        inp_prompt = processor(
            text=texts_prompt, images=images, padding=True, return_tensors="pt"
        )
        inputs = processor(
            text=texts_full, images=images, padding=True, return_tensors="pt"
        )

        labels = torch.full_like(inputs["input_ids"], -100)
        for i in range(len(batch)):
            prompt_real_len = inp_prompt["attention_mask"][i].sum().item()
            full_real_len   = inputs["attention_mask"][i].sum().item()
            label_token_count = full_real_len - prompt_real_len
            if label_token_count <= 0:
                continue
            # Real (non-padding) positions in the full sequence
            real_pos = inputs["attention_mask"][i].nonzero(as_tuple=True)[0]
            # Supervise only the last label_token_count real tokens
            label_positions = real_pos[-label_token_count:]
            labels[i][label_positions] = inputs["input_ids"][i][label_positions]

        inputs["labels"] = labels
        return inputs, labels_text
    else:
        inputs = processor(
            text=texts_prompt, images=images, padding=True, return_tensors="pt"
        )
        return inputs, labels_text


def collate_internvl(batch: list, tokenizer, image_size: int = 448) -> tuple:
    """Collate for InternVL (uses its own tokenizer + pixel_values pipeline)."""
    raise NotImplementedError(
        "InternVL collate requires custom dynamic_preprocess. "
        "See InternVL3_5-4B/modeling_internvl_chat.py for reference."
    )


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(outputs: List[str], labels: List[str]) -> float:
    correct = sum(
        bool(re.findall(r'\d+', o)) and re.findall(r'\d+', o)[0] == l
        for o, l in zip(outputs, labels)
    )
    return correct / len(labels) if labels else 0.0


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(plugin, epoch: int, val_acc: float, output_dir: str, tag: str = "latest"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{tag}_plugin.pt")
    torch.save({
        "epoch":          epoch,
        "boost_strength": plugin.boost_strength.data.cpu(),
        "layer_range":    plugin.layer_range,
        "mode":           plugin.mode,
        "val_acc":        val_acc,
    }, path)
    return path


def load_checkpoint(plugin, checkpoint_path: str, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    with torch.no_grad():
        plugin.boost_strength.data.copy_(
            ckpt["boost_strength"].to(plugin.boost_strength.device)
        )
    return ckpt
