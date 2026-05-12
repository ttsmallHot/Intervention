"""
Training utilities: dataset and collate functions.

Works with any VQA-style parquet dataset where each row has:
  - 'image'  : PIL.Image or bytes
  - 'prompt' : str
  - 'label'  : int | str  (ground truth answer)
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.common.image import prepare_image


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
        image = prepare_image(item["image"])

        return {
            "image":  image,
            "prompt": item.get("prompt", ""),
            "label":  str(item["label"]),
            "mode":   self.mode,
        }


# ---------------------------------------------------------------------------
# Collate – one generic implementation, three thin wrappers
# ---------------------------------------------------------------------------
#
# All three model families share the same recipe:
#   1. build a chat-style message per sample
#   2. apply_chat_template -> prompt-only text
#   3. (train) tokenize both prompt-only and prompt+label so that
#         label_token_count = full_real_len - prompt_real_len
#      gives an attention-mask–accurate label span regardless of padding side
#      or standalone-tokenization space-prefix quirks.
#
# Only three things differ between families:
#   - whether `images` must be nested as [[img], [img], ...] (gemma3)
#   - whether the chat message embeds the PIL image or just a placeholder (llava)
#   - whether the tokenizer needs a pad_token set (llava-mistral)

def _build_messages(prompts, images, embed_image: bool):
    """Return a list of single-turn user messages, with or without inline image."""
    messages = []
    for prompt, img in zip(prompts, images):
        content = (
            [{"type": "image", "image": img}, {"type": "text", "text": prompt}]
            if embed_image
            else [{"type": "image"}, {"type": "text", "text": prompt}]
        )
        messages.append([{"role": "user", "content": content}])
    return messages


def _label_tensor(inputs, inp_prompt, batch_size):
    """Mask everything except the label tokens (the trailing real tokens)."""
    labels = torch.full_like(inputs["input_ids"], -100)
    for i in range(batch_size):
        prompt_real_len = inp_prompt["attention_mask"][i].sum().item()
        full_real_len   = inputs["attention_mask"][i].sum().item()
        label_token_count = full_real_len - prompt_real_len
        if label_token_count <= 0:
            continue
        real_pos = inputs["attention_mask"][i].nonzero(as_tuple=True)[0]
        label_positions = real_pos[-label_token_count:]
        labels[i][label_positions] = inputs["input_ids"][i][label_positions]
    return labels


def _collate_generic(
    batch: list,
    processor,
    *,
    nest_images: bool = False,
    embed_image: bool = True,
    ensure_pad_token: bool = False,
) -> tuple:
    images      = [item["image"]  for item in batch]
    prompts     = [item["prompt"] for item in batch]
    labels_text = [item["label"]  for item in batch]
    mode        = batch[0]["mode"]

    if ensure_pad_token and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    images_arg = [[img] for img in images] if nest_images else images

    messages = _build_messages(prompts, images, embed_image=embed_image)
    texts_prompt = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]

    if mode == "train":
        texts_full = [tp + lb for tp, lb in zip(texts_prompt, labels_text)]
        inp_prompt = processor(text=texts_prompt, images=images_arg, padding=True, return_tensors="pt")
        inputs     = processor(text=texts_full,   images=images_arg, padding=True, return_tensors="pt")
        inputs["labels"] = _label_tensor(inputs, inp_prompt, len(batch))
        return inputs, labels_text

    inputs = processor(text=texts_prompt, images=images_arg, padding=True, return_tensors="pt")
    return inputs, labels_text


def collate_qwen(batch, processor):
    return _collate_generic(batch, processor)


def collate_gemma3(batch, processor):
    # Gemma3Processor requires images nested as [[img1], [img2], ...].
    return _collate_generic(batch, processor, nest_images=True)


def collate_llava(batch, processor):
    # LLaVA messages use a placeholder image; PIL is passed via the `images` kwarg.
    # LLaVA-Mistral processors have no pad_token by default.
    return _collate_generic(batch, processor, embed_image=False, ensure_pad_token=True)


def collate_internvl(batch, tokenizer, image_size: int = 448):
    raise NotImplementedError(
        "InternVL collate requires custom dynamic_preprocess. "
        "See InternVL3_5-4B/modeling_internvl_chat.py for reference."
    )
