"""
Shared evaluation utilities.

Provides:
  - compute_rapt()     : Relative Attention Proportion to Target
  - parse_action()     : extract FrozenLake action from model response
  - FrozenLake helpers : system prompt, observation prompt
"""

from __future__ import annotations
import re
from typing import Optional

import torch
import numpy as np


# ---------------------------------------------------------------------------
# RAPT (Relative Attention Proportion to Target)
# ---------------------------------------------------------------------------

def compute_rapt(
    outputs,
    input_ids: torch.Tensor,           # [L]  (single sample, already on CPU or device)
    vision_start: int = 151652,
    vision_end:   int = 151653,
) -> dict:
    """
    Compute RAPT for the last generated token.

    RAPT_image = mean_attention_to_image_tokens / global_mean_attention
    RAPT_text  = mean_attention_to_text_tokens  / global_mean_attention

    Args:
        outputs    : model output with `output_attentions=True`
        input_ids  : 1-D token id tensor for the single sample

    Returns:
        {"image": float, "text": float}
    """
    if outputs.attentions is None:
        return {"image": 0.0, "text": 0.0}

    num_layers  = len(outputs.attentions)
    deep_layers = outputs.attentions[num_layers // 2:]

    # Average over deep layers and heads; take last token's attention row
    avg_attn = torch.stack(
        [a[0].mean(dim=0).cpu().float() for a in deep_layers]
    ).mean(dim=0)                        # [seq, seq]
    received = avg_attn[-1, :]           # [seq]  last token attends to …

    seq_len = input_ids.shape[0]
    img_mask  = torch.zeros(seq_len, dtype=torch.bool)
    text_mask = torch.ones(seq_len,  dtype=torch.bool)
    in_vision = False

    for i, tid in enumerate(input_ids.cpu().tolist()):
        if tid == vision_start:
            in_vision = True;  text_mask[i] = False
        elif tid == vision_end:
            in_vision = False; text_mask[i] = False
        elif in_vision:
            img_mask[i]  = True
            text_mask[i] = False

    global_mean = received.mean().item()
    if global_mean == 0:
        return {"image": 0.0, "text": 0.0}

    img_rapt  = (received[img_mask].mean().item()  / global_mean) if img_mask.any()  else 0.0
    text_rapt = (received[text_mask].mean().item() / global_mean) if text_mask.any() else 0.0
    return {"image": img_rapt, "text": text_rapt}


# ---------------------------------------------------------------------------
# FrozenLake helpers
# ---------------------------------------------------------------------------

ACTION_LOOKUP = {"Left": 0, "Down": 1, "Right": 2, "Up": 3}

FROZEN_LAKE_SYSTEM_PROMPT = """\
You are a FrozenLake solver.
FrozenLake Quick Guide:
Goal: Reach the goal (G).
Symbols in image: The player is red circle, holes are blue circles, goal is gift box.
Rules:
1. Avoid falling into holes.
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.
Actions you can take: Left, Down, Right, Up.

Format: <think>...</think><answer>...</answer>
You should first give your reasoning, then your answer.
Example: <think>The goal is below and to the left</think><answer>Down</answer>
"""


def get_observation_prompt(is_first_step: bool = True) -> str:
    if is_first_step:
        return "This is the initial state. Decide your next action."
    return "This is the current state after your last action. Decide your next action."


def parse_action(response: str) -> Optional[str]:
    """
    Extract one of Left/Down/Right/Up from a model response.
    First looks inside <answer>…</answer>; falls back to the full text.
    Returns None if no valid action found.
    """
    m = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
    text = m.group(1).strip() if m else response.strip()
    for action in ACTION_LOOKUP:
        if action.lower() in text.lower():
            return action
    return None
