"""
Attention Intervention plugin for Qwen3-VL.

Layer access  : model.model.language_model.layers
Image tokens  : tokens between vision_start (151652) and vision_end (151653)
Tested model  : Qwen3-VL-4B-Instruct
"""

from __future__ import annotations
from typing import List, Optional

import torch
from .base import BaseAttentionPlugin


class Qwen3VLPlugin(BaseAttentionPlugin):
    """
    AttentionSteerPlugin for Qwen3-VL family.

    Structurally similar to Qwen2.5-VL but the language model is nested one
    level deeper: `model.model.language_model.layers`.

    Example usage::

        from src.model.qwen3_vl import Qwen3VLPlugin

        plugin = Qwen3VLPlugin(model, boost_strength=1.0, mode="image", learnable=True)
        plugin.apply()
    """

    VISION_START = 151652
    VISION_END   = 151653

    def __init__(
        self,
        model,
        boost_strength: float = 1.0,
        mode: str = "image",
        layer_range: Optional[List[int]] = None,
        learnable: bool = False,
        free_train: bool = True,
    ):
        super().__init__(model, boost_strength, mode, layer_range, learnable, free_train)

    # ------------------------------------------------------------------
    def _get_layers(self) -> list:
        return self.model.model.language_model.layers

    def _build_img_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        mask = torch.zeros((B, L), dtype=torch.bool, device=input_ids.device)
        for b in range(B):
            in_vision = False
            for i in range(L):
                tid = input_ids[b, i].item()
                if tid == self.VISION_START:
                    in_vision = True
                elif tid == self.VISION_END:
                    in_vision = False
                elif in_vision:
                    mask[b, i] = True
        return mask
