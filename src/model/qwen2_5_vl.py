"""
Attention Intervention plugin for Qwen2.5-VL.

Layer access  : model.model.layers
Image tokens  : tokens between vision_start (151652) and vision_end (151653)
Tested model  : Qwen2.5-VL-3B-Instruct / 7B-Instruct
"""

from __future__ import annotations
from typing import List, Optional

import torch
from .base import BaseAttentionPlugin


class Qwen25VLPlugin(BaseAttentionPlugin):
    """
    AttentionSteerPlugin for Qwen2.5-VL family.

    Example usage::

        from src.model.qwen2_5_vl import Qwen25VLPlugin

        plugin = Qwen25VLPlugin(model, boost_strength=1.0, mode="image", learnable=True)
        plugin.apply()

        inputs = processor(...)
        plugin.update_masks(inputs["input_ids"])
        output = model.generate(**inputs)
    """

    # Qwen2.5-VL special token ids
    VISION_START = 151652
    VISION_END   = 151653

    def __init__(
        self,
        model,
        boost_strength: float = 1.0,
        mode: str = "image",
        layer_range: Optional[List[int]] = None,
        learnable: bool = False,
    ):
        super().__init__(model, boost_strength, mode, layer_range, learnable)

    # ------------------------------------------------------------------
    def _get_layers(self) -> list:
        return self.model.model.layers

    def _build_img_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mark tokens between vision_start and vision_end as image tokens."""
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
