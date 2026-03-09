"""
Attention Intervention plugin for Gemma-3 multimodal.

Layer access  : model.language_model.model.layers
Image tokens  : tokens with id == image_token_index (262144),
                surrounded by BOI (255999) / EOI (256000) markers
Tested model  : gemma-3-4b-it
"""

from __future__ import annotations
from typing import List, Optional

import torch
from .base import BaseAttentionPlugin


class Gemma3Plugin(BaseAttentionPlugin):
    """
    AttentionSteerPlugin for Gemma-3 multimodal family.

    Example usage::

        from src.model.gemma3 import Gemma3Plugin

        plugin = Gemma3Plugin(model, boost_strength=1.0, mode="image", learnable=True)
        plugin.apply()

        inputs = processor(...)
        plugin.update_masks(inputs["input_ids"])
        output = model.generate(**inputs)
    """

    # Gemma-3 special token ids (from config.json)
    IMAGE_TOKEN_ID = 262144   # soft image token repeated mm_tokens_per_image times
    BOI_TOKEN_ID   = 255999   # begin of image
    EOI_TOKEN_ID   = 256000   # end of image

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
        # Gemma3ForConditionalGeneration.language_model (Gemma3TextModel) → .layers
        return self.model.language_model.layers

    def _build_img_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Mark all image-related tokens as True:
          - soft vision tokens  (IMAGE_TOKEN_ID = 262144)
          - BOI / EOI boundary markers
        """
        mask = (
            (input_ids == self.IMAGE_TOKEN_ID) |
            (input_ids == self.BOI_TOKEN_ID)   |
            (input_ids == self.EOI_TOKEN_ID)
        )
        return mask  # [B, L] bool

    def build_prompt(self, processor, image, prompt_text: str) -> tuple:
        """
        Gemma-3 chat template with embedded PIL image in content list.
        """
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
        return inputs, messages
