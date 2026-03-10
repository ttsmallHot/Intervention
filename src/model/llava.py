"""
Attention Intervention plugin for LLaVA-Next  (llava-v1.6-*-hf).

Layer access  : model.language_model.model.layers
Image tokens  : input_ids == model.config.image_token_index
Tested model  : llava-hf/llava-v1.6-mistral-7b-hf
                llava-hf/llava-v1.6-vicuna-7b-hf
"""

from __future__ import annotations
from typing import List, Optional

import torch
from .base import BaseAttentionPlugin


class LlavaNextPlugin(BaseAttentionPlugin):
    """
    AttentionSteerPlugin for LLaVA-Next (LlavaNextForConditionalGeneration).

    The language backbone (Mistral / Vicuna / Llama) is stored at
    ``model.language_model``.  Image tokens in ``input_ids`` all carry the
    single special id ``model.config.image_token_index`` (default 32000).

    Example usage::

        from transformers import LlavaNextForConditionalGeneration, AutoProcessor
        from src.model.llava_next import LlavaNextPlugin

        model = LlavaNextForConditionalGeneration.from_pretrained(...)
        plugin = LlavaNextPlugin(model, mode="image", learnable=True)
        plugin.apply()
    """

    def __init__(
        self,
        model,
        boost_strength: float = 1.0,
        mode: str = "image",
        layer_range: Optional[List[int]] = None,
        learnable: bool = False,
        free_train: bool = True,
    ):
        # Resolve image token id from model config
        self.image_token_index: int = getattr(
            model.config, "image_token_index", 32000
        )
        super().__init__(model, boost_strength, mode, layer_range, learnable, free_train)

    # ------------------------------------------------------------------
    def _get_layers(self) -> list:
        # LlavaNextForConditionalGeneration -> language_model (e.g. MistralForCausalLM)
        #   -> model (MistralModel) -> layers
        return self.model.language_model.model.layers

    def _build_img_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids == self.image_token_index
