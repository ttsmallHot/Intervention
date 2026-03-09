"""
Attention Intervention plugin for InternVL 3.5 (and InternVL 2.x).

Layer access  : model.language_model.model.layers
Image tokens  : input_ids == model.img_context_token_id
Tested model  : OpenGVLab/InternVL3-5-4B  (LLM backbone: Qwen3-36L)

Note: ``img_context_token_id`` is set on the *model object* at load time
(not in config.json). Make sure to load the model via its own
``from_pretrained`` before initialising this plugin.
"""

from __future__ import annotations
from typing import List, Optional

import torch
from .base import BaseAttentionPlugin


class InternVLPlugin(BaseAttentionPlugin):
    """
    AttentionSteerPlugin for InternVL Chat models.

    InternVL wraps a vision encoder and a language model under
    ``model.language_model`` (a Qwen3 / InternLM model).
    Image context tokens in ``input_ids`` all share the same id stored in
    ``model.img_context_token_id``.

    Example usage::

        import torch
        from transformers import AutoModel, AutoTokenizer
        from src.model.internvl import InternVLPlugin

        model = AutoModel.from_pretrained("/code/InternVL3_5-4B", trust_remote_code=True,
                                          torch_dtype=torch.bfloat16, device_map="auto")
        plugin = InternVLPlugin(model, mode="image", learnable=True)
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
        # img_context_token_id is set dynamically on the model after loading
        self.img_context_token_id: Optional[int] = getattr(
            model, "img_context_token_id", None
        )
        super().__init__(model, boost_strength, mode, layer_range, learnable, free_train)

    # ------------------------------------------------------------------
    def _get_layers(self) -> list:
        # InternVLChatModel -> language_model (Qwen3ForCausalLM)
        #   -> model (Qwen3Model) -> layers
        return self.model.language_model.model.layers

    def _build_img_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.img_context_token_id is None:
            # Try to read it again in case it was set after __init__
            self.img_context_token_id = getattr(
                self.model, "img_context_token_id", None
            )
        if self.img_context_token_id is None:
            raise ValueError(
                "img_context_token_id is not set on the InternVL model. "
                "Make sure to load the model before creating InternVLPlugin."
            )
        return input_ids == self.img_context_token_id
