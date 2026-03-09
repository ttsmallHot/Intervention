"""
Attention Intervention plugin for LLaVA-Next  (llava-v1.6-*-hf).

Layer access  : model.language_model.model.layers
Image tokens  : resolved via dummy-forward hook (LLaVA expands <image>
                placeholder into many visual tokens internally, so we
                cannot build the mask from input_ids alone)
Tested model  : llava-hf/llava-v1.6-mistral-7b-hf
                llava-hf/llava-v1.6-vicuna-7b-hf
"""

from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn
from .base import BaseAttentionPlugin


class LlavaNextPlugin(BaseAttentionPlugin):
    """
    AttentionSteerPlugin for LLaVA-Next (LlavaNextForConditionalGeneration).

    Key difference vs Qwen-VL plugins:
    LLaVA stores only a single <image> placeholder in input_ids; the real
    visual tokens are expanded inside model.forward().  Therefore we cannot
    build the bias_pattern from input_ids directly.  Instead we run a
    lightweight dummy forward with a hook to capture the expanded seq_len,
    then construct the pattern accordingly.

    Example usage::

        from transformers import LlavaNextForConditionalGeneration, AutoProcessor
        from src.model.llava import LlavaNextPlugin

        model = LlavaNextForConditionalGeneration.from_pretrained(...)
        plugin = LlavaNextPlugin(model, mode="image", learnable=True)
        plugin.apply()

        # Must pass pixel_values + image_sizes from processor output
        inputs = processor(...)
        plugin.update_masks(inputs["input_ids"],
                            pixel_values=inputs.get("pixel_values"),
                            image_sizes=inputs.get("image_sizes"))
    """

    def __init__(
        self,
        model,
        boost_strength: float = 1.0,
        mode: str = "image",
        layer_range: Optional[List[int]] = None,
        learnable: bool = False,
    ):
        self.image_token_index: int = getattr(model.config, "image_token_index", 32000)
        super().__init__(model, boost_strength, mode, layer_range, learnable)

    # ------------------------------------------------------------------
    def _get_layers(self) -> list:
        return self.model.language_model.model.layers

    def _build_img_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Fallback: marks only the single placeholder position.
        # Real mask is built in update_masks() via dummy forward.
        return input_ids == self.image_token_index

    def _get_language_model(self) -> nn.Module:
        if hasattr(self.model, "language_model"):
            return self.model.language_model
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            return self.model.model.language_model
        return self.model

    # ------------------------------------------------------------------
    # Override update_masks: use dummy-forward + hook to get expanded seq_len
    # ------------------------------------------------------------------
    def update_masks(
        self,
        input_ids: torch.Tensor,
        pixel_values=None,
        image_sizes=None,
    ):
        """
        Build bias_pattern using the true expanded sequence length.

        Args:
            input_ids    : [B, L_orig]
            pixel_values : tensor from processor (required for image expansion)
            image_sizes  : tensor from processor (required for image expansion)
        """
        device = input_ids.device
        batch_size, orig_seq_len = input_ids.shape

        # --- Step 1: capture expanded seq_len via hook ---
        captured: dict = {}

        def _hook(module, args, kwargs):
            embeds = kwargs.get("inputs_embeds", None)
            if embeds is None and args:
                # Some versions pass inputs_embeds as the first positional arg
                for a in args:
                    if isinstance(a, torch.Tensor) and a.dim() == 3:
                        embeds = a
                        break
            if embeds is not None:
                captured["seq_len"] = embeds.shape[1]

        lang_model = self._get_language_model()
        handle = lang_model.register_forward_pre_hook(_hook, with_kwargs=True)

        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            try:
                fwd_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "use_cache": False,
                }
                if pixel_values is not None:
                    fwd_kwargs["pixel_values"] = pixel_values
                if image_sizes is not None:
                    fwd_kwargs["image_sizes"] = image_sizes
                self.model(**fwd_kwargs)
            except Exception as e:
                pass  # hook already fired before any error
            finally:
                handle.remove()

        if was_training:
            self.model.train()

        if "seq_len" not in captured:
            # Fallback: use base class behaviour
            super().update_masks(input_ids)
            return

        expanded_len = captured["seq_len"]

        # --- Step 2: build img_pattern over expanded length ---
        img_pattern = torch.zeros(
            (batch_size, expanded_len), dtype=torch.float32, device=device
        )

        for b in range(batch_size):
            ids = input_ids[b]
            img_positions = (ids == self.image_token_index).nonzero(as_tuple=False).squeeze(-1)
            n_ph = img_positions.numel()
            if n_ph == 0:
                continue

            extra = expanded_len - orig_seq_len
            tokens_per_ph = (extra + n_ph) // max(n_ph, 1)

            offset = 0
            for idx_ph in range(n_ph):
                pos = img_positions[idx_ph].item()
                start = pos + offset
                end = min(start + tokens_per_ph, expanded_len)
                img_pattern[b, start:end] = 1.0
                offset += tokens_per_ph - 1

        # --- Step 3: apply mode ---
        pattern = torch.zeros_like(img_pattern)
        if self.mode == "image":
            pattern = img_pattern.clone()
        elif self.mode == "text":
            pattern = (img_pattern == 0).float()
        elif self.mode == "both":
            pattern = torch.ones_like(img_pattern)
        elif self.mode == "oppose":
            pattern[img_pattern == 1.0] = 1.0
            pattern[img_pattern == 0.0] = -1.0

        self.bias_pattern = pattern
