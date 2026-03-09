"""
Base class for all Attention Intervention plugins.

Each model-specific subclass only needs to implement:
  - `_get_layers()` -> list of transformer decoder layers
  - `_build_img_mask(input_ids)` -> BoolTensor [B, L] marking image token positions
"""

from __future__ import annotations
import types
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn


class BaseAttentionPlugin(nn.Module, ABC):
    """
    Injects a per-token additive bias into the attention mask before softmax,
    steering attention toward (or away from) image / text tokens.

    The bias applied to layer l at key position j is:
        bias[b, :, :, j] = pattern[b, j] * boost_strength[layer_idx]

    When `learnable=True`, boost_strength is an nn.Parameter (one value per
    targeted layer) trained end-to-end while the backbone is frozen.
    """

    def __init__(
        self,
        model: nn.Module,
        boost_strength: float = 1.0,
        mode: str = "image",
        layer_range: Optional[List[int]] = None,
        learnable: bool = False,
    ):
        super().__init__()
        self.model = model
        self.mode = mode          # "image" | "text" | "both" | "oppose"
        self.learnable = learnable

        # Resolve total number of transformer layers
        self.num_layers = self._count_layers()

        # Default: steer only the deeper half of the network
        if layer_range is None:
            self.layer_range = list(range(self.num_layers // 2, self.num_layers))
        else:
            self.layer_range = layer_range

        # Learnable per-layer scalar OR fixed scalar
        if learnable:
            self.boost_strength = nn.Parameter(
                torch.ones(len(self.layer_range)) * boost_strength
            )
        else:
            self.boost_strength = boost_strength

        self._orig_forwards: dict = {}
        self.bias_pattern: Optional[torch.Tensor] = None   # [B, L] float

    # ------------------------------------------------------------------
    # Abstract interface – implement in each model-specific subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_layers(self) -> list:
        """Return the list of all decoder layers (nn.ModuleList or list)."""
        raise NotImplementedError

    @abstractmethod
    def _build_img_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return a bool tensor [B, L] that is True for image token positions."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers that depend on the abstract methods above
    # ------------------------------------------------------------------

    def _count_layers(self) -> int:
        try:
            layers = self._get_layers()
            return len(layers)
        except Exception:
            # Fallback: read from config
            cfg = getattr(self.model, "config", None)
            for attr in ["num_hidden_layers",
                         "text_config.num_hidden_layers",
                         "llm_config.num_hidden_layers"]:
                obj = cfg
                for part in attr.split("."):
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                if isinstance(obj, int):
                    return obj
            raise AttributeError(
                "Cannot determine num_hidden_layers. "
                "Override _count_layers() in your subclass."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_masks(self, input_ids: torch.Tensor, **kwargs):
        """Compute and store the bias pattern from the current input_ids."""
        img_mask = self._build_img_mask(input_ids)            # [B, L] bool
        pad_id = getattr(self.model.config, "pad_token_id", 0)
        text_mask = (~img_mask) & (input_ids != pad_id)       # [B, L] bool

        B, L = input_ids.shape
        pattern = torch.zeros((B, L), dtype=torch.float32, device=input_ids.device)

        if self.mode == "image":
            pattern[img_mask] = 1.0
        elif self.mode == "text":
            pattern[text_mask] = 1.0
        elif self.mode == "both":
            pattern[img_mask | text_mask] = 1.0
        elif self.mode == "oppose":
            pattern[img_mask] = 1.0
            pattern[text_mask] = -1.0

        self.bias_pattern = pattern   # [B, L]

    def apply(self):
        """Patch self_attn.forward on every targeted layer."""
        layers = self._get_layers()

        def make_patched(orig_forward, layer_idx: int):
            plugin = self   # capture self

            def patched(self_attn, *args, **kwargs):
                # attention_mask may arrive as a keyword arg or as args[2]
                # Qwen-style eager: kwargs["attention_mask"]
                # Some models pass it positionally (args index varies by impl)
                attn_mask = kwargs.get("attention_mask", None)

                if (
                    plugin.bias_pattern is not None
                    and attn_mask is not None
                ):
                    k_len   = attn_mask.shape[-1]
                    bias_len = plugin.bias_pattern.shape[1]

                    if k_len >= bias_len:
                        device, dtype = attn_mask.device, attn_mask.dtype
                        bp = plugin.bias_pattern.to(device=device, dtype=dtype)

                        # Pad for auto-regressive token generation
                        if k_len > bias_len:
                            pad = torch.zeros(
                                (bp.shape[0], k_len - bias_len),
                                dtype=dtype, device=device
                            )
                            bp = torch.cat([bp, pad], dim=1)

                        strength = plugin.boost_strength
                        if plugin.learnable:
                            strength = plugin.boost_strength[layer_idx]
                        if isinstance(strength, torch.Tensor):
                            strength = strength.to(device=device, dtype=dtype)

                        bias = bp.unsqueeze(1).unsqueeze(2) * strength
                        kwargs["attention_mask"] = attn_mask + bias

                return orig_forward(*args, **kwargs)

            return patched

        for idx, layer_num in enumerate(self.layer_range):
            self_attn = layers[layer_num].self_attn
            if layer_num not in self._orig_forwards:
                self._orig_forwards[layer_num] = self_attn.forward
                self_attn.forward = types.MethodType(
                    make_patched(self._orig_forwards[layer_num], idx),
                    self_attn
                )

        mode_str = self.mode.capitalize()
        strength_str = "Learnable" if self.learnable else str(self.boost_strength)
        print(
            f"[Plugin] {mode_str}Boost applied | "
            f"layers {self.layer_range[0]}-{self.layer_range[-1]} | "
            f"strength: {strength_str}"
        )

    def disable(self):
        """Restore original self_attn.forward on all patched layers."""
        layers = self._get_layers()
        for layer_num, orig in self._orig_forwards.items():
            layers[layer_num].self_attn.forward = orig
        self._orig_forwards.clear()
        self.bias_pattern = None
        print(f"[Plugin] {self.mode.capitalize()}Boost disabled.")

    # Convenience aliases
    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def eval(self):
        return self.train(False)
