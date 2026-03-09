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
        free_train: bool = True,
    ):
        """
        Parameters
        ----------
        free_train : bool
            Only relevant when ``learnable=True``.
            True  → one independent scalar per layer  (per-layer, shape [N])
            False → single shared scalar across layers (unified,   shape [])
        """
        super().__init__()
        self.model = model
        self.mode = mode          # "image" | "text" | "both" | "oppose"
        self.learnable = learnable
        self.free_train = free_train

        # Resolve total number of transformer layers
        self.num_layers = self._count_layers()

        # Default: steer only the deeper half of the network
        if layer_range is None:
            self.layer_range = list(range(self.num_layers // 2, self.num_layers))
        else:
            self.layer_range = layer_range

        # Learnable:
        #   free_train=True  → shape [N], independent scalar per layer
        #   free_train=False → shape [1], one shared scalar for all layers
        # Hook always indexes [layer_idx] or [0] accordingly.
        if learnable:
            n = len(self.layer_range) if free_train else 1
            self.boost_strength = nn.Parameter(
                torch.ones(n) * boost_strength
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

    def build_prompt(self, processor, image, prompt_text: str) -> tuple:
        """
        Build processor inputs for a single sample.

        Returns:
            inputs  : dict of tensors ready for model()
            messages: the message list (for debugging)

        Default implementation: Qwen-style with embedded PIL in content.
        Override in subclasses that need different message formats (e.g. LLaVA).
        """
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt_text},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        return inputs, messages

    def compute_rapt(self, outputs, input_ids: torch.Tensor) -> dict:
        """
        Compute RAPT (Relative Attention Proportion to Target) for the last token.

        Uses `_build_img_mask` so each subclass automatically gets the correct
        image-token positions — no hardcoded token ids needed.

        Args:
            outputs   : model output with output_attentions=True (single sample)
            input_ids : [L] 1-D token id tensor (single sample)
        Returns:
            {"image": float, "text": float}
        """
        if outputs.attentions is None:
            return {"image": 0.0, "text": 0.0}

        num_layers  = len(outputs.attentions)
        deep_layers = outputs.attentions[num_layers // 2:]

        # Average over deep layers and all heads; last-token row
        avg_attn = torch.stack(
            [a[0].mean(dim=0).cpu().float() for a in deep_layers]
        ).mean(dim=0)                       # [seq, seq]

        seq_len = avg_attn.shape[-1]        # may differ from input_ids len (LLaVA expands)
        received = avg_attn[-1, :]          # [seq]

        # Build img_mask at the attention sequence length
        img_mask_2d = self._build_img_mask(input_ids.unsqueeze(0))  # [1, L_orig]
        img_mask    = img_mask_2d[0].cpu()                          # [L_orig] on CPU

        # If attention is over the expanded sequence (e.g. LLaVA), use bias_pattern
        if seq_len != img_mask.shape[0] and self.bias_pattern is not None:
            bp = self.bias_pattern[0]          # [L_expanded]
            if bp.shape[0] == seq_len:
                img_mask  = bp.cpu().bool()
        elif seq_len != img_mask.shape[0]:
            # Cannot align — return zeros
            return {"image": 0.0, "text": 0.0}

        pad_id    = getattr(self.model.config, "pad_token_id", 0)
        text_mask = (~img_mask) & (input_ids.cpu() != pad_id) \
                    if seq_len == input_ids.shape[0] \
                    else ~img_mask.cpu()

        global_mean = received.mean().item()
        if global_mean == 0:
            return {"image": 0.0, "text": 0.0}

        img_rapt  = (received[img_mask].mean().item()  / global_mean) if img_mask.any()  else 0.0
        text_rapt = (received[text_mask].mean().item() / global_mean) if text_mask.any() else 0.0
        return {"image": img_rapt, "text": text_rapt}

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
                            # free_train: each layer has its own scalar
                            # unified:    all layers share boost_strength[0]
                            idx = layer_idx if plugin.free_train else 0
                            strength = plugin.boost_strength[idx]
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
        if self.learnable:
            n = self.boost_strength.numel()
            kind = "per-layer" if self.free_train else "unified"
            strength_str = f"Learnable ({kind}, {n} param{'s' if n > 1 else ''})"
        else:
            strength_str = str(self.boost_strength)
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
