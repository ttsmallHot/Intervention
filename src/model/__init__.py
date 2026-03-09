"""
src.model — Attention Intervention plugins for each supported architecture.

Supported:
  - Qwen2.5-VL  : Qwen25VLPlugin
  - Qwen3-VL    : Qwen3VLPlugin
  - LLaVA-Next  : LlavaNextPlugin
  - InternVL    : InternVLPlugin
  - Gemma-3     : Gemma3Plugin

Factory function::

    from src.model import build_plugin

    plugin = build_plugin("qwen2_5vl", model, boost_strength=1.0,
                          mode="image", learnable=True)
"""

from .base import BaseAttentionPlugin
from .qwen2_5_vl import Qwen25VLPlugin
from .qwen3_vl import Qwen3VLPlugin
from .llava import LlavaNextPlugin
from .internvl import InternVLPlugin
from .gemma3 import Gemma3Plugin

_REGISTRY = {
    "qwen2_5vl": Qwen25VLPlugin,
    "qwen3vl":   Qwen3VLPlugin,
    "llava":     LlavaNextPlugin,
    "internvl":  InternVLPlugin,
    "gemma3":    Gemma3Plugin,
}


def build_plugin(model_type: str, model, **kwargs) -> BaseAttentionPlugin:
    """
    Instantiate the correct plugin by model_type string.

    Args:
        model_type: one of "qwen2_5vl", "qwen3vl", "llava", "internvl"
        model: the loaded HuggingFace model instance
        **kwargs: forwarded to the plugin constructor
                  (boost_strength, mode, layer_range, learnable)
    """
    key = model_type.lower().replace("-", "_").replace(".", "_")
    # normalise common aliases
    key = key.replace("qwen2_5_vl", "qwen2_5vl").replace("qwen3_vl", "qwen3vl").replace("gemma_3", "gemma3")
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](model, **kwargs)


__all__ = [
    "BaseAttentionPlugin",
    "Qwen25VLPlugin",
    "Qwen3VLPlugin",
    "LlavaNextPlugin",
    "InternVLPlugin",
    "build_plugin",
]
