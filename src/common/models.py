"""Model loaders shared by train and eval phases."""

import torch
from transformers import AutoProcessor


def _load_qwen2_5vl(model_path):
    from transformers import Qwen2_5_VLForConditionalGeneration
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )

def _load_qwen3vl(model_path):
    from transformers import Qwen3VLForConditionalGeneration
    return Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )

def _load_llava(model_path):
    from transformers import LlavaNextForConditionalGeneration
    return LlavaNextForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )

def _load_internvl(model_path):
    import transformers
    return transformers.AutoModel.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )

def _load_gemma3(model_path):
    from transformers import Gemma3ForConditionalGeneration
    return Gemma3ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )


MODEL_LOADERS = {
    "qwen2_5vl": _load_qwen2_5vl,
    "qwen3vl":   _load_qwen3vl,
    "llava":     _load_llava,
    "internvl":  _load_internvl,
    "gemma3":    _load_gemma3,
}


def load_model_and_processor(cfg: dict):
    model_type = cfg["model_type"]
    model_path = cfg["model_path"]
    if model_type not in MODEL_LOADERS:
        raise ValueError(f"Unsupported model_type: {model_type}")
    model = MODEL_LOADERS[model_type](model_path)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor
