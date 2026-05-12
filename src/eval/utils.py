import torch
from transformers import AutoProcessor

def load_model(cfg: dict):
    model_type = cfg["model_type"]
    model_path = cfg["model_path"]

    if model_type == "qwen2_5vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager",
        )
    elif model_type == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager",
        )
    elif model_type == "llava":
        from transformers import LlavaNextForConditionalGeneration
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager",
        )
    elif model_type == "internvl":
        import transformers
        model = transformers.AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto",
        )
    elif model_type == "gemma3":
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager",
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def infer_one(model, processor, image, prompt: str, plugin, max_new_tokens: int = 10, compute_rapt: bool = False) -> tuple[str, dict]:
    """
    Run model on one sample; return (predicted_text, rapt_dict).
    OOM protections (thumbnail processing, memory clearing) are included.
    """
    # Parse dict to PIL image if needed (from Parquet dict with "bytes")
    if isinstance(image, dict) and "bytes" in image:
        import io
        from PIL import Image
        image = Image.open(io.BytesIO(image["bytes"])).convert("RGB")

    # Protect against OOM from extremely large images
    if hasattr(image, 'width') and hasattr(image, 'height'):
        if image.width > 1536 or image.height > 1536:
            image.thumbnail((1536, 1536))

    inputs, _ = plugin.build_prompt(processor, image, prompt)
    inputs = inputs.to(model.device)

    plugin.update_masks(
        inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        image_sizes=inputs.get("image_sizes"),
    )

    rapt = None
    with torch.no_grad():
        if compute_rapt:
            out_with_attn = model(**inputs, output_attentions=True)
            rapt = plugin.compute_rapt(out_with_attn, inputs["input_ids"][0])
            del out_with_attn

        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        gen_ids_sliced = gen_ids[0][inputs["input_ids"].shape[1]:]
        text_out = processor.decode(gen_ids_sliced, skip_special_tokens=True)

    # Clear GPU memory aggressively
    inputs = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    del inputs
    del gen_ids
    del gen_ids_sliced
    torch.cuda.empty_cache()

    return text_out, rapt
