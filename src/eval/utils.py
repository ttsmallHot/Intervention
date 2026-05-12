import torch

from src.common.image import prepare_image
from src.common.models import load_model_and_processor as load_model


def infer_one(model, processor, image, prompt: str, plugin,
              max_new_tokens: int = 10, compute_rapt: bool = False) -> tuple[str, dict]:
    """Run model on one sample; return (predicted_text, rapt_dict)."""
    image = prepare_image(image)

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

    del inputs, gen_ids, gen_ids_sliced
    torch.cuda.empty_cache()

    return text_out, rapt
