"""Image normalization shared by VQADataset and inference paths."""

import io

from PIL import Image


_MAX_SIDE = 1536


def prepare_image(image):
    """Decode parquet dict images and shrink oversized images to avoid OOM."""
    if isinstance(image, dict) and "bytes" in image:
        image = Image.open(io.BytesIO(image["bytes"])).convert("RGB")

    if hasattr(image, "width") and hasattr(image, "height"):
        if image.width > _MAX_SIDE or image.height > _MAX_SIDE:
            image.thumbnail((_MAX_SIDE, _MAX_SIDE))

    return image
