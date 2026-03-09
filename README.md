# Intervention — The code for paper "RL Teaches VLMs to Look: From Mechanistic Insights on Attention Shift to Efficient Adaptation"

Steers a frozen multimodal LLM's attention toward image tokens by injecting learned per-layer scalar biases into the attention mask.

## Project Structure

```
Intervention/
├── src/
│   ├── model/          # Per-architecture plugins
│   │   ├── base.py     # Abstract base class
│   │   ├── qwen2_5_vl.py
│   │   ├── qwen3_vl.py
│   │   ├── llava.py
│   │   └── internvl.py
│   ├── train/
│   │   ├── train.py
│   │   └── utils.py
│   └── eval/
│       ├── eval_vqa.py
│       └── utils.py
└── configs/
    ├── *_train.yaml    # Training configs
    └── *_eval.yaml     # Eval configs
```

## Data Format

Training and evaluation both expect a **Parquet file** where each row is one VQA sample.

**Required columns**

| Column  | Type | Description |
|---------|------|-------------|
| `image` | `dict` with key `bytes` (PNG/JPEG bytes) | Input image — HuggingFace datasets format |
| `prompt` | `str` | Question asked to the model |
| `label`  | `str` \| `int` | Ground-truth answer (compared as string) |

Any extra columns (e.g. `id`, `hole_positions`) are silently ignored.

**Example row**

```python
{
    "id":     0,
    "prompt": "How many holes are in this map? Answer with only the number.",
    "label":  "2",
    "image":  {"bytes": b"\x89PNG\r\n..."},   # raw PNG bytes
}
```

**Creating a compatible dataset**

```python
from datasets import Dataset
from PIL import Image
import io

def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

rows = [
    {"image": {"bytes": pil_to_bytes(img)}, "prompt": q, "label": str(a)}
    for img, q, a in your_data
]
Dataset.from_list(rows).to_parquet("train.parquet")
```

## Usage

**Train**
```bash
python src/train/train.py --config configs/qwen2_5_frozenlake_train.yaml
```

**Evaluate (VQA accuracy)**
```bash
python src/eval/eval_vqa.py --config configs/qwen2_5_frozenlake_eval.yaml
```
