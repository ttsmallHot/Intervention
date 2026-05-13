"""
Saliency-based Key Patch Attention Analysis.

Workflow
--------
1. Use the FINAL RL checkpoint to compute grad_att saliency for every sample
   and pick the top-k image patches as the "key patches" for that sample.
2. With those key patches fixed, measure the last-token attention on
   key vs non-key image patches at every checkpoint (base + all RL steps).
3. Plot how attention to key patches evolves across RL training.

Notes
-----
- All attention statistics use the last-token row  attn_mean[-1, :]  for
  consistency with how the intervention plugin actually steers attention.
- grad_att follows mllms_know-main:  attn * ReLU(dLoss/dAttn),
  loss = -CE(last-token logits, argmax),  averaged over selected layers.
- Image token positions and counts are deterministic across checkpoints
  (same processor, same image, same prompt), so the key_patch_mask
  computed on the RL checkpoint applies unchanged to base & intermediates.
"""

import os
import sys
import json
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import (
    AttentionConfig,
    DataLoader,
    ModelManager,
    QwenVLAttentionAnalyzer,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. Saliency extractor (grad_att on a single model+sample)
# ============================================================

class SaliencyExtractor:
    """Compute grad_att saliency over image patches for a loaded model."""

    def __init__(
        self,
        analyzer: QwenVLAttentionAnalyzer,
        saliency_layers: str = "deep",  # "deep" | "shallow" | "all"
    ):
        self.analyzer = analyzer
        self.saliency_layers = saliency_layers

    def _layer_indices(self, num_layers: int) -> List[int]:
        if self.saliency_layers == "deep":
            return list(range(num_layers // 2, num_layers))
        if self.saliency_layers == "shallow":
            return list(range(0, num_layers // 2))
        return list(range(num_layers))

    def compute(self, inputs: Dict, image_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns saliency [num_img_tokens] on CPU.
            saliency = mean_layers( attn * ReLU(dL/dAttn) )[last_token_row][image_tokens]
        """
        model = self.analyzer.model
        model.eval()

        # Need grad through attentions; HF models have requires_grad=True by default.
        outputs = model(**inputs, output_attentions=True, return_dict=True)
        attentions = outputs.attentions  # tuple of [1, H, S, S]

        logits = outputs.logits[:, -1, :]                # [1, V]
        target = torch.argmax(logits, dim=-1)            # [1]
        loss = -F.cross_entropy(logits, target)          # scalar; sign matches mllms_know

        layer_ids = self._layer_indices(len(attentions))
        tracked = [attentions[l] for l in layer_ids]

        grads = torch.autograd.grad(
            loss, tracked, retain_graph=False, allow_unused=False
        )

        sals = []
        for attn, g in zip(tracked, grads):
            grad_att = attn * F.relu(g)                  # [1, H, S, S]
            last_row = grad_att[0].mean(dim=0)[-1]       # [S]  last-token, head-avg
            sals.append(last_row.detach().float().cpu())

        # average over selected layers
        sal_full = torch.stack(sals).mean(dim=0)         # [S]
        sal_img = sal_full[image_mask.cpu()]             # [num_img_tokens]
        return sal_img

    @staticmethod
    def topk_mask(saliency: torch.Tensor, k_ratio: float = 0.1) -> torch.Tensor:
        """Boolean mask selecting the top-k_ratio fraction of patches by saliency."""
        n = saliency.shape[0]
        k = max(1, int(round(n * k_ratio)))
        thresh = torch.topk(saliency, k).values.min()
        return saliency >= thresh


# ============================================================
# 2. Key-patch attention tracker
# ============================================================

class KeyPatchTracker:
    """Measures last-token attention on key vs non-key image patches per layer."""

    @staticmethod
    @torch.no_grad()
    def measure(
        attentions: List[torch.Tensor],
        image_mask: torch.Tensor,
        key_mask: torch.Tensor,  # bool over image tokens only [num_img_tokens]
    ) -> Dict[str, np.ndarray]:
        """
        Returns:
            {
                'key_per_layer'   : [L]  mean last-token attention on key patches
                'nonkey_per_layer': [L]  mean last-token attention on non-key patches
            }
        """
        img_mask_cpu = image_mask.cpu()
        key_cpu = key_mask.cpu()

        key_pl, nonkey_pl = [], []
        for attn in attentions:
            # attn: [1, H, S, S]  ->  head-avg, last-token row
            row = attn[0].mean(dim=0)[-1].float().cpu()  # [S]
            img_row = row[img_mask_cpu]                  # [N_img]
            key_pl.append(img_row[key_cpu].mean().item())
            nonkey_pl.append(img_row[~key_cpu].mean().item())

        return {
            "key_per_layer": np.array(key_pl),
            "nonkey_per_layer": np.array(nonkey_pl),
        }


# ============================================================
# 3. Plotting
# ============================================================

class KeyPatchPlotter:

    @staticmethod
    def plot_evolution(
        results: Dict[int, Dict[str, np.ndarray]],
        save_path: str,
        layer_range: str = "deep",
    ):
        """
        results: {step: {'key_per_layer': [L], 'nonkey_per_layer': [L]}}
                 (already averaged over samples)
        Plots key vs non-key average attention, and their ratio, across steps.
        """
        steps = sorted(results.keys())

        def select(arr: np.ndarray) -> float:
            L = len(arr)
            if layer_range == "deep":
                return float(arr[L // 2:].mean())
            if layer_range == "shallow":
                return float(arr[:L // 2].mean())
            return float(arr.mean())

        key_vals = [select(results[s]["key_per_layer"]) for s in steps]
        nonkey_vals = [select(results[s]["nonkey_per_layer"]) for s in steps]
        ratios = [k / (n + 1e-12) for k, n in zip(key_vals, nonkey_vals)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(steps, key_vals, marker="o", linewidth=2, color="tab:red", label="Key patches")
        ax.plot(steps, nonkey_vals, marker="s", linewidth=2, color="tab:blue", label="Non-key patches")
        ax.set_xlabel("RL Training Step")
        ax.set_ylabel(f"Mean last-token attention ({layer_range} layers)")
        ax.set_title("Absolute attention on key vs non-key patches")
        ax.set_xticks(steps)
        ax.set_xticklabels([("Base" if s == 0 else str(s)) for s in steps], rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1]
        ax.plot(steps, ratios, marker="o", linewidth=2.5, color="tab:purple")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
        ax.set_xlabel("RL Training Step")
        ax.set_ylabel("Key / Non-key attention ratio")
        ax.set_title("Concentration on key patches")
        ax.set_xticks(steps)
        ax.set_xticklabels([("Base" if s == 0 else str(s)) for s in steps], rotation=45)
        ax.grid(True, alpha=0.3)

        plt.suptitle(f"Key-patch attention evolution under RL ({layer_range})", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  saved: {save_path}")

    @staticmethod
    def plot_layer_heatmap(
        results: Dict[int, Dict[str, np.ndarray]],
        save_path: str,
    ):
        """Heatmap of key/non-key ratio across (step, layer)."""
        steps = sorted(results.keys())
        ratios = []
        for s in steps:
            k = results[s]["key_per_layer"]
            n = results[s]["nonkey_per_layer"]
            ratios.append(k / (n + 1e-12))
        mat = np.stack(ratios, axis=0)  # [num_steps, num_layers]

        fig, ax = plt.subplots(figsize=(max(8, 0.25 * mat.shape[1]), max(4, 0.4 * mat.shape[0])))
        im = ax.imshow(mat, cmap="RdBu_r", aspect="auto",
                       vmin=max(0.5, mat.min()), vmax=min(3.0, mat.max()))
        ax.set_xlabel("Layer")
        ax.set_ylabel("Step")
        ax.set_yticks(range(len(steps)))
        ax.set_yticklabels([("Base" if s == 0 else str(s)) for s in steps])
        ax.set_title("Key/Non-key attention ratio  (deeper red = more concentrated on key patches)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ratio")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  saved: {save_path}")

    @staticmethod
    def _img_grid_shape(inputs: Dict, num_img_tokens: int) -> Tuple[int, int]:
        """Best-effort 2D shape for image tokens (Qwen2.5-VL: from image_grid_thw)."""
        if "image_grid_thw" in inputs:
            thw = inputs["image_grid_thw"][0].cpu().tolist()
            if len(thw) >= 3:
                _, h, w = thw[:3]
                # spatial merge of 2 in Qwen2.5-VL
                for merge in (2, 1):
                    if (h % merge == 0) and (w % merge == 0) and (h // merge) * (w // merge) == num_img_tokens:
                        return h // merge, w // merge
        side = int(np.sqrt(num_img_tokens))
        return side, side

    @staticmethod
    def plot_key_patches_on_image(
        sample: Dict,
        inputs: Dict,
        saliency: torch.Tensor,
        key_mask: torch.Tensor,
        save_path: str,
    ):
        """Overlay saliency heatmap and key-patch mask on the original image."""
        n = saliency.shape[0]
        h, w = KeyPatchPlotter._img_grid_shape(inputs, n)
        if h * w != n:
            h = w = int(np.sqrt(n))
            saliency = saliency[: h * w]
            key_mask = key_mask[: h * w]

        sal2d = saliency.numpy().reshape(h, w)
        key2d = key_mask.numpy().reshape(h, w).astype(np.float32)

        image = Image.open(sample["image"]).convert("RGB")
        img_size = image.size

        sal_norm = (sal2d - sal2d.min()) / (sal2d.max() - sal2d.min() + 1e-12)
        sal_resized = np.array(
            Image.fromarray((sal_norm * 255).astype(np.uint8)).resize(img_size, Image.BILINEAR)
        ) / 255.0
        key_resized = np.array(
            Image.fromarray((key2d * 255).astype(np.uint8)).resize(img_size, Image.NEAREST)
        ) / 255.0

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(image); axes[1].imshow(sal_resized, cmap="hot", alpha=0.6)
        axes[1].set_title("grad_att saliency (RL final)"); axes[1].axis("off")
        axes[2].imshow(image); axes[2].imshow(key_resized, cmap="Reds", alpha=0.55)
        axes[2].set_title(f"Top-k key patches ({int(key_mask.sum())}/{n})"); axes[2].axis("off")

        plt.suptitle(f"Q: {sample['question']}  |  GT: {sample.get('ground_truth','')}", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  saved: {save_path}")


# ============================================================
# 4. Driver
# ============================================================

def run_saliency_key_patch_analysis(
    base_model_path: str,
    checkpoints_dir: str,
    json_path: str,
    images_base_path: str,
    output_dir: str,
    max_samples: int = 20,
    vis_samples: int = 5,
    k_ratio: float = 0.10,
    saliency_layers: str = "deep",
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("Saliency-based Key Patch Attention Analysis")
    print("=" * 70)

    # --- data ---
    print("\n[1/5] Loading samples...")
    samples = DataLoader(json_path, images_base_path).load_samples(max_samples)
    if not samples:
        print("No samples loaded."); return

    # --- model list ---
    print("\n[2/5] Discovering checkpoints...")
    model_paths = ModelManager(base_model_path, checkpoints_dir).get_all_model_paths()
    if len(model_paths) < 2:
        print("Need at least 1 RL checkpoint in addition to base."); return
    final_step, final_path = model_paths[-1]
    print(f"  Final RL checkpoint: step {final_step}  ({final_path})")
    print(f"  Total checkpoints (incl. base): {len(model_paths)}")

    # --- analyzer & extractor ---
    config = AttentionConfig(model_name=base_model_path)
    analyzer = QwenVLAttentionAnalyzer(config)
    sal_extractor = SaliencyExtractor(analyzer, saliency_layers=saliency_layers)

    # =========================================================
    # Stage 1: load FINAL RL model, compute key_patch_mask per sample
    # =========================================================
    print(f"\n[3/5] Computing saliency on final RL model (step {final_step})...")
    analyzer.load_model(final_path)

    sample_cache: Dict[int, Dict] = {}
    vis_dir = os.path.join(output_dir, "key_patches")
    os.makedirs(vis_dir, exist_ok=True)

    for i, sample in enumerate(samples):
        inputs = analyzer.prepare_inputs(sample["image"], sample["question"])
        text_mask, image_mask, info = analyzer.get_token_type_masks(inputs)

        if info["num_image_tokens"] == 0:
            print(f"  [skip] sample {i}: no image tokens"); continue

        saliency = sal_extractor.compute(inputs, image_mask)
        key_mask = SaliencyExtractor.topk_mask(saliency, k_ratio=k_ratio)

        sample_cache[i] = {
            "image_mask": image_mask.cpu(),
            "key_mask": key_mask,
            "saliency": saliency,
            "inputs_meta": {"image_grid_thw": inputs.get("image_grid_thw", None)},
        }

        if i < vis_samples:
            KeyPatchPlotter.plot_key_patches_on_image(
                sample, inputs, saliency, key_mask,
                os.path.join(vis_dir, f"sample{i+1}_key_patches.png"),
            )
        if (i + 1) % 5 == 0:
            print(f"    saliency: {i + 1}/{len(samples)}")

    if not sample_cache:
        print("No usable samples after saliency stage."); return

    # =========================================================
    # Stage 2: for each checkpoint, measure attention on key vs non-key
    # =========================================================
    print("\n[4/5] Measuring key-patch attention across checkpoints...")
    # results[step] = {'key_per_layer': [L], 'nonkey_per_layer': [L]} averaged over samples
    results: Dict[int, Dict[str, np.ndarray]] = {}

    for step, path in model_paths:
        print(f"\n  === Step {step} ===  {path}")
        analyzer.load_model(path)

        key_acc, nonkey_acc, n_used = None, None, 0
        for i, sample in enumerate(samples):
            if i not in sample_cache:
                continue
            inputs = analyzer.prepare_inputs(sample["image"], sample["question"])
            attentions = analyzer.extract_attention_weights(inputs)
            stats = KeyPatchTracker.measure(
                attentions,
                sample_cache[i]["image_mask"],
                sample_cache[i]["key_mask"],
            )
            if key_acc is None:
                key_acc = stats["key_per_layer"].copy()
                nonkey_acc = stats["nonkey_per_layer"].copy()
            else:
                key_acc += stats["key_per_layer"]
                nonkey_acc += stats["nonkey_per_layer"]
            n_used += 1

        results[step] = {
            "key_per_layer": key_acc / n_used,
            "nonkey_per_layer": nonkey_acc / n_used,
        }
        k_d = results[step]["key_per_layer"][len(results[step]["key_per_layer"]) // 2:].mean()
        n_d = results[step]["nonkey_per_layer"][len(results[step]["nonkey_per_layer"]) // 2:].mean()
        print(f"    deep-layer mean   key={k_d:.5f}  nonkey={n_d:.5f}  ratio={k_d / (n_d + 1e-12):.3f}")

    # =========================================================
    # Stage 3: plot
    # =========================================================
    print("\n[5/5] Plotting...")
    KeyPatchPlotter.plot_evolution(
        results, os.path.join(output_dir, "evolution_deep.png"), layer_range="deep"
    )
    KeyPatchPlotter.plot_evolution(
        results, os.path.join(output_dir, "evolution_shallow.png"), layer_range="shallow"
    )
    KeyPatchPlotter.plot_evolution(
        results, os.path.join(output_dir, "evolution_all.png"), layer_range="all"
    )
    KeyPatchPlotter.plot_layer_heatmap(
        results, os.path.join(output_dir, "layer_heatmap.png")
    )

    # --- dump raw numbers ---
    serial = {
        "config": {
            "base_model": base_model_path,
            "checkpoints_dir": checkpoints_dir,
            "k_ratio": k_ratio,
            "saliency_layers": saliency_layers,
            "num_samples_used": len(sample_cache),
            "final_step": final_step,
        },
        "results": {
            str(s): {
                "key_per_layer": results[s]["key_per_layer"].tolist(),
                "nonkey_per_layer": results[s]["nonkey_per_layer"].tolist(),
            } for s in sorted(results.keys())
        },
    }
    with open(os.path.join(output_dir, "key_patch_report.json"), "w", encoding="utf-8") as f:
        json.dump(serial, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Output dir: {output_dir}")


# ============================================================
if __name__ == "__main__":
    BASE_MODEL_PATH = "/code/Qwen2.5-VL-3B-Instruct"
    CHECKPOINTS_DIR = "/code/verl-agent/checkpoints/grpo_qwen2.5_vl_3b"
    JSON_PATH = "/code/verl-agent/AgentRL/sokoban_dataset/annotations.json"
    IMAGES_BASE_PATH = "/code/verl-agent/AgentRL/sokoban_dataset"
    OUTPUT_DIR = "./saliency_key_patch_analysis"

    MAX_SAMPLES = 20
    VIS_SAMPLES = 5
    K_RATIO = 0.10           # top-10% patches are "key"
    SALIENCY_LAYERS = "deep" # which layers grad_att aggregates over

    run_saliency_key_patch_analysis(
        base_model_path=BASE_MODEL_PATH,
        checkpoints_dir=CHECKPOINTS_DIR,
        json_path=JSON_PATH,
        images_base_path=IMAGES_BASE_PATH,
        output_dir=OUTPUT_DIR,
        max_samples=MAX_SAMPLES,
        vis_samples=VIS_SAMPLES,
        k_ratio=K_RATIO,
        saliency_layers=SALIENCY_LAYERS,
    )
