"""
FrozenLake multi-turn task success rate evaluation.
Compares Base model vs Base + trained plugin.

Supports: qwen2_5vl | qwen3vl | llava | internvl

Usage
-----
python src/eval/eval_frozenlake.py --config configs/qwen2_5_frozenlake.yaml

Optional overrides:
  --checkpoint   /path/to/best_plugin.pt
  --num_episodes 200
  --max_steps    20
  --output_dir   /tmp/eval_out
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime

import yaml
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor
from PIL import Image as PILImage

try:
    import gymnasium as gym
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map
except ImportError:
    raise ImportError("Install gymnasium: pip install gymnasium")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from src.model import build_plugin
from src.train.utils import load_checkpoint
from src.eval.utils import (
    parse_action, ACTION_LOOKUP,
    FROZEN_LAKE_SYSTEM_PROMPT, get_observation_prompt,
)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(cfg: dict):
    model_type = cfg["model_type"]
    model_path = cfg["model_path"]

    if model_type == "qwen2_5vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    elif model_type == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager"
        )
    elif model_type == "llava":
        from transformers import LlavaNextForConditionalGeneration
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager"
        )
    elif model_type == "internvl":
        import transformers
        model = transformers.AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto"
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def run_episode(
    env,
    model,
    processor,
    plugin,
    use_plugin: bool,
    max_steps: int = 20,
    verbose: bool = False,
) -> tuple[bool, int]:
    """
    Run one FrozenLake episode.

    Returns:
        (success, steps_taken)
    """
    obs, _ = env.reset()

    for step_idx in range(max_steps):
        img_array = env.render()
        img_pil   = PILImage.fromarray(img_array)

        obs_prompt = get_observation_prompt(is_first_step=(step_idx == 0))
        prompt_text = f"{FROZEN_LAKE_SYSTEM_PROMPT}\n\n{obs_prompt}"

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_pil},
                {"type": "text",  "text": prompt_text},
            ],
        }]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # LLaVA-Next needs qwen_vl_utils for image pre-processing
        model_type = getattr(processor, "_model_type", None)
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
        except Exception:
            inputs = processor(
                text=[text],
                images=[img_pil],
                padding=True,
                return_tensors="pt",
            ).to(model.device)

        if use_plugin and plugin is not None:
            plugin.update_masks(
                inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                image_sizes=inputs.get("image_sizes"),
            )

        with torch.no_grad():
            gen_ids  = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            gen_ids  = gen_ids[0][inputs["input_ids"].shape[1]:]
            response = processor.decode(gen_ids, skip_special_tokens=True)

        action_str = parse_action(response)
        if verbose:
            print(f"  step={step_idx+1}  resp={response[:80]}  action={action_str}")

        if action_str is None:
            return False, step_idx + 1

        obs, reward, terminated, truncated, _ = env.step(ACTION_LOOKUP[action_str])
        if terminated or truncated:
            return reward > 0, step_idx + 1

    return False, max_steps


# ---------------------------------------------------------------------------
# Evaluate one mode
# ---------------------------------------------------------------------------

def evaluate_mode(
    mode_name: str,
    model,
    processor,
    plugin,
    use_plugin: bool,
    num_episodes: int = 200,
    max_steps:    int = 20,
    seed:         int = 42,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Mode: {mode_name}")
    print(f"{'='*60}")

    if plugin is not None:
        if use_plugin:
            plugin.apply()
        else:
            plugin.disable()

    successes   = 0
    total_steps = 0

    for ep in tqdm(range(num_episodes), desc=f"  {mode_name}"):
        np.random.seed(seed + ep)
        random_map = generate_random_map(size=4, p=0.8)
        env = gym.make(
            "FrozenLake-v1",
            desc=random_map,
            is_slippery=True,
            render_mode="rgb_array",
        )
        success, steps = run_episode(
            env, model, processor, plugin, use_plugin=use_plugin,
            max_steps=max_steps,
        )
        env.close()
        if success:
            successes += 1
        total_steps += steps

    rate     = successes / num_episodes
    avg_steps = total_steps / num_episodes
    result = {
        "mode":         mode_name,
        "success_rate": rate,
        "successes":    successes,
        "episodes":     num_episodes,
        "avg_steps":    avg_steps,
    }
    print(f"  Success : {rate:.2%} ({successes}/{num_episodes})")
    print(f"  Avg steps: {avg_steps:.2f}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Eval: FrozenLake task success rate")
    parser.add_argument("--config",       required=True)
    parser.add_argument("--checkpoint",   default=None)
    parser.add_argument("--num_episodes", type=int,   default=200)
    parser.add_argument("--max_steps",    type=int,   default=20)
    parser.add_argument("--output_dir",   default=None)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    checkpoint_path = args.checkpoint or os.path.join(
        cfg.get("output_dir", ""), "best_plugin.pt"
    )
    output_dir = cfg.get("output_dir", "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("\n[1] Loading model...")
    model, processor = load_model(cfg)
    for p in model.parameters():
        p.requires_grad = False

    # Load plugin (if checkpoint exists)
    plugin = None
    if os.path.exists(checkpoint_path):
        print(f"\n[2] Loading plugin: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        plugin = build_plugin(
            cfg["model_type"], model,
            boost_strength = 0.0,
            mode           = ckpt.get("mode", cfg.get("mode", "image")),
            layer_range    = ckpt.get("layer_range", None),
            learnable      = True,
        )
        load_checkpoint(plugin, checkpoint_path)
        print(f"    epoch={ckpt.get('epoch')}  val_acc={ckpt.get('val_acc', 0):.4f}")
        print(f"    strengths: {plugin.boost_strength.data.cpu().numpy().round(3)}")
    else:
        print(f"\n[!] Checkpoint not found: {checkpoint_path}. Only Base mode will be tested.")

    results = []

    # Base (no plugin)
    results.append(evaluate_mode(
        "Base (No Plugin)", model, processor,
        plugin=plugin, use_plugin=False,
        num_episodes=args.num_episodes, max_steps=args.max_steps, seed=args.seed,
    ))

    # Base + trained plugin
    if plugin is not None:
        results.append(evaluate_mode(
            "Base + Plugin", model, processor,
            plugin=plugin, use_plugin=True,
            num_episodes=args.num_episodes, max_steps=args.max_steps, seed=args.seed,
        ))

    # Summary
    print(f"\n{'='*60}")
    print("  Final Summary")
    print(f"{'='*60}")
    print(f"  {'Mode':<25} {'SuccessRate':>12}  {'Successes':>10}  {'AvgSteps':>9}")
    print(f"  {'-'*25} {'-'*12}  {'-'*10}  {'-'*9}")
    for r in results:
        print(
            f"  {r['mode']:<25} {r['success_rate']:>11.2%}  "
            f"{r['successes']:>5}/{r['episodes']:<4}  {r['avg_steps']:>9.2f}"
        )
    if len(results) == 2:
        delta = results[1]["success_rate"] - results[0]["success_rate"]
        print(f"\n  Plugin effect: {'+' if delta>=0 else ''}{delta:.2%}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"eval_frozenlake_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
