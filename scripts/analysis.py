"""
Qwen2.5-VL 注意力分析工具 (V6 - 增加跨模型注意力对比)

新增功能：
- 同一样本在不同模型间的注意力对比（差值/比值）
- 更直观展示RL训练带来的注意力分布变化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re

warnings.filterwarnings('ignore')

# ==================== 1. 配置 ====================

@dataclass
class AttentionConfig:
    """分析配置"""
    model_name: str = "/code/Qwen2.5-VL-3B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ==================== 2. 数据加载器 ====================

class DataLoader:
    """数据加载器"""
    
    def __init__(self, json_path: str, images_base_path: str):
        self.json_path = json_path
        self.images_base_path = images_base_path
        
    def load_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """加载样本"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            image_path = os.path.join(self.images_base_path, item["image"])
            if os.path.exists(image_path):
                samples.append({
                    "id": item["id"],
                    "image": image_path,
                    "question": item["prompt"],
                    "ground_truth": str(item["label"])
                })
        
        if max_samples:
            samples = samples[:max_samples]
            
        print(f"Loaded {len(samples)} samples from {self.json_path}")
        return samples

# ==================== 3. 模型管理器 ====================

class ModelManager:
    """管理多个模型的加载"""
    
    def __init__(self, base_model_path: str, checkpoints_dir: str):
        self.base_model_path = base_model_path
        self.checkpoints_dir = checkpoints_dir
        
    def get_all_model_paths(self) -> List[Tuple[int, str]]:
        """获取所有模型路径，返回 (step, path) 列表"""
        models = []
        
        # Base 模型 (step = 0)
        models.append((0, self.base_model_path))
        
        # Checkpoint 模型
        if os.path.exists(self.checkpoints_dir):
            for folder in sorted(os.listdir(self.checkpoints_dir)):
                match = re.match(r'global_step_(\d+)', folder)
                if match:
                    step = int(match.group(1))
                    actor_path = os.path.join(self.checkpoints_dir, folder, "actor")
                    if os.path.exists(actor_path):
                        models.append((step, actor_path))
                    else:
                        models.append((step, os.path.join(self.checkpoints_dir, folder)))
        
        models.sort(key=lambda x: x[0])
        return models

# ==================== 4. Qwen-VL 分析器核心类 ====================

class QwenVLAttentionAnalyzer:
    """Qwen2.5-VL 注意力分析器"""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.process_vision_info = None
        self.num_layers = None
        
    def load_model(self, model_path: str):
        """加载模型"""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        
        print(f"  Loading model: {model_path}")
        
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.config.dtype,
            device_map="auto",
            attn_implementation="eager"
        )
        
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.config.model_name)
            self.process_vision_info = process_vision_info
            
        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers
        print(f"  Model loaded! Layers: {self.num_layers}")
        
    def prepare_inputs(self, image_path: str, question: str) -> Dict:
        """准备模型输入"""
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image_path}, 
                    {"type": "text", "text": question}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text], 
            images=image_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(self.config.device)
        return inputs
    
    def get_token_type_masks(self, inputs: Dict) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """获取文本和图像token的mask"""
        input_ids = inputs["input_ids"][0]
        seq_len = input_ids.shape[0]
        
        try:
            vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        except:
            vision_start_id, vision_end_id = 151652, 151653
            
        image_mask = torch.zeros(seq_len, dtype=torch.bool, device=self.config.device)
        in_vision = False
        
        for i, token_id in enumerate(input_ids):
            if token_id.item() == vision_start_id: 
                in_vision = True
            elif token_id.item() == vision_end_id: 
                in_vision = False
            elif in_vision: 
                image_mask[i] = True
            
        text_mask = ~image_mask
        special_ids = {vision_start_id, vision_end_id, self.processor.tokenizer.pad_token_id, self.processor.tokenizer.eos_token_id}
        for sid in special_ids:
            if sid is not None:
                text_mask = text_mask & (input_ids != sid)
        
        info = {
            "seq_len": seq_len,
            "num_image_tokens": image_mask.sum().item(),
            "num_text_tokens": text_mask.sum().item(),
        }
        
        return text_mask, image_mask, info
    
    @torch.no_grad()
    def extract_attention_weights(self, inputs: Dict) -> List[torch.Tensor]:
        """提取所有层的注意力权重"""
        outputs = self.model(**inputs, output_attentions=True, return_dict=True)
        return [attn[0].float() for attn in outputs.attentions]
    
    @torch.no_grad()
    def generate_answer(self, inputs: Dict, max_new_tokens: int = 128) -> str:
        """生成答案"""
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        answer = self.processor.decode(output[0], skip_special_tokens=True)
        if "assistant" in answer.lower():
            answer = answer.split("assistant")[-1].strip()
        return answer

# ==================== 5. 跨模型 RAPT 分析模块 ====================

class CrossModelRAPTAnalyzer:
    """跨模型 RAPT 分析"""
    
    def __init__(self, analyzer: QwenVLAttentionAnalyzer):
        self.analyzer = analyzer

    def compute_rapt_for_sample(
        self, 
        attentions: List[torch.Tensor], 
        text_mask: torch.Tensor, 
        image_mask: torch.Tensor,
        layer_range: str = "all"
    ) -> Dict[str, float]:
        """计算单个样本的 RAPT"""
        num_layers = len(attentions)
        
        if layer_range == "shallow":
            indices = list(range(0, num_layers // 2))
        elif layer_range == "deep":
            indices = list(range(num_layers // 2, num_layers))
        else:
            indices = list(range(num_layers))
        
        rapt_text_list = []
        rapt_image_list = []
        
        num_text = text_mask.sum().item()
        num_image = image_mask.sum().item()
        seq_len = text_mask.shape[0]

        for layer_idx in indices:
            layer_attn = attentions[layer_idx]
            attn_mean = layer_attn.mean(dim=0)
            received_attn = attn_mean.sum(dim=0)
            global_avg = received_attn.sum().item() / seq_len
            
            if num_text > 0:
                text_received_total = received_attn[text_mask].sum().item()
                text_avg = text_received_total / num_text
                rapt_text = text_avg / global_avg
            else:
                rapt_text = 0.0
            
            if num_image > 0:
                image_received_total = received_attn[image_mask].sum().item()
                image_avg = image_received_total / num_image
                rapt_image = image_avg / global_avg
            else:
                rapt_image = 0.0
            
            rapt_text_list.append(rapt_text)
            rapt_image_list.append(rapt_image)
            
        return {
            "text": np.mean(rapt_text_list),
            "image": np.mean(rapt_image_list),
            "text_per_layer": rapt_text_list,
            "image_per_layer": rapt_image_list
        }

    def analyze_single_model(self, samples: List[Dict], layer_range: str = "all") -> Dict[str, float]:
        """分析单个模型在多个样本上的 RAPT"""
        all_rapt_text = []
        all_rapt_image = []
        
        for sample in samples:
            inputs = self.analyzer.prepare_inputs(sample["image"], sample["question"])
            text_mask, image_mask, _ = self.analyzer.get_token_type_masks(inputs)
            attentions = self.analyzer.extract_attention_weights(inputs)
            
            rapt = self.compute_rapt_for_sample(attentions, text_mask, image_mask, layer_range)
            all_rapt_text.append(rapt["text"])
            all_rapt_image.append(rapt["image"])
        
        return {
            "text_mean": np.mean(all_rapt_text),
            "text_std": np.std(all_rapt_text),
            "image_mean": np.mean(all_rapt_image),
            "image_std": np.std(all_rapt_image),
        }

# ==================== 6. 跨模型注意力对比可视化 (新增) ====================

class CrossModelAttentionComparator:
    """
    跨模型注意力对比可视化
    - 同一样本在不同模型间的注意力差异
    - 支持差值和比值两种对比方式
    """
    
    def __init__(self, analyzer: QwenVLAttentionAnalyzer):
        self.analyzer = analyzer
        
    def get_image_attention_2d(
        self, 
        attentions: List[torch.Tensor], 
        image_mask: torch.Tensor,
        layer_range: str = "deep"
    ) -> np.ndarray:
        """获取图像区域的2D注意力图"""
        num_layers = len(attentions)
        
        if layer_range == "deep":
            indices = list(range(num_layers // 2, num_layers))
        elif layer_range == "shallow":
            indices = list(range(0, num_layers // 2))
        else:
            indices = list(range(num_layers))
        
        layer_attns = []
        for idx in indices:
            attn_mean = attentions[idx].mean(dim=0)
            received = attn_mean.sum(dim=0)
            layer_attns.append(received)
        
        avg_received = torch.stack(layer_attns).mean(dim=0)
        img_attn = avg_received[image_mask]
        
        # Reshape to 2D
        n = img_attn.shape[0]
        side = int(np.sqrt(n))
        actual = side * side
        attn_np = img_attn[:actual].cpu().numpy()
        return attn_np.reshape(side, side)
    
    def extract_attention_for_sample(
        self, 
        image_path: str, 
        question: str
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """提取单个样本的注意力（深层和浅层）"""
        inputs = self.analyzer.prepare_inputs(image_path, question)
        text_mask, image_mask, _ = self.analyzer.get_token_type_masks(inputs)
        attentions = self.analyzer.extract_attention_weights(inputs)
        answer = self.analyzer.generate_answer(inputs)
        
        deep_attn = self.get_image_attention_2d(attentions, image_mask, "deep")
        shallow_attn = self.get_image_attention_2d(attentions, image_mask, "shallow")
        
        return deep_attn, shallow_attn, answer
    
    def normalize(self, arr: np.ndarray) -> np.ndarray:
        """归一化到[0,1]"""
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val > 1e-8:
            return (arr - min_val) / (max_val - min_val)
        return np.zeros_like(arr)
    
    def visualize_cross_model_comparison(
        self,
        sample: Dict,
        model_attentions: Dict[int, Tuple[np.ndarray, np.ndarray, str]],
        save_path: str,
        comparison_mode: str = "ratio"  # "ratio" or "diff"
    ):
        """
        可视化跨模型注意力对比
        
        Args:
            sample: 样本信息
            model_attentions: {step: (deep_attn, shallow_attn, answer)}
            save_path: 保存路径
            comparison_mode: "ratio" (比值) 或 "diff" (差值)
        """
        steps = sorted(model_attentions.keys())
        base_step = steps[0]  # 假设第一个是base
        base_deep, base_shallow, base_answer = model_attentions[base_step]
        
        # 加载原图
        image = Image.open(sample["image"]).convert("RGB")
        img_size = image.size
        
        # 计算需要显示的模型数（包括base）
        num_models = len(steps)
        
        # 创建图：3行 x num_models列
        # 第1行: 原始注意力(深层)
        # 第2行: 相对于base的变化(深层)
        # 第3行: 相对于base的变化(浅层)
        fig, axes = plt.subplots(3, num_models, figsize=(5 * num_models, 12))
        
        if num_models == 1:
            axes = axes.reshape(-1, 1)
        
        for col, step in enumerate(steps):
            deep_attn, shallow_attn, answer = model_attentions[step]
            model_name = f"Step {step}" if step > 0 else "Base"
            
            # 第1行: 原始深层注意力
            deep_norm = self.normalize(deep_attn)
            deep_resized = np.array(Image.fromarray(
                (deep_norm * 255).astype(np.uint8)
            ).resize(img_size, Image.BILINEAR)) / 255.0
            
            axes[0, col].imshow(image)
            axes[0, col].imshow(deep_resized, cmap='hot', alpha=0.6)
            axes[0, col].set_title(f"{model_name}\nDeep Attention", fontsize=11, fontweight='bold')
            axes[0, col].axis('off')
            
            # 第2行: 深层变化（相对于base）
            if step == base_step:
                # Base本身显示原始注意力
                axes[1, col].imshow(deep_norm, cmap='hot')
                axes[1, col].set_title(f"Deep Attn\n(Reference)", fontsize=10)
            else:
                if comparison_mode == "ratio":
                    # 比值: current / base
                    ratio = np.divide(deep_attn, base_deep + 1e-8)
                    # 使用对数比值更好展示
                    log_ratio = np.log2(ratio + 1e-8)
                    # 裁剪到合理范围
                    log_ratio = np.clip(log_ratio, -2, 2)
                    im = axes[1, col].imshow(log_ratio, cmap='RdBu_r', vmin=-2, vmax=2)
                    axes[1, col].set_title(f"Deep: log₂(Step{step}/Base)", fontsize=10)
                else:
                    # 差值: current - base
                    diff = deep_attn - base_deep
                    vmax = max(abs(diff.min()), abs(diff.max()))
                    im = axes[1, col].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                    axes[1, col].set_title(f"Deep: Step{step} - Base", fontsize=10)
                plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
            axes[1, col].axis('off')
            
            # 第3行: 浅层变化（相对于base）
            if step == base_step:
                shallow_norm = self.normalize(shallow_attn)
                axes[2, col].imshow(shallow_norm, cmap='hot')
                axes[2, col].set_title(f"Shallow Attn\n(Reference)", fontsize=10)
            else:
                if comparison_mode == "ratio":
                    ratio = np.divide(shallow_attn, base_shallow + 1e-8)
                    log_ratio = np.log2(ratio + 1e-8)
                    log_ratio = np.clip(log_ratio, -2, 2)
                    im = axes[2, col].imshow(log_ratio, cmap='RdBu_r', vmin=-2, vmax=2)
                    axes[2, col].set_title(f"Shallow: log₂(Step{step}/Base)", fontsize=10)
                else:
                    diff = shallow_attn - base_shallow
                    vmax = max(abs(diff.min()), abs(diff.max()))
                    im = axes[2, col].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                    axes[2, col].set_title(f"Shallow: Step{step} - Base", fontsize=10)
                plt.colorbar(im, ax=axes[2, col], fraction=0.046, pad=0.04)
            axes[2, col].axis('off')
        
        # 添加答案信息
        answers_text = " | ".join([f"Step{s}: {model_attentions[s][2][:50]}" for s in steps])
        plt.suptitle(f"Q: {sample['question']}\nGT: {sample.get('ground_truth', 'N/A')}\n"
                    f"Red=增强, Blue=减弱 (相对于Base)", 
                    fontsize=11, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Cross-model comparison saved: {save_path}")
    
    def visualize_attention_evolution(
        self,
        sample: Dict,
        model_attentions: Dict[int, Tuple[np.ndarray, np.ndarray, str]],
        save_path: str
    ):
        """
        可视化注意力随训练的演变（更紧凑的视图）
        一行显示所有模型的深层注意力
        """
        steps = sorted(model_attentions.keys())
        base_step = steps[0]
        base_deep, _, _ = model_attentions[base_step]
        
        image = Image.open(sample["image"]).convert("RGB")
        img_size = image.size
        
        num_models = len(steps)
        fig, axes = plt.subplots(2, num_models, figsize=(4 * num_models, 8))
        
        if num_models == 1:
            axes = axes.reshape(-1, 1)
        
        for col, step in enumerate(steps):
            deep_attn, _, answer = model_attentions[step]
            model_name = f"Step {step}" if step > 0 else "Base"
            
            # 第1行: 叠加在原图上
            deep_norm = self.normalize(deep_attn)
            deep_resized = np.array(Image.fromarray(
                (deep_norm * 255).astype(np.uint8)
            ).resize(img_size, Image.BILINEAR)) / 255.0
            
            axes[0, col].imshow(image)
            axes[0, col].imshow(deep_resized, cmap='hot', alpha=0.6)
            axes[0, col].set_title(f"{model_name}", fontsize=12, fontweight='bold')
            axes[0, col].axis('off')
            
            # 第2行: 相对变化
            if step == base_step:
                axes[1, col].imshow(deep_norm, cmap='hot')
                axes[1, col].set_title("Reference", fontsize=10)
            else:
                ratio = np.divide(deep_attn, base_deep + 1e-8)
                log_ratio = np.log2(ratio + 1e-8)
                log_ratio = np.clip(log_ratio, -2, 2)
                im = axes[1, col].imshow(log_ratio, cmap='RdBu_r', vmin=-2, vmax=2)
                axes[1, col].set_title(f"vs Base", fontsize=10)
                plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
            axes[1, col].axis('off')
        
        plt.suptitle(f"Attention Evolution (Deep Layers)\nQ: {sample['question']}", 
                    fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Attention evolution saved: {save_path}")
    
    def visualize_attention_change_summary(
        self,
        all_samples_attentions: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray, str]]],
        save_path: str
    ):
        """
        汇总所有样本的注意力变化
        
        Args:
            all_samples_attentions: {sample_idx: {step: (deep, shallow, answer)}}
        """
        sample_indices = sorted(all_samples_attentions.keys())
        first_sample = all_samples_attentions[sample_indices[0]]
        steps = sorted(first_sample.keys())
        base_step = steps[0]
        
        # 计算每个模型相对于base的平均变化
        avg_changes = {step: {"deep_increase": [], "deep_decrease": [], 
                              "shallow_increase": [], "shallow_decrease": []} 
                       for step in steps if step != base_step}
        
        for sample_idx in sample_indices:
            sample_attns = all_samples_attentions[sample_idx]
            base_deep, base_shallow, _ = sample_attns[base_step]
            
            for step in steps:
                if step == base_step:
                    continue
                deep_attn, shallow_attn, _ = sample_attns[step]
                
                # 计算变化比例
                deep_ratio = deep_attn / (base_deep + 1e-8)
                shallow_ratio = shallow_attn / (base_shallow + 1e-8)
                
                avg_changes[step]["deep_increase"].append((deep_ratio > 1.1).mean())
                avg_changes[step]["deep_decrease"].append((deep_ratio < 0.9).mean())
                avg_changes[step]["shallow_increase"].append((shallow_ratio > 1.1).mean())
                avg_changes[step]["shallow_decrease"].append((shallow_ratio < 0.9).mean())
        
        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        non_base_steps = [s for s in steps if s != base_step]
        x = np.arange(len(non_base_steps))
        width = 0.35
        
        # 深层
        deep_inc = [np.mean(avg_changes[s]["deep_increase"]) * 100 for s in non_base_steps]
        deep_dec = [np.mean(avg_changes[s]["deep_decrease"]) * 100 for s in non_base_steps]
        
        axes[0].bar(x - width/2, deep_inc, width, label='Increased (>10%)', color='red', alpha=0.7)
        axes[0].bar(x + width/2, deep_dec, width, label='Decreased (>10%)', color='blue', alpha=0.7)
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('% of Image Regions')
        axes[0].set_title('Deep Layers: Attention Change vs Base', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f'Step {s}' for s in non_base_steps])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 浅层
        shallow_inc = [np.mean(avg_changes[s]["shallow_increase"]) * 100 for s in non_base_steps]
        shallow_dec = [np.mean(avg_changes[s]["shallow_decrease"]) * 100 for s in non_base_steps]
        
        axes[1].bar(x - width/2, shallow_inc, width, label='Increased (>10%)', color='red', alpha=0.7)
        axes[1].bar(x + width/2, shallow_dec, width, label='Decreased (>10%)', color='blue', alpha=0.7)
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('% of Image Regions')
        axes[1].set_title('Shallow Layers: Attention Change vs Base', fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'Step {s}' for s in non_base_steps])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Summary: Image Attention Changes During RL Training', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Attention change summary saved: {save_path}")

# ==================== 7. 原有可视化模块 ====================

class AttentionVisualizer:
    """注意力可视化"""
    
    def __init__(self, analyzer: QwenVLAttentionAnalyzer):
        self.analyzer = analyzer
    
    def reshape_to_2d(self, attention: torch.Tensor) -> np.ndarray:
        n = attention.shape[0]
        side = int(np.sqrt(n))
        actual = side * side
        attn_np = attention[:actual].cpu().numpy()
        return attn_np.reshape(side, side)
    
    def normalize(self, arr: np.ndarray) -> np.ndarray:
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val > 1e-8:
            return (arr - min_val) / (max_val - min_val)
        return np.zeros_like(arr)

# ==================== 8. 跨模型绘图 ====================

class CrossModelPlotter:
    """跨模型结果绘图"""
    
    @staticmethod
    def plot_rapt_across_steps(results: Dict[int, Dict], save_path: str, layer_range: str = "all"):
        """绘制 RAPT 随训练步数变化的双轴图"""
        steps = sorted(results.keys())
        text_means = [results[s]["text_mean"] for s in steps]
        text_stds = [results[s]["text_std"] for s in steps]
        image_means = [results[s]["image_mean"] for s in steps]
        image_stds = [results[s]["image_std"] for s in steps]
        
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        color_image = 'tab:blue'
        ax1.set_xlabel('RL Training Steps', fontsize=14)
        ax1.set_ylabel('Image RAPT', color=color_image, fontsize=14)
        line1 = ax1.errorbar(steps, image_means, yerr=image_stds, 
                            color=color_image, marker='o', markersize=8, 
                            linewidth=2, capsize=5, label='Image RAPT')
        ax1.tick_params(axis='y', labelcolor=color_image)
        ax1.fill_between(steps, 
                         np.array(image_means) - np.array(image_stds),
                         np.array(image_means) + np.array(image_stds),
                         color=color_image, alpha=0.15)

        ax2 = ax1.twinx()
        color_text = 'tab:red'
        ax2.set_ylabel('Text RAPT', color=color_text, fontsize=14)
        line2 = ax2.errorbar(steps, text_means, yerr=text_stds,
                            color=color_text, marker='s', markersize=8,
                            linewidth=2, capsize=5, label='Text RAPT')
        ax2.tick_params(axis='y', labelcolor=color_text)
        ax2.fill_between(steps,
                         np.array(text_means) - np.array(text_stds),
                         np.array(text_means) + np.array(text_stds),
                         color=color_text, alpha=0.15)

        lines = [line1, line2]
        labels = ['Image RAPT', 'Text RAPT']
        ax1.legend(lines, labels, loc='upper right', fontsize=12)
        
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_xticks(steps)
        ax1.set_xticklabels([f'{s}' if s > 0 else 'Base' for s in steps], rotation=45)
        
        layer_desc = {"all": "All Layers", "shallow": "Shallow Layers", "deep": "Deep Layers"}
        plt.title(f'RAPT across RL Training Steps ({layer_desc.get(layer_range, layer_range)})', 
                  fontsize=16, pad=15)
        fig.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Cross-model RAPT plot saved: {save_path}")

    @staticmethod
    def plot_rapt_single_axis(results: Dict[int, Dict], save_path: str, layer_range: str = "all"):
        """单Y轴绘图"""
        steps = sorted(results.keys())
        text_means = [results[s]["text_mean"] for s in steps]
        text_stds = [results[s]["text_std"] for s in steps]
        image_means = [results[s]["image_mean"] for s in steps]
        image_stds = [results[s]["image_std"] for s in steps]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.errorbar(steps, text_means, yerr=text_stds, color='red',
                   marker='s', markersize=8, linewidth=2.5, capsize=5, label='Text RAPT')
        ax.errorbar(steps, image_means, yerr=image_stds, color='blue',
                   marker='o', markersize=8, linewidth=2.5, capsize=5, label='Image RAPT')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Baseline (RAPT=1)')
        
        ax.set_xlabel('RL Training Steps', fontsize=14)
        ax.set_ylabel('Relative Attention Per Token (RAPT)', fontsize=14)
        ax.set_xticks(steps)
        ax.set_xticklabels([f'{s}' if s > 0 else 'Base' for s in steps], rotation=45)
        
        layer_desc = {"all": "All Layers", "shallow": "Shallow Layers", "deep": "Deep Layers"}
        ax.set_title(f'RAPT: Text vs Image ({layer_desc.get(layer_range, layer_range)})', 
                    fontsize=16, pad=15)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Cross-model RAPT single-axis plot saved: {save_path}")

# ==================== 9. 主运行函数 ====================

def run_cross_model_analysis(
    base_model_path: str,
    checkpoints_dir: str,
    json_path: str,
    images_base_path: str,
    output_dir: str,
    max_samples: int = 10,
    compare_samples: int = 5
):
    """运行跨模型分析"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Qwen2.5-VL Cross-Model Attention Analysis (V6)")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1/6] Loading data...")
    data_loader = DataLoader(json_path, images_base_path)
    samples = data_loader.load_samples(max_samples)
    
    if not samples:
        print("❌ No samples loaded!")
        return
    
    # 2. 获取所有模型路径
    print("\n[2/6] Discovering models...")
    model_manager = ModelManager(base_model_path, checkpoints_dir)
    model_paths = model_manager.get_all_model_paths()
    
    print(f"  Found {len(model_paths)} models:")
    for step, path in model_paths:
        print(f"    - Step {step}: {path}")
    
    # 3. 初始化分析器
    print("\n[3/6] Initializing analyzer...")
    config = AttentionConfig(model_name=base_model_path)
    analyzer = QwenVLAttentionAnalyzer(config)
    rapt_analyzer = CrossModelRAPTAnalyzer(analyzer)
    comparator = CrossModelAttentionComparator(analyzer)
    
    # 4. 收集所有模型在所有样本上的注意力
    print("\n[4/6] Extracting attention from all models...")
    
    all_results = {}
    all_results_shallow = {}
    all_results_deep = {}
    
    # {sample_idx: {step: (deep_attn, shallow_attn, answer)}}
    all_samples_attentions = {i: {} for i in range(min(compare_samples, len(samples)))}
    
    for step, model_path in model_paths:
        print(f"\n  === Model: Step {step} ===")
        analyzer.load_model(model_path)
        
        # RAPT 分析
        print(f"    Computing RAPT...")
        all_results[step] = rapt_analyzer.analyze_single_model(samples, "all")
        all_results_shallow[step] = rapt_analyzer.analyze_single_model(samples, "shallow")
        all_results_deep[step] = rapt_analyzer.analyze_single_model(samples, "deep")
        
        print(f"    RAPT (all): Text={all_results[step]['text_mean']:.4f}, "
              f"Image={all_results[step]['image_mean']:.4f}")
        
        # 提取注意力用于对比
        print(f"    Extracting attention for comparison...")
        for i in range(min(compare_samples, len(samples))):
            sample = samples[i]
            deep_attn, shallow_attn, answer = comparator.extract_attention_for_sample(
                sample["image"], sample["question"]
            )
            all_samples_attentions[i][step] = (deep_attn, shallow_attn, answer)
    
    # 5. 生成跨模型对比可视化
    print("\n[5/6] Generating cross-model visualizations...")
    
    comparison_dir = os.path.join(output_dir, "cross_model_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    for i in range(min(compare_samples, len(samples))):
        sample = samples[i]
        print(f"\n  Sample {i+1}: {os.path.basename(sample['image'])}")
        
        # 完整对比视图
        comparator.visualize_cross_model_comparison(
            sample,
            all_samples_attentions[i],
            os.path.join(comparison_dir, f"sample{i+1}_comparison_ratio.png"),
            comparison_mode="ratio"
        )
        
        comparator.visualize_cross_model_comparison(
            sample,
            all_samples_attentions[i],
            os.path.join(comparison_dir, f"sample{i+1}_comparison_diff.png"),
            comparison_mode="diff"
        )
        
        # 紧凑演变视图
        comparator.visualize_attention_evolution(
            sample,
            all_samples_attentions[i],
            os.path.join(comparison_dir, f"sample{i+1}_evolution.png")
        )
    
    # 汇总统计
    comparator.visualize_attention_change_summary(
        all_samples_attentions,
        os.path.join(output_dir, "attention_change_summary.png")
    )
    
    # 6. 绘制RAPT图
    print("\n[6/6] Generating RAPT plots...")
    
    plotter = CrossModelPlotter()
    
    plotter.plot_rapt_across_steps(all_results, 
                                   os.path.join(output_dir, "rapt_dual_axis_all.png"), "all")
    plotter.plot_rapt_single_axis(all_results, 
                                  os.path.join(output_dir, "rapt_single_axis_all.png"), "all")
    
    plotter.plot_rapt_across_steps(all_results_shallow, 
                                   os.path.join(output_dir, "rapt_dual_axis_shallow.png"), "shallow")
    plotter.plot_rapt_across_steps(all_results_deep, 
                                   os.path.join(output_dir, "rapt_dual_axis_deep.png"), "deep")
    
    # 保存结果
    report = {
        "base_model": base_model_path,
        "checkpoints_dir": checkpoints_dir,
        "num_samples": len(samples),
        "rapt_results": {
            "all_layers": {str(k): v for k, v in all_results.items()},
            "shallow_layers": {str(k): v for k, v in all_results_shallow.items()},
            "deep_layers": {str(k): v for k, v in all_results_deep.items()},
        }
    }
    
    with open(os.path.join(output_dir, "analysis_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("✓ Analysis Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\n📁 Generated files:")
    print(f"  📊 RAPT plots:")
    print(f"     - rapt_dual_axis_*.png")
    print(f"     - rapt_single_axis_*.png")
    print(f"  🔥 Cross-model comparison (NEW):")
    print(f"     - cross_model_comparison/sample*_comparison_ratio.png (比值对比)")
    print(f"     - cross_model_comparison/sample*_comparison_diff.png  (差值对比)")
    print(f"     - cross_model_comparison/sample*_evolution.png        (演变视图)")
    print(f"     - attention_change_summary.png                        (汇总统计)")
    
    return all_results

# ==================== 10. 主程序入口 ====================

if __name__ == "__main__":
    
    BASE_MODEL_PATH = "/code/Qwen2.5-VL-3B-Instruct"
    CHECKPOINTS_DIR = "/code/verl-agent/checkpoints/grpo_qwen2.5_vl_3b"
    JSON_PATH = "/code/verl-agent/AgentRL/sokoban_dataset/annotations.json"
    IMAGES_BASE_PATH = "/code/verl-agent/AgentRL/sokoban_dataset"
    OUTPUT_DIR = "./cross_model_attention_analysis_v6"
    
    MAX_SAMPLES = 10       # 用于RAPT计算的样本数
    COMPARE_SAMPLES = 5    # 跨模型对比可视化的样本数
    
    # 检查路径
    print("Checking paths...")
    paths_ok = True
    
    for name, path in [("Base model", BASE_MODEL_PATH), 
                       ("Checkpoints", CHECKPOINTS_DIR),
                       ("JSON", JSON_PATH), 
                       ("Images", IMAGES_BASE_PATH)]:
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name} NOT FOUND: {path}")
            paths_ok = False
    
    if paths_ok:
        run_cross_model_analysis(
            BASE_MODEL_PATH,
            CHECKPOINTS_DIR,
            JSON_PATH,
            IMAGES_BASE_PATH,
            OUTPUT_DIR,
            MAX_SAMPLES,
            COMPARE_SAMPLES
        )
    else:
        print("\n❌ Please fix the paths above before running.")
