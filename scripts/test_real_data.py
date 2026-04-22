"""
测试用真实渲染数据的 pipeline。
"""
import os
import sys

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.data.dataset import WebpageDataset

# 测试加载真实渲染数据
dataset = WebpageDataset(data_dir="data/processed", use_demo=False)
print(f"加载了 {len(dataset)} 个样本")

if len(dataset) > 0:
    sample = dataset[0]
    print(f"\n节点数量: {len(sample['node_texts'])}")
    print(f"节点文本: {sample['node_texts'][:3]}...")
    print(f"节点框形状: {sample['node_boxes'].shape}")
    print(f"对齐标签形状: {sample['alignment_labels'].shape}")
else:
    print("没有找到渲染数据")

# 测试 demo 模式
print("\n=== 测试 demo 模式 ===")
demo_dataset = WebpageDataset(use_demo=True)
demo_sample = demo_dataset[0]
print(f"Demo 节点数量: {len(demo_sample['node_texts'])}")
