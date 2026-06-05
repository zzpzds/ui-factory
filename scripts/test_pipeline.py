"""
端到端 pipeline 验证脚本（适配新架构）：
WebpageDataset → FigmaGenerationModel → compute_losses → build_figma_json
"""
import os
import sys

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch

from src.data.dataset import WebpageDataset, webpage_collate_fn
from src.training.trainer import FigmaGenerationModel, compute_losses, build_type_targets
from src.models.decoder import build_figma_json


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 1. 加载 demo 数据（batch=1）
    dataset = WebpageDataset(use_demo=True, max_nodes=32)
    sample = dataset[0]
    batch = webpage_collate_fn([sample])
    print(f"有效节点数: {batch['num_nodes'].tolist()}")
    print(f"node_mask:  {batch['node_mask'].shape}")
    print(f"node_boxes: {batch['node_boxes'].shape}")
    print(f"alignment_labels: {batch['alignment_labels'].shape}")
    print(f"styles keys: {list(batch['styles'].keys())}\n")

    # 2. 模型
    model = FigmaGenerationModel().to(device)
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {n_total/1e6:.1f}M  可训练: {n_trainable/1e6:.1f}M\n")

    model.train()
    target_types = build_type_targets(batch["node_texts"], batch["node_mask"].shape[1])
    outputs = model(batch, return_styles=True, teacher_force=True, target_types=target_types)
    print(f"type_logits:    {outputs['type_logits'].shape}")
    print(f"parent_logits:  {outputs['parent_logits'].shape}")
    print(f"sim_logits:     {outputs['sim_logits'].shape}")
    print(f"styles.reg:     {outputs['styles']['reg'].shape}")
    print(f"styles.color:   {outputs['styles']['color'].shape}\n")

    # 3. 损失
    target_types_dev = target_types.to(device)
    losses = compute_losses(
        outputs,
        target_types=target_types_dev,
        parents=batch["parents"].to(device),
        node_mask=batch["node_mask"].to(device),
        style_targets=batch["styles"],
    )
    print("各项损失：")
    for k, v in losses.items():
        print(f"  {k:14s} = {v.item():.4f}")
    print()

    # 4. 反向传播（验证可训练）
    losses["total"].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.trainable_parameters() if p.grad is not None)
    print(f"反向传播 OK，可训练参数总梯度范数 = {grad_norm:.4f}\n")

    # 5. 推理路径：build_figma_json
    model.eval()
    with torch.no_grad():
        out = model(batch, return_styles=True, teacher_force=False)
        b = 0
        figma_json = build_figma_json(
            type_indices=out["type_indices"][b].cpu(),
            parent_logits=out["parent_logits"][b].cpu(),
            candidate_mask=out["candidate_mask"][b].cpu(),
            node_mask=out["node_mask"][b].cpu(),
            styles={k: v[b].cpu() for k, v in out["styles"].items()},
            node_texts=batch["node_texts"][b],
            node_boxes=batch["node_boxes"][b].cpu(),
        )
    print(f"Figma JSON 根节点数: {len(figma_json)}")
    if figma_json:
        print("第一个根节点示例:")
        print(json.dumps(figma_json[0], indent=2, ensure_ascii=False)[:800])

    print("\n✅ Pipeline 端到端验证通过")


if __name__ == "__main__":
    main()
