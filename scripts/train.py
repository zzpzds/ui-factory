"""
端到端训练脚本：编码器 → 对齐模块 → 解码器

用法：
    python scripts/train.py --data_dir data/processed --epochs 10 --batch_size 4
"""
import os
import sys
import json
import argparse
from pathlib import Path

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import WebpageDataset, webpage_collate_fn
from src.models.encoders.visual_encoder import VisualEncoder
from src.models.encoders.code_encoder import CodeEncoder
from src.models.alignment.cross_modal_alignment import CrossModalAlignment
from src.models.decoder import (
    FigmaDecoder,
    build_figma_json,
    html_tag_to_node_type,
    NODE_TYPE_TO_IDX,
)
from src.training.trainer import FigmaGenerationModel, compute_losses, build_targets


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
):
    """训练一个 epoch"""
    model.train()
    total_losses = {}
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # 移动到设备
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # 前向传播
        visual_features = model.visual_encoder(batch["image"])
        code_features = model.code_encoder([batch["node_texts"]])
        fused_features, attn_weights, align_loss = model.alignment(
            visual_features,
            code_features,
            alignment_labels=batch["alignment_labels"].unsqueeze(0),
        )
        decoder_outputs = model.decoder(fused_features, return_styles=True)
        decoder_outputs["align_loss"] = align_loss

        # 构建目标
        target_types, target_pc, _ = build_targets(
            batch["node_texts"],
            batch["parents"].cpu(),
        )

        # 计算损失
        losses = compute_losses(
            decoder_outputs,
            target_types.unsqueeze(0).to(device),
            target_pc.unsqueeze(0).to(device),
            None,
        )

        # 反向传播
        optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 记录
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item()

        pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})

        # 清理 MPS 缓存
        if device.type == "mps":
            torch.mps.empty_cache()

    # 平均损失
    n = len(dataloader)
    return {k: v / n for k, v in total_losses.items()}


@torch.no_grad()
def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_losses = {}

    for batch in dataloader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        visual_features = model.visual_encoder(batch["image"])
        code_features = model.code_encoder([batch["node_texts"]])
        fused_features, attn_weights, align_loss = model.alignment(
            visual_features,
            code_features,
            alignment_labels=batch["alignment_labels"].unsqueeze(0),
        )
        decoder_outputs = model.decoder(fused_features, return_styles=True)
        decoder_outputs["align_loss"] = align_loss

        target_types, target_pc, _ = build_targets(
            batch["node_texts"],
            batch["parents"].cpu(),
        )

        losses = compute_losses(
            decoder_outputs,
            target_types.unsqueeze(0).to(device),
            target_pc.unsqueeze(0).to(device),
            None,
        )

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item()

    n = len(dataloader)
    return {k: v / n for k, v in total_losses.items()}


@torch.no_grad()
def generate_figma_json(model, sample, device):
    """生成 Figma JSON"""
    model.eval()

    visual_features = model.visual_encoder([sample["image"]])
    code_features = model.code_encoder([sample["node_texts"]])
    fused_features, _, _ = model.alignment(visual_features, code_features, None)
    decoder_outputs = model.decoder(fused_features, return_styles=True)

    style_attrs = {k: v[0] for k, v in decoder_outputs["style_attrs"].items()}

    figma_json = build_figma_json(
        decoder_outputs["type_indices"][0],
        decoder_outputs["parent_child_logits"][0],
        style_attrs,
        sample["node_texts"],
        sample["node_boxes"],
    )
    return figma_json


def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据
    if args.demo:
        dataset = WebpageDataset(use_demo=True)
    else:
        dataset = WebpageDataset(data_dir=args.data_dir)

    print(f"数据集大小: {len(dataset)}")

    if len(dataset) == 0:
        print("错误：数据集为空！")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=webpage_collate_fn,
    )

    # 模型
    model = FigmaGenerationModel(device=device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_losses = train_epoch(model, dataloader, optimizer, device, epoch)
        scheduler.step()

        print(f"\n=== Epoch {epoch} 训练结果 ===")
        for k, v in train_losses.items():
            print(f"  {k}: {v:.4f}")

        # 评估
        if epoch % args.eval_every == 0:
            eval_losses = evaluate(model, dataloader, device)
            print(f"\n=== Epoch {epoch} 评估结果 ===")
            for k, v in eval_losses.items():
                print(f"  {k}: {v:.4f}")

            # 保存最佳模型
            if eval_losses.get("total_loss", float("inf")) < best_loss:
                best_loss = eval_losses["total_loss"]
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
                print(f"  ✅ 保存最佳模型 (loss={best_loss:.4f})")

        # 定期保存
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch{epoch}.pt"))

        # 生成示例 Figma JSON
        if epoch % args.generate_every == 0:
            sample = dataset[0]
            figma_json = generate_figma_json(model, sample, device)
            output_path = os.path.join(args.output_dir, f"figma_epoch{epoch}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(figma_json, f, indent=2, ensure_ascii=False)
            print(f"  📦 保存 Figma JSON 到 {output_path}")

    print("\n训练完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--generate_every", type=int, default=1)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    main(args)
