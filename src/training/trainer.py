import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import json

from src.models.encoders.visual_encoder import VisualEncoder
from src.models.encoders.code_encoder import CodeEncoder
from src.models.alignment.cross_modal_alignment import CrossModalAlignment
from src.models.decoder import FigmaDecoder, html_tag_to_node_type, NODE_TYPE_TO_IDX
from src.data.dataset import WebpageDataset


class FigmaGenerationModel(nn.Module):
    """
    完整的网页截图 → Figma JSON 生成模型。
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.visual_encoder = VisualEncoder().to(device)
        self.code_encoder = CodeEncoder().to(device)
        self.alignment = CrossModalAlignment().to(device)
        self.decoder = FigmaDecoder().to(device)

    def forward(self, batch, use_styles: bool = True):
        """
        batch: WebpageDataset 返回的字典
        """
        # 1. 视觉编码
        visual_features = self.visual_encoder(batch["image"])

        # 2. 代码编码
        code_features = self.code_encoder([batch["node_texts"]])

        # 3. 跨模态对齐
        fused_features, attn_weights, align_loss = self.alignment(
            visual_features,
            code_features,
            alignment_labels=batch["alignment_labels"].unsqueeze(0),
        )

        # 4. Figma 解码
        decoder_outputs = self.decoder(
            fused_features,
            return_styles=use_styles,
        )
        decoder_outputs["align_loss"] = align_loss

        return decoder_outputs


def compute_losses(
    decoder_outputs: dict,
    target_node_types: torch.Tensor,
    target_parent_child: torch.Tensor,
    target_styles: Optional[dict[str, torch.Tensor]] = None,
    lambda_align: float = 0.1,
    lambda_structure: float = 0.3,
) -> dict[str, torch.Tensor]:
    """
    计算总损失。

    decoder_outputs: 模型输出的字典
    target_node_types: [B, N] 目标节点类型
    target_parent_child: [B, N, N] 目标父子关系
    target_styles: 目标样式属性字典
    """
    losses = {}

    # 对齐损失
    align_loss = decoder_outputs["align_loss"]
    losses["align_loss"] = align_loss

    # 节点类型分类损失
    type_logits = decoder_outputs["type_logits"]
    B, N, T = type_logits.shape
    type_loss = nn.CrossEntropyLoss()(
        type_logits.view(B * N, T),
        target_node_types.view(B * N),
    )
    losses["type_loss"] = type_loss

    # 结构规划损失（父子关系）
    pc_logits = decoder_outputs["parent_child_logits"]
    pc_loss = nn.BCEWithLogitsLoss()(
        pc_logits.view(B * N * N),
        target_parent_child.view(B * N * N),
    )
    losses["structure_loss"] = pc_loss

    # 总损失
    total_loss = (
        lambda_align * align_loss
        + type_loss
        + lambda_structure * pc_loss
    )

    # 样式损失
    if target_styles is not None and "style_attrs" in decoder_outputs:
        style_losses = []
        pred_styles = decoder_outputs["style_attrs"]
        for name, pred in pred_styles.items():
            if name in target_styles:
                target = target_styles[name].to(pred.device)
                style_losses.append(nn.MSELoss()(pred, target))
        if style_losses:
            style_loss = sum(style_losses) / len(style_losses)
            losses["style_loss"] = style_loss
            total_loss = total_loss + 0.2 * style_loss

    losses["total_loss"] = total_loss
    return losses


def train_step(
    model: FigmaGenerationModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    target_node_types: torch.Tensor,
    target_parent_child: torch.Tensor,
    target_styles: Optional[dict[str, torch.Tensor]] = None,
) -> dict[str, float]:
    """
    单步训练。
    """
    model.train()
    optimizer.zero_grad()

    outputs = model(batch)

    losses = compute_losses(
        outputs,
        target_node_types.unsqueeze(0),  # [1, N]
        target_parent_child.unsqueeze(0),  # [1, N, N]
        target_styles,
    )

    losses["total_loss"].backward()
    optimizer.step()

    return {k: v.item() for k, v in losses.items()}


def build_targets(
    node_texts: list[str],
    parents: torch.Tensor,
    style_attrs: Optional[dict[str, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, torch.Tensor]]]:
    """
    从节点信息构建训练目标。

    node_texts: 节点文本列表
    parents: 真实 DOM 父索引 [N]，-1 表示根节点（来自 render_pages.py 输出的 parents.pt）
    返回：(target_types, target_parent_child, target_styles)
    """
    # 节点类型：从 HTML 标签推断
    target_types = []
    for text in node_texts:
        tag = text.split(" ")[0].strip("<").strip(">")
        type_name = html_tag_to_node_type(tag)
        type_idx = NODE_TYPE_TO_IDX.get(type_name, 0)
        target_types.append(type_idx)
    target_types = torch.tensor(target_types, dtype=torch.long)

    # 父子关系：使用真实 DOM 父索引，target_parent_child[i, j] = 1 表示 i 是 j 的父节点
    N = len(node_texts)
    target_parent_child = torch.zeros(N, N)
    for j, p in enumerate(parents.tolist()):
        if p >= 0:
            target_parent_child[p, j] = 1.0

    return target_types, target_parent_child, style_attrs


def train(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "outputs",
    use_demo: bool = False,
):
    """
    训练主函数。
    """
    # 数据
    dataset = WebpageDataset(data_dir=data_dir if not use_demo else None, use_demo=use_demo)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型
    model = FigmaGenerationModel(device=device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 训练循环
    for epoch in range(epochs):
        epoch_losses = {}
        for step, batch in enumerate(dataloader):
            # 移动到设备
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # 构建目标
            node_texts = batch["node_texts"]
            target_types, target_pc, _ = build_targets(node_texts, batch["parents"].cpu())

            # 训练
            losses = train_step(
                model, batch, optimizer,
                target_types.to(device),
                target_pc.to(device),
            )

            # 记录
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v

            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} Step {step}: {losses}")

        # Epoch 总结
        avg_losses = {k: v / len(dataloader) for k, v in epoch_losses.items()}
        print(f"\n=== Epoch {epoch+1} Summary ===")
        for k, v in avg_losses.items():
            print(f"  {k}: {v:.4f}")

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch{epoch+1}.pt"))

    print("训练完成！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_demo=args.demo,
    )
