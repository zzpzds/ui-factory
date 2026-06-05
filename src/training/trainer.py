"""
训练系统：模型整合 + 损失计算 + 目标构建。

只暴露：
    FigmaGenerationModel   端到端模型类（含 iterative_inference）
    compute_losses         多任务损失（带 mask）
    build_type_targets     从 batch 推断节点类型监督

旧的 trainer.train() 已删除（P9：单一训练入口在 scripts/train.py）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.encoders.visual_encoder import VisualEncoder
from src.models.encoders.code_encoder import CodeEncoder
from src.models.encoders.node_context import NodeContextEncoder, compute_depth_and_sibling
from src.models.alignment.cross_modal_alignment import CrossModalAlignment
from src.models.decoder import FigmaDecoder, NODE_TYPE_TO_IDX, html_tag_to_node_type
from src.models.decoder.figma_decoder import (
    STYLE_REG_DIM, STYLE_COLOR_DIM, STYLE_CLS_DIMS,
)
from src.models.decoder import decode_parents_recursive


# ──────────────────────────── 端到端模型 ────────────────────────────

class FigmaGenerationModel(nn.Module):
    """
    Visual → Code → CrossModal → NodeContext → Decoder。
    """

    def __init__(
        self,
        dim: int = 768,
        align_heads: int = 4,
        ctx_layers: int = 2,
        ctx_heads: int = 4,
        struct_layers: int = 2,
        struct_heads: int = 4,
        freeze_visual: bool = True,
        codebert_unfreeze_last: int = 2,
    ):
        super().__init__()
        self.visual_encoder = VisualEncoder()
        self.code_encoder = CodeEncoder()
        self.alignment = CrossModalAlignment(dim=dim, num_heads=align_heads)
        self.node_context = NodeContextEncoder(
            dim=dim, num_layers=ctx_layers, num_heads=ctx_heads,
        )
        self.decoder = FigmaDecoder(dim=dim)

        # 冻结策略：节省 MPS 内存
        if freeze_visual:
            for p in self.visual_encoder.parameters():
                p.requires_grad = False
        # CodeBERT 仅解冻最后 N 层 + pooler
        self._freeze_codebert_except_last(codebert_unfreeze_last)

    def _freeze_codebert_except_last(self, n_unfreeze: int):
        # CodeBERT (RoBERTa) 12 层 transformer，结构：embeddings + encoder.layer[0..11] + pooler
        encoder = self.code_encoder.model
        # 全部冻
        for p in encoder.parameters():
            p.requires_grad = False
        # 解冻最后 n 层 encoder
        layers = encoder.encoder.layer
        for layer in layers[-n_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad = True
        # pooler
        if hasattr(encoder, "pooler") and encoder.pooler is not None:
            for p in encoder.pooler.parameters():
                p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(
        self,
        batch: dict,
        return_styles: bool = True,
        teacher_force: bool = True,
        target_types: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        batch 来自 webpage_collate_fn，包含 image / node_texts / parents / node_mask /
        alignment_labels / styles / num_nodes 等。
        """
        # 设备
        device = next(self.parameters()).device

        # 1) 视觉
        visual_features = self.visual_encoder(batch["image"])  # [B, P, D]

        # 2) 代码（每个样本节点列表已经 batch 化在外层 list 中）
        code_features_list = []
        max_n = batch["node_mask"].shape[1]
        for sample_texts in batch["node_texts"]:
            feats = self.code_encoder([sample_texts])[0]  # [n, D]
            n = feats.shape[0]
            if n < max_n:
                pad = torch.zeros(max_n - n, feats.shape[-1], device=feats.device, dtype=feats.dtype)
                feats = torch.cat([feats, pad], dim=0)
            elif n > max_n:
                feats = feats[:max_n]
            code_features_list.append(feats)
        code_features = torch.stack(code_features_list, dim=0).to(device)  # [B, max_n, D]

        node_mask = batch["node_mask"].to(device)
        parents = batch["parents"].to(device)
        alignment_labels = batch.get("alignment_labels")
        if alignment_labels is not None:
            alignment_labels = alignment_labels.to(device)

        # 3) 跨模态对齐（融合 + 监督独立）
        align_out = self.alignment(
            visual_features=visual_features,
            code_features=code_features,
            alignment_labels=alignment_labels,
            node_mask=node_mask,
        )

        # 4) NodeContextEncoder（兄弟+深度）
        ctx = self.node_context(align_out["fused"], parents, node_mask)

        # 5) Decoder（训练 teacher forcing：传 GT parents/depth/types）
        if teacher_force:
            depth, _ = compute_depth_and_sibling(parents, node_mask)
            depth = depth.to(device)
            decoder_out = self.decoder(
                ctx=ctx,
                node_mask=node_mask,
                parents=parents,
                depth=depth,
                return_styles=return_styles,
                target_types=target_types.to(device) if target_types is not None else None,
            )
        else:
            decoder_out = self.decoder(
                ctx=ctx,
                node_mask=node_mask,
                parents=None,
                depth=None,
                return_styles=return_styles,
                target_types=None,
            )

        return {
            "visual": visual_features,
            "code": code_features,
            "fused": align_out["fused"],
            "attn": align_out["attn"],
            "sim_logits": align_out["sim_logits"],
            "loss_align_bce": align_out.get("loss_bce"),
            "loss_align_infonce": align_out.get("loss_infonce"),
            "ctx": ctx,
            "node_mask": node_mask,
            **decoder_out,
        }

    @torch.no_grad()
    def iterative_inference(
        self,
        batch: dict,
        n_iters: int = 3,
        init_with_heuristic: bool = False,
    ) -> dict:
        """
        迭代式推理：解决 train/inference 在 depth/sibling 上的 mismatch。

        训练时 NodeContextEncoder 用 GT parents 算 depth/sibling，RecursiveStructDecoder
        的 depth_emb 也用 GT depth → 推理时如果 parents 全部置 -1，模型表现会显著恶化。
        本方法把"上一轮预测的 parents"喂给下一轮：
          iter 0: parents=all_root, depth=0
          iter k: parents=decoded_from_iter_(k-1)_logits, depth 由 parents 推出
        2-3 轮后 parents 通常稳定。

        返回 forward 同样格式的 dict，外加 `predicted_parents [B, N]`。
        """
        device = next(self.parameters()).device
        max_n = batch["node_mask"].shape[1]
        node_mask = batch["node_mask"].to(device)

        # 视觉/代码编码只跑一次（昂贵）
        visual = self.visual_encoder(batch["image"])
        code_list = []
        for sample_texts in batch["node_texts"]:
            f = self.code_encoder([sample_texts])[0]
            n = f.shape[0]
            if n < max_n:
                pad = torch.zeros(max_n - n, f.shape[-1], device=f.device, dtype=f.dtype)
                f = torch.cat([f, pad], dim=0)
            elif n > max_n:
                f = f[:max_n]
            code_list.append(f)
        code = torch.stack(code_list, dim=0).to(device)

        # 对齐通道（只跑一次，不需要监督标签）
        align_out = self.alignment(
            visual_features=visual,
            code_features=code,
            alignment_labels=None,
            node_mask=node_mask,
        )
        fused = align_out["fused"]

        # 初始化：默认所有节点视为 root；可选用启发式 box-containment 作为更好起点
        B = node_mask.shape[0]
        parents = torch.full((B, max_n), -1, dtype=torch.long, device=device)
        if init_with_heuristic:
            from src.models.decoder.figma_decoder import heuristic_parents_from_boxes
            for b in range(B):
                pp = heuristic_parents_from_boxes(
                    batch["node_boxes"][b].cpu(), batch["node_mask"][b].cpu(),
                )
                for i, p in enumerate(pp):
                    parents[b, i] = p

        decoder_out = None
        for it in range(n_iters):
            # 用当前 parents 算 NodeContext
            ctx = self.node_context(fused, parents, node_mask)
            # 用当前 parents 算 depth（喂给 RecursiveStructDecoder 的 depth_emb）
            depth, _ = compute_depth_and_sibling(parents, node_mask)
            decoder_out = self.decoder(
                ctx=ctx,
                node_mask=node_mask,
                parents=None,        # inference 模式（candidate_mask 启用 leaf prior）
                depth=depth.to(device),
                return_styles=True,
                target_types=None,
            )
            # 由 logits 解码新 parents
            new_parents = []
            for b in range(B):
                pp = decode_parents_recursive(
                    decoder_out["parent_logits"][b].cpu(),
                    decoder_out["candidate_mask"][b].cpu(),
                    decoder_out["type_indices"][b].cpu(),
                    node_mask[b].cpu(),
                )
                # padding 位置也填 -1
                row = torch.full((max_n,), -1, dtype=torch.long)
                for i, p in enumerate(pp):
                    row[i] = p
                new_parents.append(row)
            parents = torch.stack(new_parents, dim=0).to(device)

        out = {
            "visual": visual,
            "code": code,
            "fused": fused,
            "attn": align_out["attn"],
            "sim_logits": align_out["sim_logits"],
            "ctx": ctx,
            "node_mask": node_mask,
            "predicted_parents": parents,
            **decoder_out,
        }
        return out


# ──────────────────────────── 目标构建 ────────────────────────────

def build_type_targets(
    node_texts_list: list[list[str]],
    max_nodes: int,
) -> torch.Tensor:
    """
    从 HTML 标签推断节点类型 → [B, max_nodes] long。padding 位置填 0（FRAME），由 mask 屏蔽。
    """
    B = len(node_texts_list)
    types = torch.zeros(B, max_nodes, dtype=torch.long)
    for b, texts in enumerate(node_texts_list):
        for i, text in enumerate(texts[:max_nodes]):
            tag = text.split(" ")[0].strip("<").strip(">")
            type_name = html_tag_to_node_type(tag)
            types[b, i] = NODE_TYPE_TO_IDX.get(type_name, 0)
    return types


# ──────────────────────────── 损失 ────────────────────────────

def compute_losses(
    outputs: dict,
    target_types: torch.Tensor,                 # [B, N]
    parents: torch.Tensor,                      # [B, N]
    node_mask: torch.Tensor,                    # [B, N]
    style_targets: Optional[dict] = None,       # dict from dataset.styles
    weights: Optional[dict] = None,
    enable_styles: bool = True,
) -> dict:
    """
    多任务损失：
      L_type        CE on padded node_mask
      L_struct      已在 RecursiveStructDecoder 内算出
      L_align_bce   独立相似度 BCE
      L_align_nce   InfoNCE 辅助
      L_style_reg   MSE on 5 个回归项（reg_valid mask）
      L_style_color MSE on 3 colors RGBA（color_valid mask）
      L_style_cls   CE on textAlign / display
    """
    w = {
        "type": 1.0,
        "struct": 0.5,
        "align_bce": 0.3,
        "align_nce": 0.05,
        "style_reg": 0.2,
        "style_color": 0.05,
        "style_cls": 0.1,
    }
    if weights:
        w.update(weights)

    losses: dict[str, torch.Tensor] = {}
    device = node_mask.device

    # 1) Type CE（mask 加权）
    type_logits = outputs["type_logits"]  # [B, N, T]
    B, N, T = type_logits.shape
    type_loss = F.cross_entropy(
        type_logits.reshape(-1, T),
        target_types.reshape(-1),
        reduction="none",
    ).reshape(B, N)
    denom = node_mask.sum().clamp(min=1.0)
    losses["type"] = (type_loss * node_mask).sum() / denom

    # 2) Structure（已在 decoder 里算）
    if "loss_struct" in outputs:
        losses["struct"] = outputs["loss_struct"]

    # 3) Alignment
    if outputs.get("loss_align_bce") is not None:
        losses["align_bce"] = outputs["loss_align_bce"]
    if outputs.get("loss_align_infonce") is not None:
        losses["align_nce"] = outputs["loss_align_infonce"]

    # 4) Style（多任务）
    if enable_styles and style_targets is not None and "styles" in outputs:
        pred_styles = outputs["styles"]
        # 4a) 回归项
        if "reg" in pred_styles and "reg" in style_targets:
            pred_reg = pred_styles["reg"]                          # [B, N, 5]
            tgt_reg = style_targets["reg"].to(device)              # [B, N, 5]
            valid = style_targets["reg_valid"].to(device)          # [B, N, 5]
            mask = valid * node_mask.unsqueeze(-1)                 # [B, N, 5]
            mse = F.mse_loss(pred_reg, tgt_reg, reduction="none") * mask
            losses["style_reg"] = mse.sum() / mask.sum().clamp(min=1.0)

        # 4b) 颜色
        if "color" in pred_styles and "color" in style_targets:
            pred_c = pred_styles["color"]                          # [B, N, 3, 4]
            tgt_c = style_targets["color"].to(device)              # [B, N, 3, 4]
            valid = style_targets["color_valid"].to(device)        # [B, N, 3]
            mask = valid.unsqueeze(-1) * node_mask.unsqueeze(-1).unsqueeze(-1)  # [B,N,3,1]
            mse = F.mse_loss(pred_c, tgt_c, reduction="none") * mask
            losses["style_color"] = mse.sum() / mask.sum().clamp(min=1.0)

        # 4c) 分类
        cls_losses = []
        for k in STYLE_CLS_DIMS.keys():
            key = f"cls_{k}"
            if key in pred_styles and key in style_targets:
                logits = pred_styles[key]                          # [B, N, C]
                tgt = style_targets[key].to(device)                # [B, N]
                C = logits.shape[-1]
                ce = F.cross_entropy(
                    logits.reshape(-1, C),
                    tgt.reshape(-1),
                    reduction="none",
                ).reshape(B, N)
                cls_losses.append((ce * node_mask).sum() / denom)
        if cls_losses:
            losses["style_cls"] = sum(cls_losses) / len(cls_losses)

    # 总损失
    total = torch.zeros((), device=device)
    for k, v in losses.items():
        total = total + w.get(k, 1.0) * v
    losses["total"] = total
    return losses
