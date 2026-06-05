"""
跨模态对齐模块（论文核心模块）。

设计要点：**融合通道与监督通道完全解耦**。
- 融合通道：标准 MultiheadAttention（softmax-normalized），把视觉信息注入 code 表示。
  这一通道不参与对齐损失计算，只产出融合特征。
- 监督通道：独立的 Bilinear 相似度（sigmoid-normalized），与 IoU 标签做 BCE。
  这是对一对多对齐关系的数学正确建模。
- 辅助项：InfoNCE，每个 node 取 IoU 最大的 patch 作为正样本，提供更强的判别信号。

旧实现的 bug：直接用 softmax 后的 attention 权重（行和=1）和多对多 0/1 标签做 BCE，
当一个节点对应多个 patch 时永远学不到——softmax 强制权重和为 1，无法同时给多个
patch 高分。修复见 `BilinearSimilarity` 与 `forward` 中的损失计算。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_overlap_coefficient(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """
    计算两个边界框的 Overlap Coefficient：inter / min(area_a, area_b)。
    处理大小不对称的包含关系（IoU 在此场景下会系统性低估覆盖程度）。
    box_a, box_b: [x1, y1, x2, y2] 格式
    """
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    min_area = min(area_a, area_b)

    if min_area == 0:
        return 0.0
    return float(inter_area / min_area)


def build_alignment_labels(
    node_boxes: torch.Tensor,
    patch_boxes: torch.Tensor,
    threshold: float = 0.3,
) -> torch.Tensor:
    """
    根据节点框和 patch 框计算对齐标签矩阵。
    node_boxes:  [N, 4]
    patch_boxes: [P, 4]
    返回：[N, P]，Overlap Coefficient >= threshold 的位置为 1，否则为 0
    """
    N = node_boxes.shape[0]
    P = patch_boxes.shape[0]
    labels = torch.zeros(N, P)
    for n in range(N):
        for p in range(P):
            score = compute_overlap_coefficient(node_boxes[n], patch_boxes[p])
            if score >= threshold:
                labels[n][p] = 1.0
    return labels


class BilinearSimilarity(nn.Module):
    """
    独立的 [N, P] 相似度矩阵，用于监督多对多对齐关系。
    sim[i, j] = code_i^T · W · visual_j / sqrt(D)，sigmoid 后做 BCE。
    """

    def __init__(self, dim_v: int = 768, dim_c: int = 768, proj_dim: int = 256):
        super().__init__()
        self.code_proj = nn.Linear(dim_c, proj_dim)
        self.visual_proj = nn.Linear(dim_v, proj_dim)
        self.temperature = proj_dim ** 0.5

    def forward(
        self,
        code_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        code_features:   [B, N, D_c]
        visual_features: [B, P, D_v]
        返回：similarity logits [B, N, P]（未经过 sigmoid）
        """
        c = self.code_proj(code_features)  # [B, N, d]
        v = self.visual_proj(visual_features)  # [B, P, d]
        sim = torch.matmul(c, v.transpose(-1, -2)) / self.temperature  # [B, N, P]
        return sim


def alignment_bce_loss(
    sim_logits: torch.Tensor,
    labels: torch.Tensor,
    node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    sim_logits: [B, N, P] 未经过 sigmoid
    labels:     [B, N, P] 0/1
    node_mask:  [B, N] 1=有效，0=padding
    返回标量 loss（已对有效节点平均，正负样本不平衡用 pos_weight 缓解）
    """
    pos = labels.sum().clamp(min=1.0)
    neg = (1.0 - labels).sum().clamp(min=1.0)
    pos_weight = (neg / pos).detach().clamp(max=20.0)
    elementwise = F.binary_cross_entropy_with_logits(
        sim_logits, labels, reduction="none", pos_weight=pos_weight
    )  # [B, N, P]
    if node_mask is not None:
        m = node_mask.unsqueeze(-1).float()  # [B, N, 1]
        elementwise = elementwise * m
        denom = m.sum() * sim_logits.shape[-1]
    else:
        denom = elementwise.numel()
    return elementwise.sum() / denom.clamp(min=1.0)


def alignment_infonce_loss(
    sim_logits: torch.Tensor,
    labels: torch.Tensor,
    node_mask: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    辅助 InfoNCE：每个有效节点取 IoU 最大的 patch 当正样本（标签里随便取一个 1），
    其余 patch 为负样本。仅当节点至少有一个正标签时才计入 loss。

    sim_logits: [B, N, P]
    labels:     [B, N, P]
    node_mask:  [B, N]
    """
    B, N, P = sim_logits.shape
    sim = sim_logits / temperature

    has_pos = labels.sum(dim=-1) > 0  # [B, N]
    if node_mask is not None:
        has_pos = has_pos & node_mask.bool()

    if has_pos.sum() == 0:
        return torch.zeros((), device=sim_logits.device, dtype=sim_logits.dtype)

    # 对每个节点取一个正样本（argmax of label）
    pos_idx = labels.argmax(dim=-1)  # [B, N]
    pos_score = sim.gather(-1, pos_idx.unsqueeze(-1)).squeeze(-1)  # [B, N]

    # logsumexp over all P
    denom = torch.logsumexp(sim, dim=-1)  # [B, N]
    loss = -(pos_score - denom)  # [B, N]
    loss = loss.masked_select(has_pos).mean()
    return loss


class CrossModalAlignment(nn.Module):
    """
    跨模态对齐模块。

    输入：
        visual_features:  [B, P, D]   (P=196 by default)
        code_features:    [B, N, D]
        alignment_labels: [B, N, P]   (None when inferring)
        node_mask:        [B, N]      1=有效 0=padding

    输出：
        fused_features:   [B, N, D]   (融合后的代码表征)
        attn_weights:     [B, N, P]   (softmax 后的融合 attention，可视化用)
        sim_logits:       [B, N, P]   (独立的相似度，监督用，未 sigmoid)
        loss_bce:         scalar 或 None
        loss_infonce:     scalar 或 None
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 4,
        proj_dim: int = 256,
        use_infonce: bool = True,
    ):
        super().__init__()
        # 融合通道：完全独立的 attention，参数不与监督通道共享
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.fuse_norm = nn.LayerNorm(dim)
        # 监督通道：独立 bilinear 相似度
        self.sim = BilinearSimilarity(dim_v=dim, dim_c=dim, proj_dim=proj_dim)
        self.use_infonce = use_infonce

    def forward(
        self,
        visual_features: torch.Tensor,
        code_features: torch.Tensor,
        alignment_labels: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> dict:
        # 融合通道（softmax attention，不参与监督）
        fused, attn_weights = self.cross_attn(
            query=code_features,
            key=visual_features,
            value=visual_features,
        )
        fused = self.fuse_norm(fused + code_features)  # 残差

        # 监督通道（独立相似度，bilinear + sigmoid + BCE）
        sim_logits = self.sim(code_features, visual_features)  # [B, N, P]

        out = {
            "fused": fused,
            "attn": attn_weights,
            "sim_logits": sim_logits,
            "loss_bce": None,
            "loss_infonce": None,
        }

        if alignment_labels is not None:
            out["loss_bce"] = alignment_bce_loss(sim_logits, alignment_labels, node_mask)
            if self.use_infonce:
                out["loss_infonce"] = alignment_infonce_loss(
                    sim_logits, alignment_labels, node_mask
                )

        return out
