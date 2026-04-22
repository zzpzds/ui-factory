import torch
import torch.nn as nn


def compute_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """
    计算两个边界框的 IoU（标准交并比）。
    box_a, box_b: [x1, y1, x2, y2] 格式
    """
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_area = 0.0
    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return float(inter_area / union_area)


def build_alignment_labels(
    node_boxes: torch.Tensor,
    patch_boxes: torch.Tensor,
    threshold: float = 0.3,
) -> torch.Tensor:
    """
    根据节点框和 patch 框计算对齐标签矩阵。
    node_boxes:  [N, 4]
    patch_boxes: [P, 4]
    返回：[N, P]，IoU >= threshold 的位置为 1，否则为 0
    """
    N = node_boxes.shape[0]
    P = patch_boxes.shape[0]
    labels = torch.zeros(N, P)
    for n in range(N):
        for p in range(P):
            iou = compute_iou(node_boxes[n], patch_boxes[p])
            if iou >= threshold:
                labels[n][p] = 1.0
    return labels


class CrossModalAlignment(nn.Module):
    """
    跨模态交叉注意力对齐模块。
    输入：视觉特征 [B, 196, 768] + 代码特征 [B, N, 768]
    输出：融合特征 [B, N, 768]，注意力权重 [B, N, 196]，对齐损失（训练时）
    """

    def __init__(self, dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.loss_fn = nn.BCELoss()

    def forward(
        self,
        visual_features: torch.Tensor,
        code_features: torch.Tensor,
        alignment_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        visual_features:   [B, 196, 768]
        code_features:     [B, N,   768]
        alignment_labels:  [B, N,   196]（训练时传入，推理时为 None）

        返回：(fused_features, attn_weights, loss)
        """
        fused_features, attn_weights = self.cross_attn(
            query=code_features,
            key=visual_features,
            value=visual_features,
        )
        loss = None
        if alignment_labels is not None:
            loss = self.loss_fn(attn_weights, alignment_labels)
        return fused_features, attn_weights, loss
