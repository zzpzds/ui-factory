"""
NodeContextEncoder：在融合特征上加入节点间结构感知。

设计动机：CodeBERT 单独编码每个节点的 CLS，节点之间没有交互；交叉注意力虽然
让每个节点看到了视觉，但兄弟/层级关系仍然缺失。本模块用 2 层 self-attention
让节点彼此可见，并显式注入：
- 深度位置嵌入（depth：BFS 距离根的层数）
- 兄弟序号嵌入（sibling_idx：在父的 children 列表里的位置）

这样后续递归结构解码器才能"基于父节点 + 兄弟 + 自身"做有效预测。
"""

import torch
import torch.nn as nn


def compute_depth_and_sibling(parents: torch.Tensor, node_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    parents:   [B, N] long, -1=根, padding 位置 -1
    node_mask: [B, N] float, 1=有效

    返回：
        depth:        [B, N] long  根=0，孤儿/padding=0
        sibling_idx:  [B, N] long  在父亲的孩子列表中的序号；根/孤儿=0；padding=0
    """
    B, N = parents.shape
    depth = torch.zeros(B, N, dtype=torch.long, device=parents.device)
    sibling_idx = torch.zeros(B, N, dtype=torch.long, device=parents.device)

    parents_cpu = parents.cpu().tolist()
    mask_cpu = node_mask.cpu().tolist()

    for b in range(B):
        # depth：从根开始迭代式计算（最多 N 层）
        d = [0] * N
        for _ in range(N):
            changed = False
            for i in range(N):
                if mask_cpu[b][i] < 0.5:
                    continue
                p = parents_cpu[b][i]
                if 0 <= p < N and mask_cpu[b][p] >= 0.5:
                    new_d = d[p] + 1
                    if new_d != d[i]:
                        d[i] = new_d
                        changed = True
            if not changed:
                break
        for i in range(N):
            depth[b, i] = min(d[i], 31)  # 截断到嵌入表上限

        # sibling_idx：扫一遍父→子序号
        counter: dict[int, int] = {}
        for i in range(N):
            if mask_cpu[b][i] < 0.5:
                continue
            p = parents_cpu[b][i]
            key = p if p >= 0 else -1
            sibling_idx[b, i] = min(counter.get(key, 0), 31)
            counter[key] = counter.get(key, 0) + 1

    return depth, sibling_idx


class NodeContextEncoder(nn.Module):
    """
    输入：fused [B, N, D] + parents + node_mask
    输出：ctx   [B, N, D]
    """

    def __init__(self, dim: int = 768, num_layers: int = 2, num_heads: int = 4,
                 max_depth: int = 32, max_siblings: int = 32, dropout: float = 0.1):
        super().__init__()
        self.depth_emb = nn.Embedding(max_depth, dim)
        self.sibling_emb = nn.Embedding(max_siblings, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        fused: torch.Tensor,
        parents: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        fused:     [B, N, D]
        parents:   [B, N]
        node_mask: [B, N] (1=valid, 0=pad)
        """
        depth, sibling_idx = compute_depth_and_sibling(parents, node_mask)
        x = fused + self.depth_emb(depth) + self.sibling_emb(sibling_idx)

        # PyTorch TransformerEncoder 用 src_key_padding_mask（True=mask 掉）
        key_padding = (node_mask < 0.5)  # [B, N] bool
        ctx = self.transformer(x, src_key_padding_mask=key_padding)
        return self.out_norm(ctx)
