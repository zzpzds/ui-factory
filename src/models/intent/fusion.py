"""bbox 引导的视觉-页面图融合。"""

from __future__ import annotations

import torch
import torch.nn as nn


class BBoxGuidedFusion(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        visual_dim: int = 768,
        num_heads: int = 8,
        patch_grid_size: int = 14,
    ):
        super().__init__()
        self.patch_grid_size = patch_grid_size
        self.visual_proj = nn.Linear(visual_dim, dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
        )
        self.output_norm = nn.LayerNorm(dim)
        self.node_align_proj = nn.Linear(dim, dim)
        self.patch_align_proj = nn.Linear(dim, dim)

        centers = torch.stack(
            torch.meshgrid(
                (torch.arange(patch_grid_size, dtype=torch.float32) + 0.5)
                / patch_grid_size,
                (torch.arange(patch_grid_size, dtype=torch.float32) + 0.5)
                / patch_grid_size,
                indexing="ij",
            ),
            dim=-1,
        )
        self.register_buffer("patch_centers", centers.reshape(-1, 2), persistent=False)

    def _roi_weights(
        self,
        boxes: torch.Tensor,
        node_mask: torch.Tensor,
        patch_count: int,
    ) -> torch.Tensor:
        if patch_count != self.patch_centers.shape[0]:
            return torch.full(
                (*boxes.shape[:2], patch_count),
                1.0 / patch_count,
                device=boxes.device,
                dtype=boxes.dtype,
            ) * node_mask.unsqueeze(-1)
        patch_y = self.patch_centers[:, 0].to(boxes)
        patch_x = self.patch_centers[:, 1].to(boxes)
        inside = (
            (patch_x.view(1, 1, -1) >= boxes[..., 0:1])
            & (patch_x.view(1, 1, -1) <= boxes[..., 2:3])
            & (patch_y.view(1, 1, -1) >= boxes[..., 1:2])
            & (patch_y.view(1, 1, -1) <= boxes[..., 3:4])
        ).to(boxes.dtype)
        empty = inside.sum(dim=-1, keepdim=True) == 0
        fallback = torch.full_like(inside, 1.0 / patch_count)
        weights = torch.where(empty, fallback, inside)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return weights * node_mask.unsqueeze(-1)

    def forward(
        self,
        node_features: torch.Tensor,
        visual_features: torch.Tensor,
        boxes: torch.Tensor,
        node_mask: torch.Tensor,
        use_visual: bool = True,
    ) -> dict[str, torch.Tensor]:
        visual = self.visual_proj(visual_features)
        if not use_visual:
            visual = torch.zeros_like(visual)
        roi_weights = self._roi_weights(boxes, node_mask, visual.shape[1])
        roi_features = torch.matmul(roi_weights, visual)
        attended, attention = self.cross_attention(
            query=node_features,
            key=visual,
            value=visual,
        )

        if use_visual:
            sources = torch.stack([node_features, roi_features, attended], dim=-2)
            gates = torch.softmax(
                self.gate(
                    torch.cat([node_features, roi_features, attended], dim=-1)
                ),
                dim=-1,
            )
            fused = (sources * gates.unsqueeze(-1)).sum(dim=-2)
        else:
            gates = torch.zeros(
                *node_features.shape[:2], 3, device=node_features.device
            )
            gates[..., 0] = 1.0
            fused = node_features
        fused = self.output_norm(fused + node_features) * node_mask.unsqueeze(-1)

        node_align = self.node_align_proj(node_features)
        patch_align = self.patch_align_proj(visual)
        alignment_logits = torch.matmul(
            node_align, patch_align.transpose(-1, -2)
        ) / (node_align.shape[-1] ** 0.5)
        return {
            "fused": fused,
            "gates": gates,
            "attention": attention,
            "roi_weights": roi_weights,
            "alignment_logits": alignment_logits,
        }
