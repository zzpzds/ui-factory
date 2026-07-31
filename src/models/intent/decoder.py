"""Design Intent 多任务解码 heads。"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.intent_dataset import (
    ACTION_TYPES,
    CROSS_ALIGNMENTS,
    ELEMENT_TYPES,
    GROUP_ROLES,
    LAYOUT_MODES,
    PRIMARY_ALIGNMENTS,
    TOKEN_RELATION_KINDS,
)


class PairwiseHead(nn.Module):
    def __init__(self, dim: int, projection_dim: int = 128):
        super().__init__()
        self.left = nn.Linear(dim, projection_dim)
        self.right = nn.Linear(dim, projection_dim)
        self.bias = nn.Parameter(torch.zeros(()))
        self.scale = projection_dim ** 0.5

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        forward = torch.matmul(
            self.left(features), self.right(features).transpose(-1, -2)
        ) / self.scale
        return (forward + forward.transpose(-1, -2)) / 2 + self.bias


class ParentPointerHead(nn.Module):
    def __init__(self, dim: int, projection_dim: int = 128):
        super().__init__()
        self.child = nn.Linear(dim, projection_dim)
        self.parent = nn.Linear(dim, projection_dim)
        self.root = nn.Parameter(torch.randn(projection_dim) * 0.02)
        self.scale = projection_dim ** 0.5

    def forward(
        self,
        features: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        child = self.child(features)
        parent = self.parent(features)
        node_logits = torch.matmul(child, parent.transpose(-1, -2)) / self.scale
        root_logits = torch.matmul(child, self.root) / self.scale
        logits = torch.cat([node_logits, root_logits.unsqueeze(-1)], dim=-1)

        batch_size, node_count = node_mask.shape
        candidate_mask = node_mask.unsqueeze(1).expand(
            batch_size, node_count, node_count
        ).bool()
        candidate_mask = candidate_mask & ~torch.eye(
            node_count, device=node_mask.device, dtype=torch.bool
        ).unsqueeze(0)
        candidate_mask = torch.cat(
            [
                candidate_mask,
                torch.ones(
                    batch_size, node_count, 1,
                    device=node_mask.device, dtype=torch.bool,
                ),
            ],
            dim=-1,
        )
        return logits.masked_fill(~candidate_mask, -1e4), candidate_mask


class IntentDecoder(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.action_head = nn.Linear(dim, len(ACTION_TYPES))
        self.element_head = nn.Linear(dim, len(ELEMENT_TYPES))
        self.role_head = nn.Linear(dim, len(GROUP_ROLES))
        self.layout_mode_head = nn.Linear(dim, len(LAYOUT_MODES))
        self.primary_align_head = nn.Linear(dim, len(PRIMARY_ALIGNMENTS))
        self.cross_align_head = nn.Linear(dim, len(CROSS_ALIGNMENTS))
        self.gap_head = nn.Linear(dim, 1)
        self.padding_head = nn.Linear(dim, 4)
        self.group_affinity_head = PairwiseHead(dim)
        self.token_affinity_heads = nn.ModuleList(
            [PairwiseHead(dim) for _ in TOKEN_RELATION_KINDS]
        )
        self.parent_head = ParentPointerHead(dim)

    def forward(
        self,
        features: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.trunk(features) * node_mask.unsqueeze(-1)
        parent_logits, parent_candidate_mask = self.parent_head(hidden, node_mask)
        return {
            "action_logits": self.action_head(hidden),
            "element_type_logits": self.element_head(hidden),
            "role_logits": self.role_head(hidden),
            "layout_mode_logits": self.layout_mode_head(hidden),
            "primary_align_logits": self.primary_align_head(hidden),
            "cross_align_logits": self.cross_align_head(hidden),
            "gap": torch.sigmoid(self.gap_head(hidden).squeeze(-1)),
            "padding": torch.sigmoid(self.padding_head(hidden)),
            "group_affinity_logits": self.group_affinity_head(hidden),
            "token_affinity_logits": torch.stack(
                [head(hidden) for head in self.token_affinity_heads],
                dim=1,
            ),
            "parent_logits": parent_logits,
            "parent_candidate_mask": parent_candidate_mask,
        }
