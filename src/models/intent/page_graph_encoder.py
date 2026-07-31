"""页面实现图编码器：文本、标签、结构化特征与 DOM 局部消息融合。"""

from __future__ import annotations

import torch
import torch.nn as nn


class PageGraphEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        text_dim: int = 768,
        structured_dim: int = 25,
        tag_vocab_size: int = 35,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.LayerNorm(dim),
        )
        self.structured_proj = nn.Sequential(
            nn.Linear(structured_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.tag_embedding = nn.Embedding(tag_vocab_size, dim)
        self.parent_message = nn.Linear(dim, dim, bias=False)
        self.child_message = nn.Linear(dim, dim, bias=False)
        self.input_norm = nn.LayerNorm(dim)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(dim)

    @staticmethod
    def _dom_messages(
        x: torch.Tensor,
        parents: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, node_count, dim = x.shape
        valid_parent = (
            (parents >= 0)
            & (parents < node_count)
            & node_mask.bool()
        )
        safe_parent = parents.clamp(min=0, max=max(node_count - 1, 0))
        parent_features = x.gather(
            1, safe_parent.unsqueeze(-1).expand(batch_size, node_count, dim)
        )
        parent_features = parent_features * valid_parent.unsqueeze(-1)

        child_sum = torch.zeros_like(x)
        child_count = torch.zeros(
            batch_size, node_count, 1, device=x.device, dtype=x.dtype
        )
        for batch_index in range(batch_size):
            valid_children = valid_parent[batch_index]
            if not valid_children.any():
                continue
            parent_index = safe_parent[batch_index, valid_children]
            child_values = x[batch_index, valid_children]
            child_sum[batch_index].index_add_(0, parent_index, child_values)
            child_count[batch_index].index_add_(
                0,
                parent_index,
                torch.ones(
                    parent_index.shape[0], 1, device=x.device, dtype=x.dtype
                ),
            )
        child_mean = child_sum / child_count.clamp(min=1.0)
        return parent_features, child_mean

    def forward(
        self,
        text_features: torch.Tensor,
        tag_ids: torch.Tensor,
        structured_features: torch.Tensor,
        dom_parents: torch.Tensor,
        node_mask: torch.Tensor,
        use_text: bool = True,
        use_tags: bool = True,
        use_structured: bool = True,
        use_dom_graph: bool = True,
    ) -> torch.Tensor:
        x = torch.zeros(
            *text_features.shape[:-1],
            self.tag_embedding.embedding_dim,
            device=text_features.device,
            dtype=text_features.dtype,
        )
        if use_text:
            x = x + self.text_proj(text_features)
        if use_tags:
            x = x + self.tag_embedding(tag_ids)
        if use_structured:
            x = x + self.structured_proj(structured_features)
        if use_dom_graph:
            parent_features, child_features = self._dom_messages(
                x, dom_parents, node_mask
            )
            x = (
                x
                + self.parent_message(parent_features)
                + self.child_message(child_features)
            )
        x = self.input_norm(x)
        x = self.transformer(x, src_key_padding_mask=node_mask < 0.5)
        return self.output_norm(x) * node_mask.unsqueeze(-1)
