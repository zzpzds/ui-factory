"""双流设计意图恢复模型的核心与端到端包装。"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders.code_encoder import CodeEncoder
from src.models.encoders.visual_encoder import VisualEncoder

from .decoder import IntentDecoder
from .fusion import BBoxGuidedFusion
from .page_graph_encoder import PageGraphEncoder


class DesignIntentRecoveryCore(nn.Module):
    """接收已编码文本和视觉特征，便于单元测试与消融。"""

    def __init__(
        self,
        dim: int = 256,
        text_dim: int = 768,
        visual_dim: int = 768,
        structured_dim: int = 25,
        tag_vocab_size: int = 35,
        use_visual: bool = True,
        use_text: bool = True,
        use_tags: bool = True,
        use_structured: bool = True,
        use_dom_graph: bool = True,
    ):
        super().__init__()
        self.page_encoder = PageGraphEncoder(
            dim=dim,
            text_dim=text_dim,
            structured_dim=structured_dim,
            tag_vocab_size=tag_vocab_size,
        )
        self.fusion = BBoxGuidedFusion(dim=dim, visual_dim=visual_dim)
        self.decoder = IntentDecoder(dim=dim)
        self.use_visual = use_visual
        self.use_text = use_text
        self.use_tags = use_tags
        self.use_structured = use_structured
        self.use_dom_graph = use_dom_graph

    def forward(
        self,
        *,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
        tag_ids: torch.Tensor,
        structured_features: torch.Tensor,
        dom_parents: torch.Tensor,
        boxes: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        page_features = self.page_encoder(
            text_features=text_features,
            tag_ids=tag_ids,
            structured_features=structured_features,
            dom_parents=dom_parents,
            node_mask=node_mask,
            use_text=self.use_text,
            use_tags=self.use_tags,
            use_structured=self.use_structured,
            use_dom_graph=self.use_dom_graph,
        )
        fusion = self.fusion(
            node_features=page_features,
            visual_features=visual_features,
            boxes=boxes,
            node_mask=node_mask,
            use_visual=self.use_visual,
        )
        decoded = self.decoder(fusion["fused"], node_mask)
        return {
            "page_features": page_features,
            **fusion,
            **decoded,
        }


class DesignIntentRecoveryModel(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        freeze_visual: bool = True,
        freeze_text: bool = True,
        code_model_path: str | None = None,
        visual_model_name: str | None = None,
        use_visual: bool = True,
        use_text: bool = True,
        use_tags: bool = True,
        use_structured: bool = True,
        use_dom_graph: bool = True,
    ):
        super().__init__()
        self.visual_encoder = None
        if use_visual:
            self.visual_encoder = (
                VisualEncoder(model_name=visual_model_name)
                if visual_model_name is not None
                else VisualEncoder()
            )
        self.text_encoder = None
        if use_text:
            self.text_encoder = (
                CodeEncoder(model_path=code_model_path)
                if code_model_path is not None
                else CodeEncoder()
            )
        self.core = DesignIntentRecoveryCore(
            dim=dim,
            use_visual=use_visual,
            use_text=use_text,
            use_tags=use_tags,
            use_structured=use_structured,
            use_dom_graph=use_dom_graph,
        )
        self.use_visual = use_visual
        self.use_text = use_text
        self.freeze_visual = freeze_visual
        self.freeze_text = freeze_text
        if freeze_visual and self.visual_encoder is not None:
            for parameter in self.visual_encoder.parameters():
                parameter.requires_grad = False
        if freeze_text and self.text_encoder is not None:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_visual and self.visual_encoder is not None:
            self.visual_encoder.eval()
        if self.freeze_text and self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    def _encode_texts(
        self,
        batch_texts: list[list[str]],
        max_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.text_encoder is None:
            raise RuntimeError("文本编码器未启用。")
        encoded = []
        for node_texts in batch_texts:
            if node_texts:
                features = self.text_encoder([node_texts])[0]
            else:
                features = torch.zeros(
                    0, 768, device=device, dtype=next(self.parameters()).dtype
                )
            if features.shape[0] < max_nodes:
                features = torch.cat(
                    [
                        features,
                        torch.zeros(
                            max_nodes - features.shape[0],
                            features.shape[-1],
                            device=features.device,
                            dtype=features.dtype,
                        ),
                    ],
                    dim=0,
                )
            encoded.append(features[:max_nodes])
        return torch.stack(encoded).to(device)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        node_mask = batch["node_mask"].to(device)
        max_nodes = node_mask.shape[1]
        if self.use_visual:
            assert self.visual_encoder is not None
            visual_features = self.visual_encoder(batch["image"])
        else:
            visual_features = torch.zeros(
                node_mask.shape[0], 196, 768, device=device
            )
        if self.use_text:
            text_features = self._encode_texts(
                batch["node_texts"], max_nodes=max_nodes, device=device
            )
        else:
            text_features = torch.zeros(
                node_mask.shape[0], max_nodes, 768, device=device
            )
        return self.core(
            text_features=text_features,
            visual_features=visual_features,
            tag_ids=batch["tag_ids"].to(device),
            structured_features=batch["structured_features"].to(device),
            dom_parents=batch["dom_parents"].to(device),
            boxes=batch["boxes"].to(device),
            node_mask=node_mask,
        )
