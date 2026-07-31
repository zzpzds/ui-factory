import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

CODEBERT_MODEL_PATH = os.environ.get(
    "CODEBERT_PATH",
    "/Users/didi/.cache/modelscope/microsoft/codebert-base",
)


class CodeEncoder(nn.Module):
    """
    CodeBERT 代码编码器，输入 DOM 节点文本列表，输出节点特征序列。
    输出形状：[B, N, 768]，其中 N 为节点数
    """

    def __init__(
        self,
        model_path: str = CODEBERT_MODEL_PATH,
        max_length: int = 128,
        local_files_only: bool = True,
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            model_path, local_files_only=local_files_only
        )

    def forward(self, batch_node_texts: list[list[str]]) -> torch.Tensor:
        """
        batch_node_texts: 外层 list 长度为 B，内层 list 为每个样本的节点文本列表
        返回：[B, N, 768]，N 为节点数（同一 batch 内节点数需相同，或在 DataLoader 中处理对齐）
        """
        device = next(self.parameters()).device
        batch_features = []
        for node_texts in batch_node_texts:
            inputs = self.tokenizer(
                node_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # 取每个节点的 CLS token 特征 [N, 768]
            cls_features = outputs.last_hidden_state[:, 0, :]
            batch_features.append(cls_features)
        # stack 成 [B, N, 768]
        return torch.stack(batch_features, dim=0)
