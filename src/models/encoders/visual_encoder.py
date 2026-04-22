import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
from PIL import Image

VIT_MODEL_NAME = "google/vit-base-patch16-224"


class VisualEncoder(nn.Module):
    """
    ViT 视觉编码器，输入网页截图，输出 196 个 patch 特征序列。
    输出形状：[B, 196, 768]
    """

    def __init__(self, model_name: str = VIT_MODEL_NAME, local_files_only: bool = True):
        super().__init__()
        self.processor = ViTImageProcessor.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = ViTModel.from_pretrained(
            model_name, local_files_only=local_files_only
        )

    def forward(self, images: list[Image.Image]) -> torch.Tensor:
        """
        images: PIL Image 列表，长度为 B
        返回：[B, 196, 768] patch 特征（不含 CLS token）
        """
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        patch_features = outputs.last_hidden_state[:, 1:, :]  # 去掉 CLS token
        return patch_features
