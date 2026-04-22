import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import ViTModel, ViTImageProcessor
import numpy as np

single_image_path = "./data/web_page_image.png"
image = Image.open(single_image_path).convert('RGB')
print(f'已加载图像：{single_image_path}')
print(f'图像尺寸：{image.size}')
print(f'图像模式：{image.mode}')

# plt.figure(figsize=(10, 6))
# plt.imshow(image)
# plt.title('网页截图')
# plt.axis('off')
# plt.show()

model_name = 'google/vit-base-patch16-224'

processor = ViTImageProcessor.from_pretrained(model_name, local_files_only=True)
print('图像处理器加载成功')
print(f'归一化均值：{processor.image_mean}')
print(f'归一化标准差：{processor.image_std}')

device = torch.device('mps')
model = ViTModel.from_pretrained(model_name, local_files_only=True)
model = model.to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f'模型总参数数量：{total_params / 1e6:.1f}M')
print(f'模型已加载到：{device}')

inputs = processor(images=image, return_tensors='pt')
print(f'处理后的输入形状：{inputs["pixel_values"].shape}')
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
features = outputs.last_hidden_state
print(f'特征序列形状：{features.shape}')

cls_features = features[:, 0, :]
print(f'CLS特征形状：{cls_features.shape}')
patch_features = features[:, 1:, :]
print(f'局部Patch特征形状：{patch_features.shape}')

patch_features_np = patch_features.squeeze(0).cpu().numpy()
patch_norms = np.linalg.norm(patch_features_np, axis=1)
patch_map = patch_norms.reshape((14, 14))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(image)
axes[0].set_title('原始图像')
axes[0].axis('off')

im = axes[1].imshow(patch_map, cmap='hot', interpolation='nearest')
axes[1].set_title('ViT Patch 特征激活强度(14 X 14)')
axes[1].set_xlabel('Patch 水平索引')
axes[1].set_ylabel('Patch 垂直索引')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.show()

print(f'激活强度范围: {patch_norms.min():.2f} ~ {patch_norms.max():.2f}')