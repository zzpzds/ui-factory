"""
端到端 pipeline 验证脚本：
数据加载 → 视觉编码 → 代码编码 → 跨模态对齐 → Figma 解码 → 打印形状和损失
"""
import os
import sys

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# 确保从项目根目录运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.data.dataset import WebpageDataset
from src.models.encoders.visual_encoder import VisualEncoder
from src.models.encoders.code_encoder import CodeEncoder
from src.models.alignment.cross_modal_alignment import CrossModalAlignment
from src.models.decoder import FigmaDecoder, build_figma_json


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 1. 加载 demo 数据
    dataset = WebpageDataset(use_demo=True)
    sample = dataset[0]
    print(f"节点数量: {len(sample['node_texts'])}")
    print(f"对齐标签形状: {sample['alignment_labels'].shape}")
    print(f"对齐的 patch 数（各节点）: {sample['alignment_labels'].sum(dim=1).int().tolist()}\n")

    # 2. 视觉编码
    visual_encoder = VisualEncoder().to(device)
    visual_encoder.eval()
    with torch.no_grad():
        visual_features = visual_encoder([sample["image"]])
    print(f"视觉特征: {visual_features.shape}")  # [1, 196, 768]

    # 3. 代码编码
    code_encoder = CodeEncoder().to(device)
    code_encoder.eval()
    with torch.no_grad():
        code_features = code_encoder([sample["node_texts"]])
    print(f"代码特征: {code_features.shape}")  # [1, N, 768]

    # 4. 跨模态对齐
    alignment_model = CrossModalAlignment().to(device)
    alignment_model.eval()
    labels = sample["alignment_labels"].unsqueeze(0).to(device)  # [1, N, 196]
    with torch.no_grad():
        fused, attn_weights, loss = alignment_model(visual_features, code_features, labels)
    print(f"融合特征:   {fused.shape}")        # [1, N, 768]
    print(f"注意力权重: {attn_weights.shape}")  # [1, N, 196]
    print(f"对齐损失:   {loss.item():.4f}\n")

    # 5. Figma 解码
    decoder = FigmaDecoder().to(device)
    decoder.eval()
    with torch.no_grad():
        decoder_outputs = decoder(fused, return_styles=True)

    print(f"节点类型 logits:   {decoder_outputs['type_logits'].shape}")
    print(f"节点类型预测:      {decoder_outputs['type_indices'].tolist()}")
    print(f"父子关系矩阵:      {decoder_outputs['parent_child_logits'].shape}")
    print(f"样式属性:          {list(decoder_outputs['style_attrs'].keys())}\n")

    # 6. 构建 Figma JSON
    style_attrs_batch0 = {k: v[0] for k, v in decoder_outputs["style_attrs"].items()}
    figma_json = build_figma_json(
        decoder_outputs["type_indices"][0],
        decoder_outputs["parent_child_logits"][0],
        style_attrs_batch0,
        sample["node_texts"],
        sample["node_boxes"],
    )
    print(f"Figma JSON 节点数: {len(figma_json)}")
    print("第一个节点示例:")
    import json
    print(json.dumps(figma_json[0], indent=2, ensure_ascii=False))

    print("\n✅ Pipeline 端到端验证通过")


if __name__ == "__main__":
    main()
