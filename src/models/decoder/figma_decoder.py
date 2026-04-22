import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import re
import copy

from .figma_types import (
    DESIGN_NODE_TYPES,
    NODE_TYPE_TO_IDX,
    IDX_TO_NODE_TYPE,
    NUM_NODE_TYPES,
    LayoutMode,
    html_tag_to_node_type,
)


def _hex_to_rgba(hex_str: str) -> dict:
    """将 #RGB 或 #RRGGBB 转为 Figma color dict。"""
    h = hex_str.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return {"r": r, "g": g, "b": b, "a": 1.0}


def _extract_color_from_html(html_text: str) -> list[dict]:
    """从 HTML 内联样式提取颜色，返回 Figma fills 格式。解析失败返回 []。"""
    hex_pattern = r'(?:background(?:-color)?|color)\s*:\s*(#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3}))\b'
    rgb_pattern = r'(?:background(?:-color)?|color)\s*:\s*rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)'

    m = re.search(hex_pattern, html_text)
    if m:
        try:
            color = _hex_to_rgba(m.group(1))
            return [{"type": "SOLID", "color": color}]
        except (ValueError, IndexError):
            pass

    m = re.search(rgb_pattern, html_text)
    if m:
        try:
            color = {
                "r": int(m.group(1)) / 255.0,
                "g": int(m.group(2)) / 255.0,
                "b": int(m.group(3)) / 255.0,
                "a": 1.0,
            }
            return [{"type": "SOLID", "color": color}]
        except (ValueError, IndexError):
            pass

    return []


def _make_node_name(html_text: str, node_type: str) -> str:
    """从 HTML 文本提取语义化节点名。"""
    tag_m = re.match(r'<(\w+)', html_text)
    if not tag_m:
        return node_type
    tag = tag_m.group(1)

    id_m = re.search(r'\bid=["\']([^"\']+)["\']', html_text)
    if id_m:
        return f"{tag}#{id_m.group(1)}"

    class_m = re.search(r'\bclass=["\']([^"\']+)["\']', html_text)
    if class_m:
        first_class = class_m.group(1).split()[0]
        return f"{tag}.{first_class}"

    aria_m = re.search(r'\baria-label=["\']([^"\']+)["\']', html_text)
    if aria_m:
        return f"{tag}[{aria_m.group(1)}]"

    return tag


def _build_tree(nodes: list[dict], node_boxes: torch.Tensor) -> list[dict]:
    """
    用边界框包含关系将平铺节点列表重建为嵌套树。
    node_boxes: [N, 4] float，格式 (x1, y1, x2, y2)
    返回根节点列表（每个根节点的 children 字段为嵌套子节点对象）。
    """
    N = len(nodes)
    if N == 0:
        return []

    boxes = node_boxes.float()
    # parent[j] = i 表示 i 是 j 的最近父节点，-1 表示根节点
    parent = [-1] * N

    for j in range(N):
        best_parent = -1
        best_area = float("inf")
        bj = boxes[j]
        for i in range(N):
            if i == j:
                continue
            bi = boxes[i]
            # 检查 i 是否包含 j
            if bi[0] <= bj[0] and bi[1] <= bj[1] and bi[2] >= bj[2] and bi[3] >= bj[3]:
                area = float((bi[2] - bi[0]) * (bi[3] - bi[1]))
                if area < best_area:
                    best_area = area
                    best_parent = i
        parent[j] = best_parent

    # 深拷贝节点（避免修改原始 list）
    node_copies = [copy.copy(n) for n in nodes]
    # 移除旧 children 字段
    for n in node_copies:
        n.pop("children", None)

    # 构建树
    for j, p in enumerate(parent):
        if p >= 0:
            if "children" not in node_copies[p]:
                node_copies[p]["children"] = []
            node_copies[p]["children"].append(node_copies[j])

    # 只返回根节点
    return [node_copies[i] for i in range(N) if parent[i] == -1]


class FigmaStylePredictor(nn.Module):
    """
    第三层解码器：样式属性预测器。
    输入：融合特征 [B, N, D]
    输出：样式属性字典 {property_name: [B, N]}（回归或分类）
    """

    def __init__(self, dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
        )
        # 常见样式属性（简化版）
        self.style_heads = nn.ModuleDict({
            "fill_opacity": nn.Linear(dim // 2, 1),  # 0~1
            "corner_radius": nn.Linear(dim // 2, 1),  # >= 0
            "opacity": nn.Linear(dim // 2, 1),  # 0~1
            "rotation": nn.Linear(dim // 2, 1),  # -180~180
            # 布尔属性
            "visible": nn.Linear(dim // 2, 1),  # 0/1
        })

    def forward(self, fused_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        fused_features: [B, N, D]
        返回: 样式属性字典，所有值已约束到合法范围
        """
        h = self.net(fused_features)  # [B, N, D//2]
        outputs = {}
        for name, head in self.style_heads.items():
            raw = head(h).squeeze(-1)  # [B, N]
            if name in ("opacity", "fill_opacity", "visible"):
                outputs[name] = torch.sigmoid(raw)
            elif name == "corner_radius":
                outputs[name] = F.softplus(raw)
            elif name == "rotation":
                outputs[name] = torch.tanh(raw) * 180.0
            else:
                outputs[name] = raw
        return outputs


class FigmaNodeTypeClassifier(nn.Module):
    """
    第一层解码器：节点类型分类器。
    输入：融合特征 [B, N, D]
    输出：节点类型概率分布 [B, N, NUM_NODE_TYPES]
    """

    def __init__(self, dim: int = 768, num_types: int = NUM_NODE_TYPES):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, num_types),
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        fused_features: [B, N, D]
        返回: 类型logits [B, N, NUM_NODE_TYPES]
        """
        return self.classifier(fused_features)


class FigmaStructurePlanner(nn.Module):
    """
    第二层解码器：内部结构规划器（子节点序列生成）。
    输入：融合特征 [B, N, D] + 节点类型 [B, N]
    输出：父子关系矩阵 [B, N, N]，预测节点 i 是否有子节点 j
    """

    # 叶子节点类型标记（不能有子节点）
    LEAF_TYPES = {"TEXT", "RECTANGLE", "ELLIPSE", "LINE", "VECTOR"}

    def __init__(self, dim: int = 768, num_types: int = NUM_NODE_TYPES):
        super().__init__()
        # 节点类型嵌入
        self.type_embedding = nn.Embedding(num_types, dim)
        # 父子关系评分网络
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(
        self,
        fused_features: torch.Tensor,
        node_type_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        fused_features:   [B, N, D]
        node_type_indices: [B, N] int64
        返回: 父子关系 logits [B, N, N]
        """
        B, N, D = fused_features.shape

        # 获取节点类型嵌入 [B, N, D]
        type_emb = self.type_embedding(node_type_indices)

        # 拼接特征和类型嵌入
        combined = torch.cat([fused_features, type_emb], dim=-1)  # [B, N, D*2]

        # 计算父子关系分数
        # parent_features: [B, N, 1, D*2]
        # child_features:  [B, 1, N, D*2]
        parent_features = combined.unsqueeze(2)
        child_features = combined.unsqueeze(1)
        scores = self.net(parent_features + child_features)  # [B, N, N, 1]
        scores = scores.squeeze(-1)  # [B, N, N]

        # 叶子节点不能作为父节点，将这些位置的分数设为一个很大的负值
        leaf_mask = torch.zeros(B, N, 1, device=fused_features.device)
        for type_name, type_id in NODE_TYPE_TO_IDX.items():
            if type_name in self.LEAF_TYPES:
                leaf_mask = leaf_mask + (node_type_indices == type_id).float().unsqueeze(-1)

        # 应用 mask：叶子节点的输出行设为 -inf
        scores = scores.masked_fill(leaf_mask.bool(), float("-inf"))

        return scores


class FigmaDecoder(nn.Module):
    """
    完整的三层 Figma JSON 解码器。

    输入：融合特征 [B, N, D]（来自跨模态对齐模块）
    输出：
        - node_types:      节点类型 logits [B, N, NUM_NODE_TYPES]
        - parent_child:    父子关系矩阵 [B, N, N]
        - style_attrs:     样式属性字典
    """

    def __init__(self, dim: int = 768):
        super().__init__()
        self.type_classifier = FigmaNodeTypeClassifier(dim)
        self.structure_planner = FigmaStructurePlanner(dim)
        self.style_predictor = FigmaStylePredictor(dim)

    def forward(
        self,
        fused_features: torch.Tensor,
        return_styles: bool = True,
    ) -> dict:
        """
        fused_features: [B, N, D]
        """
        # 第一层：节点类型分类
        type_logits = self.type_classifier(fused_features)  # [B, N, T]
        type_indices = type_logits.argmax(dim=-1)  # [B, N]

        # 第二层：结构规划
        parent_child_logits = self.structure_planner(
            fused_features, type_indices
        )  # [B, N, N]

        outputs = {
            "type_logits": type_logits,
            "type_indices": type_indices,
            "parent_child_logits": parent_child_logits,
        }

        # 第三层：样式预测
        if return_styles:
            style_attrs = self.style_predictor(fused_features)
            outputs["style_attrs"] = style_attrs

        return outputs


def build_figma_json(
    type_indices: torch.Tensor,
    parent_child_logits: torch.Tensor,
    style_attrs: Optional[dict[str, torch.Tensor]] = None,
    node_texts: Optional[list[list[str]]] = None,
    node_boxes: Optional[torch.Tensor] = None,
) -> list[dict]:
    """
    将解码器输出转换为 Figma JSON 格式。

    type_indices:       [N] int64，节点类型索引
    parent_child_logits: [N, N] float，子节点关系
    style_attrs:        样式属性字典
    node_texts:         [N] str，节点文本（可选）
    node_boxes:         [N, 4] float，边界框（可选）

    返回：Figma JSON 格式的节点列表
    """
    N = type_indices.shape[0]
    nodes = []

    for i in range(N):
        type_id = type_indices[i].item()
        node_type = IDX_TO_NODE_TYPE.get(type_id, "FRAME")

        node = {
            "id": f"node_{i}",
            "name": node_texts[i][:30] if node_texts and i < len(node_texts) else node_type,
            "type": node_type,
            "fills": [],
            "strokes": [],
        }

        # 添加位置和尺寸
        if node_boxes is not None and i < node_boxes.shape[0]:
            box = node_boxes[i]
            node["absoluteBoundingBox"] = {
                "x": float(box[0]),
                "y": float(box[1]),
                "width": float(box[2] - box[0]),
                "height": float(box[3] - box[1]),
            }

        # 添加样式属性
        if style_attrs is not None:
            for attr_name, attr_values in style_attrs.items():
                if i < attr_values.shape[0]:
                    node[attr_name] = float(attr_values[i].item())

        # 解析子节点
        children = []
        for j in range(N):
            if i != j and parent_child_logits[i, j] > 0:
                children.append(f"node_{j}")
        if children:
            node["children"] = children

        nodes.append(node)

    return nodes
