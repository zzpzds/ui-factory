"""
Figma 三层解码器（论文 spec 完整实现）。

层级：
    L1 节点类型分类  → FigmaNodeTypeClassifier
    L2 内部结构规划  → RecursiveStructDecoder（按深度递归的指针网络）
    L3 样式属性预测  → FigmaStylePredictor（回归 + 分类 + 颜色）

论文对齐：解码器的"递归调用"通过深度位置嵌入 + 推理时按深度逐步展开实现；
训练阶段用 GT 深度做 teacher forcing。
"""

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


# ──────────────────────────── 工具函数（建 Figma JSON 用） ────────────────────────────

def _hex_to_rgba(hex_str: str) -> dict:
    h = hex_str.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return {"r": r, "g": g, "b": b, "a": 1.0}


def _make_node_name(html_text: str, node_type: str) -> str:
    tag_m = re.match(r'<(\w+)', html_text)
    if not tag_m:
        return node_type
    tag = tag_m.group(1)
    id_m = re.search(r'\bid=["\']([^"\']+)["\']', html_text)
    if id_m:
        return f"{tag}#{id_m.group(1)}"
    class_m = re.search(r'\bclass=["\']([^"\']+)["\']', html_text)
    if class_m:
        return f"{tag}.{class_m.group(1).split()[0]}"
    aria_m = re.search(r'\baria-label=["\']([^"\']+)["\']', html_text)
    if aria_m:
        return f"{tag}[{aria_m.group(1)}]"
    return tag


_STYLE_CLAMP = {
    "opacity":       lambda v: max(0.0, min(1.0, v)),
    "fill_opacity":  lambda v: max(0.0, min(1.0, v)),
    "corner_radius": lambda v: max(0.0, v),
    "rotation":      lambda v: v,
    "visible":       lambda v: v > 0.5,
}


# ──────────────────────────── L1：节点类型分类 ────────────────────────────

class FigmaNodeTypeClassifier(nn.Module):
    """输入 ctx [B, N, D]，输出 [B, N, NUM_NODE_TYPES]。"""

    def __init__(self, dim: int = 768, num_types: int = NUM_NODE_TYPES):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, num_types),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.classifier(ctx)


# ──────────────────────────── L2：递归结构解码（指针网络） ────────────────────────────

class RecursiveStructDecoder(nn.Module):
    """
    递归（按深度逐步展开）的父指针解码器。

    训练阶段：用 GT 深度作 teacher forcing，对每个有效节点做 [N+1] 类别 CE
    （N+1 = N 个候选父 + 1 个 root 汇聚位）。candidate mask 屏蔽：
      - 自身
      - padding 位置
      - 叶子类型（按预测的节点类型，叶子不能当父）

    推理阶段：从 root 出发按深度迭代展开。深度 d 时仅暴露已分配在深度 d-1 的节点
    作为候选父；最终落不下来的节点回退为 root。
    """

    # TEXT 在 HTML 中经常含 <span>/<a> 等子节点（<p>, <h1>...），不能当叶子；
    # 仅保留真正不会有子的形状/媒体类型作为叶子约束。
    LEAF_TYPES = {"RECTANGLE", "ELLIPSE", "LINE", "VECTOR"}

    def __init__(
        self,
        dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 4,
        max_depth: int = 16,
        proj_dim: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.proj_dim = proj_dim
        self.depth_emb = nn.Embedding(max_depth, dim)
        # 节点类型嵌入用于条件化（叶/非叶差异、容器层次）
        self.type_emb = nn.Embedding(NUM_NODE_TYPES, dim)

        # 一层小型 transformer 让节点在指针打分前再做一次结构感知
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 2,
            dropout=0.1, batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # 指针打分（双线性）
        self.parent_proj = nn.Linear(dim, proj_dim)
        self.child_proj = nn.Linear(dim, proj_dim)
        # root 汇聚位：单独学一个查询向量
        self.root_query = nn.Parameter(torch.randn(proj_dim) * 0.02)

        # 注册叶子类型布尔向量（推理用）
        leaf_idx = [NODE_TYPE_TO_IDX[t] for t in self.LEAF_TYPES if t in NODE_TYPE_TO_IDX]
        leaf_bool = torch.zeros(NUM_NODE_TYPES, dtype=torch.bool)
        for i in leaf_idx:
            leaf_bool[i] = True
        self.register_buffer("leaf_type_mask", leaf_bool, persistent=False)

    def _score(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, N, D]
        返回 logits [B, N, N+1]，最后一列是 root 汇聚位
        """
        B, N, _ = h.shape
        p = self.parent_proj(h)  # [B, N, d]
        c = self.child_proj(h)   # [B, N, d]
        # logits[b, i, j] = c_i · p_j / sqrt(d)
        scale = self.proj_dim ** 0.5
        logits = torch.matmul(c, p.transpose(-1, -2)) / scale  # [B, N, N]
        # root 汇聚位
        root_logits = torch.matmul(c, self.root_query.unsqueeze(-1)).squeeze(-1) / scale  # [B, N]
        return torch.cat([logits, root_logits.unsqueeze(-1)], dim=-1)  # [B, N, N+1]

    def _candidate_mask(
        self,
        node_mask: torch.Tensor,
        type_indices: torch.Tensor,
        apply_leaf_prior: bool = True,
    ) -> torch.Tensor:
        """
        构造 candidate mask [B, N, N+1]，True=允许当父。
        - 列 j ∈ [0, N): 必须是有效节点 + 不等于行 i（不能自指）
        - 列 N: 永远允许（root 汇聚）
        - apply_leaf_prior=True 时额外屏蔽叶子类型作为父（仅推理用；训练阶段
          关掉，避免标签污染——HTML 中部分被映射为 RECTANGLE 的元素仍可能含子）
        """
        B, N = node_mask.shape
        col_valid = node_mask.unsqueeze(1).expand(B, N, N).bool()  # [B, N, N]
        eye = torch.eye(N, dtype=torch.bool, device=node_mask.device).unsqueeze(0)
        not_self = ~eye
        cand = col_valid & not_self

        if apply_leaf_prior:
            is_leaf = self.leaf_type_mask[type_indices]  # [B, N]
            col_nonleaf = (~is_leaf).unsqueeze(1).expand(B, N, N)
            cand = cand & col_nonleaf

        root_col = torch.ones(B, N, 1, dtype=torch.bool, device=node_mask.device)
        return torch.cat([cand, root_col], dim=-1)  # [B, N, N+1]

    def forward(
        self,
        ctx: torch.Tensor,
        type_indices: torch.Tensor,
        parents: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        target_types: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        训练（parents、depth、node_mask 都给；可选 target_types）：
          ctx + depth_emb + type_emb → transformer → 指针打分

        candidate_mask 用 target_types（如有）构造——避免训练初期 predicted type 错把
        真正的父节点判为叶子导致 target 落到屏蔽列上、CE 爆炸。
        推理时 target_types=None，fallback 到 type_indices（即模型预测）。

        ctx:          [B, N, D]
        type_indices: [B, N] long  （喂给 type_emb 用）
        parents:      [B, N] long  (-1=root, padding=-1)
        node_mask:    [B, N]
        depth:        [B, N] long  (训练 teacher forcing 用；推理 None)
        target_types: [B, N] long  (训练时建议传 GT 类型用于构造 candidate_mask)
        """
        B, N, D = ctx.shape
        if depth is None:
            depth = torch.zeros(B, N, dtype=torch.long, device=ctx.device)

        h = ctx + self.depth_emb(depth) + self.type_emb(type_indices)
        if node_mask is not None:
            h = self.transformer(h, src_key_padding_mask=(node_mask < 0.5))
        else:
            h = self.transformer(h)

        logits = self._score(h)  # [B, N, N+1]

        # 屏蔽非法候选：训练阶段（parents 给定）关闭叶子先验，避免标签污染；
        # 推理阶段开启叶子先验作为结构正则。
        is_training = parents is not None
        types_for_mask = target_types if target_types is not None else type_indices
        cand_mask = self._candidate_mask(
            node_mask if node_mask is not None else torch.ones(B, N, device=ctx.device),
            types_for_mask,
            apply_leaf_prior=not is_training,
        )
        masked_logits = logits.masked_fill(~cand_mask, -1e4)

        out = {
            "parent_logits": masked_logits,   # [B, N, N+1]
            "candidate_mask": cand_mask,
        }

        # 训练 loss
        if parents is not None and node_mask is not None:
            # 把 -1 (root) 映射到 N（最后一列）
            target = parents.clone()
            target[target < 0] = N  # root 汇聚位
            # 越界（截断后悬空）也置为 root
            target[target >= N] = N

            row_mask = node_mask.bool()  # [B, N]
            loss = F.cross_entropy(
                masked_logits.reshape(-1, N + 1),
                target.reshape(-1),
                reduction="none",
            ).reshape(B, N)
            denom = row_mask.float().sum().clamp(min=1.0)
            out["loss_struct"] = (loss * row_mask.float()).sum() / denom

        return out


# ──────────────────────────── L3：样式预测（多任务） ────────────────────────────

# 与 dataset.STYLE_REG_KEYS / STYLE_COLOR_KEYS / STYLE_CLS_KEYS 对齐
STYLE_REG_DIM = 5         # borderRadius, borderWidth, opacity, fontSize, fontWeight
STYLE_COLOR_DIM = 3       # backgroundColor, color, borderColor (each RGBA, 4 channels)
STYLE_CLS_DIMS = {        # 与 dataset.STYLE_CLS_KEYS 对齐
    "textAlign": 6,
    "display": 8,
}


class FigmaStylePredictor(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
        )
        h = dim // 2
        self.reg_head = nn.Linear(h, STYLE_REG_DIM)        # 输出 sigmoid 后是归一化值
        self.color_head = nn.Linear(h, STYLE_COLOR_DIM * 4)  # RGBA × 3
        self.cls_heads = nn.ModuleDict({
            k: nn.Linear(h, n) for k, n in STYLE_CLS_DIMS.items()
        })

    def forward(self, ctx: torch.Tensor) -> dict:
        """
        ctx: [B, N, D]
        返回：
            reg:   [B, N, STYLE_REG_DIM]   sigmoid 归一化
            color: [B, N, 3, 4]            sigmoid (RGBA in [0,1])
            cls_<k>: [B, N, num_choices]   logits
        """
        B, N, _ = ctx.shape
        h = self.trunk(ctx)
        out = {
            "reg": torch.sigmoid(self.reg_head(h)),
            "color": torch.sigmoid(self.color_head(h)).view(B, N, STYLE_COLOR_DIM, 4),
        }
        for k, head in self.cls_heads.items():
            out[f"cls_{k}"] = head(h)
        return out


# ──────────────────────────── 三层解码器整合 ────────────────────────────

class FigmaDecoder(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.type_classifier = FigmaNodeTypeClassifier(dim)
        self.struct_decoder = RecursiveStructDecoder(dim)
        self.style_predictor = FigmaStylePredictor(dim)

    def forward(
        self,
        ctx: torch.Tensor,
        node_mask: torch.Tensor,
        parents: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        return_styles: bool = True,
        target_types: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        ctx:          [B, N, D]
        node_mask:    [B, N]
        parents:      [B, N]   (训练用 teacher forcing；推理 None)
        depth:        [B, N]   (训练用 teacher forcing；推理 None)
        target_types: [B, N]   (训练用 GT type 构造 candidate_mask；推理 None)
        """
        type_logits = self.type_classifier(ctx)  # [B, N, T]
        type_indices = type_logits.argmax(dim=-1)

        # 训练 teacher forcing：type_emb 用 GT type，candidate_mask 也用 GT type
        # （初期 predicted type 错判会让结构 loss 爆炸）
        type_for_emb = target_types if target_types is not None else type_indices

        struct_out = self.struct_decoder(
            ctx,
            type_indices=type_for_emb,
            parents=parents,
            node_mask=node_mask,
            depth=depth,
            target_types=target_types,
        )

        out = {
            "type_logits": type_logits,
            "type_indices": type_indices,
            **struct_out,
        }

        if return_styles:
            out["styles"] = self.style_predictor(ctx)

        return out


# ──────────────────────────── 推理：由 logits 构建 Figma JSON ────────────────────────────

def decode_parents_recursive(
    parent_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    type_indices: torch.Tensor,
    node_mask: torch.Tensor,
    leaf_type_set: set | None = None,
    max_depth: int = 16,
) -> list[int]:
    """
    按深度迭代地解析每个有效节点的父索引。
    parent_logits: [N, N+1]
    candidate_mask:[N, N+1]
    type_indices:  [N]
    node_mask:     [N]

    返回：长度 N 的 list，元素是父节点索引或 -1（根）。padding 位置返回 -1。
    """
    N = parent_logits.shape[0]
    leaf_type_set = leaf_type_set or RecursiveStructDecoder.LEAF_TYPES

    valid = node_mask.bool().tolist()
    types = type_indices.tolist()
    is_leaf = [IDX_TO_NODE_TYPE.get(t, "FRAME") in leaf_type_set for t in types]

    # 屏蔽后的 logits
    masked = parent_logits.masked_fill(~candidate_mask, -1e4)
    # 排序候选：每行 logits argsort（降序）
    order = torch.argsort(masked, dim=-1, descending=True).tolist()

    parents: list[int] = [-2] * N  # -2 = 待定，-1 = root
    depth: list[int] = [-1] * N

    # 第 0 步：选择最自信的"做 root"——把 root 列得分高于任何父的节点先归 root
    for i in range(N):
        if not valid[i]:
            parents[i] = -1
            depth[i] = -1
            continue
        # 如果首选就是 root（列 N），直接定根
        if order[i][0] == N:
            parents[i] = -1
            depth[i] = 0

    # 按深度迭代展开
    for d in range(1, max_depth):
        prev_layer = {i for i in range(N) if depth[i] == d - 1 and not is_leaf[i]}
        if not prev_layer:
            break
        progressed = False
        for i in range(N):
            if not valid[i] or parents[i] != -2:
                continue
            # 在排序列表里找第一个属于 prev_layer 的候选
            for j in order[i]:
                if j == N:
                    # 想 root 但还没在 0 层选上 → 跳过（让深层迭代决定）
                    continue
                if j in prev_layer:
                    parents[i] = j
                    depth[i] = d
                    progressed = True
                    break
        if not progressed:
            break

    # 落不下来的节点：回退为 root
    for i in range(N):
        if valid[i] and parents[i] == -2:
            parents[i] = -1
            depth[i] = 0

    return parents


def heuristic_parents_from_boxes(
    node_boxes: torch.Tensor,  # [N, 4]
    node_mask: torch.Tensor,   # [N]
) -> list[int]:
    """启发式：每节点的父=最小覆盖父（box 包含且面积最小）。"""
    N = node_boxes.shape[0]
    n_valid = int(node_mask.sum().item())
    parents = [-1] * N
    boxes = node_boxes.float()
    for j in range(n_valid):
        best, best_area = -1, float("inf")
        bj = boxes[j]
        for i in range(n_valid):
            if i == j: continue
            bi = boxes[i]
            if bi[0] <= bj[0] and bi[1] <= bj[1] and bi[2] >= bj[2] and bi[3] >= bj[3]:
                area = float((bi[2] - bi[0]) * (bi[3] - bi[1]))
                if area < best_area or (area == best_area and i < best):
                    best_area = area; best = i
        parents[j] = best
    return parents


def build_figma_json(
    type_indices: torch.Tensor,        # [N]
    parent_logits: torch.Tensor,       # [N, N+1]
    candidate_mask: torch.Tensor,      # [N, N+1]
    node_mask: torch.Tensor,           # [N]
    styles: Optional[dict] = None,     # 单样本（无 batch 维）
    node_texts: Optional[list[str]] = None,
    node_boxes: Optional[torch.Tensor] = None,
    structure_mode: str = "model",     # "model" / "heuristic" / "hybrid"
) -> list[dict]:
    """
    把单个样本的解码器输出转 Figma JSON（嵌套树）。
    styles 字典中每个值是 [N, ...]（无 batch 维）。

    structure_mode:
        "model":     用模型预测的 parent_logits 解树（论文主方案）
        "heuristic": 用 box 包含关系（启发式 fallback，强 baseline）
        "hybrid":    模型预测置信度 < 0.3 时回退到启发式
    """
    N = type_indices.shape[0]
    if structure_mode == "heuristic" and node_boxes is not None:
        parents = heuristic_parents_from_boxes(node_boxes, node_mask)
    elif structure_mode == "hybrid" and node_boxes is not None:
        # 拿模型预测的 parents 与启发式 parents
        model_p = decode_parents_recursive(parent_logits, candidate_mask, type_indices, node_mask)
        heur_p = heuristic_parents_from_boxes(node_boxes, node_mask)
        # 用 softmax 概率作为置信度
        probs = torch.softmax(parent_logits, dim=-1)
        max_prob = probs.max(dim=-1).values  # [N]
        parents = [
            (int(model_p[i]) if max_prob[i].item() >= 0.3 else int(heur_p[i]))
            for i in range(N)
        ]
    else:
        parents = decode_parents_recursive(
            parent_logits, candidate_mask, type_indices, node_mask,
        )

    valid_count = int(node_mask.sum().item())
    nodes = []
    for i in range(N):
        if not bool(node_mask[i].item()):
            nodes.append(None)
            continue
        node_type = IDX_TO_NODE_TYPE.get(int(type_indices[i].item()), "FRAME")
        html_text = node_texts[i] if node_texts and i < len(node_texts) else ""
        node = {
            "id": f"node_{i}",
            "name": _make_node_name(html_text, node_type),
            "type": node_type,
            "fills": [],
            "strokes": [],
        }
        # 边界框
        if node_boxes is not None:
            box = node_boxes[i]
            node["absoluteBoundingBox"] = {
                "x": float(box[0]),
                "y": float(box[1]),
                "width": float(box[2] - box[0]),
                "height": float(box[3] - box[1]),
            }
        # 样式（如有）
        if styles is not None:
            _attach_styles(node, styles, i)
        nodes.append(node)

    # 根据 parents 建嵌套树
    return _build_tree_from_parents(nodes, parents)


def _attach_styles(node: dict, styles: dict, i: int) -> None:
    """把模型预测的 style attrs 转成 Figma 字段附加到 node。"""
    from src.data.dataset import STYLE_REG_KEYS, STYLE_REG_NORMS, STYLE_CLS_KEYS, STYLE_COLOR_KEYS

    if "reg" in styles:
        for j, k in enumerate(STYLE_REG_KEYS):
            v = float(styles["reg"][i, j].item())  # 归一化值
            v = v * STYLE_REG_NORMS[k]
            if k == "borderRadius":
                node["cornerRadius"] = max(0.0, v)
            elif k == "borderWidth":
                node["strokeWeight"] = max(0.0, v)
            elif k == "opacity":
                node["opacity"] = max(0.0, min(1.0, v))
            elif k == "fontSize":
                node.setdefault("style", {})["fontSize"] = max(0.0, v)
            elif k == "fontWeight":
                node.setdefault("style", {})["fontWeight"] = max(100.0, min(900.0, v))

    if "color" in styles:
        # color: [N, 3, 4]，对应 backgroundColor/color/borderColor
        bg = styles["color"][i, 0].tolist()
        fg = styles["color"][i, 1].tolist()
        bd = styles["color"][i, 2].tolist()
        if bg[3] > 0.05:
            node["fills"] = [{"type": "SOLID", "color": {"r": bg[0], "g": bg[1], "b": bg[2], "a": bg[3]}}]
        if bd[3] > 0.05:
            node["strokes"] = [{"type": "SOLID", "color": {"r": bd[0], "g": bd[1], "b": bd[2], "a": bd[3]}}]
        node.setdefault("style", {})["fontColor"] = {"r": fg[0], "g": fg[1], "b": fg[2], "a": fg[3]}

    for k, choices in STYLE_CLS_KEYS.items():
        key = f"cls_{k}"
        if key in styles:
            idx = int(styles[key][i].argmax(dim=-1).item())
            node.setdefault("style", {})[k] = choices[min(idx, len(choices) - 1)]


def _build_tree_from_parents(nodes: list[Optional[dict]], parents: list[int]) -> list[dict]:
    """nodes 长度 N（包含 None=padding）；parents 同长度。返回根列表（带 children）。"""
    N = len(nodes)
    # 深拷贝（避免修改 caller 的对象）
    copies = [copy.deepcopy(n) if n is not None else None for n in nodes]
    for c in copies:
        if c is not None:
            c.pop("children", None)
    for i, p in enumerate(parents):
        if copies[i] is None:
            continue
        if p >= 0 and p < N and copies[p] is not None:
            copies[p].setdefault("children", []).append(copies[i])
    return [copies[i] for i in range(N) if copies[i] is not None and (parents[i] < 0 or parents[i] >= N or copies[parents[i]] is None)]
