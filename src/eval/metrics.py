"""
评测指标模块（论文 Table 用）：
    节点级：type_accuracy / bbox_iou
    结构级：parent_f1 / tree_edit_distance（Zhang-Shasha 简化版）
    样式级：reg_mae / color_delta_e / cls_accuracy
"""
import math
from typing import Sequence

import torch
import torch.nn.functional as F

from src.models.decoder import IDX_TO_NODE_TYPE, decode_parents_recursive


# ──────────────────────────── 节点级 ────────────────────────────

def type_accuracy(
    pred_logits: torch.Tensor,    # [B, N, T]
    target_types: torch.Tensor,   # [B, N]
    node_mask: torch.Tensor,      # [B, N]
) -> float:
    pred = pred_logits.argmax(dim=-1)
    correct = (pred == target_types).float() * node_mask.float()
    return float(correct.sum() / node_mask.sum().clamp(min=1.0))


def confusion_matrix(
    pred_logits: torch.Tensor,
    target_types: torch.Tensor,
    node_mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """返回 [num_classes, num_classes] 混淆矩阵（行=GT, 列=Pred）。"""
    pred = pred_logits.argmax(dim=-1)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for b in range(pred.shape[0]):
        for n in range(pred.shape[1]):
            if node_mask[b, n] < 0.5:
                continue
            t = int(target_types[b, n].item())
            p = int(pred[b, n].item())
            cm[t, p] += 1
    return cm


def bbox_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    u = a + b - inter
    return float(inter / u) if u > 0 else 0.0


# ──────────────────────────── 结构级 ────────────────────────────

def predicted_parents(
    parent_logits: torch.Tensor,   # [N, N+1]
    candidate_mask: torch.Tensor,  # [N, N+1]
    type_indices: torch.Tensor,    # [N]
    node_mask: torch.Tensor,       # [N]
) -> list[int]:
    """对单样本调用 decode_parents_recursive 拿到预测父链。"""
    return decode_parents_recursive(
        parent_logits, candidate_mask, type_indices, node_mask,
    )


def parent_f1(pred: list[int], gold: list[int], node_mask: torch.Tensor) -> dict[str, float]:
    """
    把 (i, parent[i]) 当作有向边集合，计算 P/R/F1。
    -1（root）也算一种边："i → ROOT"。
    """
    valid = [i for i in range(len(pred)) if node_mask[i].item() >= 0.5]
    pred_edges = {(i, pred[i]) for i in valid}
    gold_edges = {(i, gold[i]) for i in valid}
    if not pred_edges and not gold_edges:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pred_edges & gold_edges)
    p = tp / max(1, len(pred_edges))
    r = tp / max(1, len(gold_edges))
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def _build_tree(parents: Sequence[int]) -> dict:
    """从 parents 列表建嵌套字典树（label 用节点索引），返回所有根的列表。"""
    children: dict[int, list[int]] = {}
    roots: list[int] = []
    for i, p in enumerate(parents):
        if p < 0 or p >= len(parents):
            roots.append(i)
        else:
            children.setdefault(p, []).append(i)
    def build(i: int) -> dict:
        return {"id": i, "children": [build(c) for c in children.get(i, [])]}
    return {"id": -1, "children": [build(r) for r in roots]}


def tree_size(t: dict) -> int:
    return 1 + sum(tree_size(c) for c in t["children"])


def tree_edit_distance(pred_parents: list[int], gold_parents: list[int]) -> int:
    """
    简化版 Zhang-Shasha：递归地按子树规模匹配。
    我们用一个粗粒度近似 —— 对节点 ID 完全一致的两棵树，仅看结构差异。
    具体实现：按层级遍历每个节点的子集合，对子集合做 multiset 差异。
    复杂度 O(N^2)；100 节点级别足够。
    """
    pred_tree = _build_tree(pred_parents)
    gold_tree = _build_tree(gold_parents)

    def edit(t1: dict, t2: dict) -> int:
        c1 = sorted(t1["children"], key=lambda x: x["id"])
        c2 = sorted(t2["children"], key=lambda x: x["id"])
        # ID -> 索引
        ids1 = [c["id"] for c in c1]
        ids2 = [c["id"] for c in c2]
        common = set(ids1) & set(ids2)
        delta = (len(ids1) - len(common)) + (len(ids2) - len(common))
        sub = 0
        for cid in common:
            i1 = next(c for c in c1 if c["id"] == cid)
            i2 = next(c for c in c2 if c["id"] == cid)
            sub += edit(i1, i2)
        return delta + sub

    return edit(pred_tree, gold_tree)


# ──────────────────────────── 样式级 ────────────────────────────

def style_reg_mae(
    pred: torch.Tensor,    # [B, N, K]
    target: torch.Tensor,  # [B, N, K]
    valid: torch.Tensor,   # [B, N, K]
    node_mask: torch.Tensor,  # [B, N]
) -> float:
    mask = valid * node_mask.unsqueeze(-1)
    diff = (pred - target).abs() * mask
    return float(diff.sum() / mask.sum().clamp(min=1.0))


def color_delta_e(
    pred: torch.Tensor,    # [B, N, 3, 4] RGBA in [0,1]
    target: torch.Tensor,
    valid: torch.Tensor,   # [B, N, 3]
    node_mask: torch.Tensor,
) -> float:
    """
    用 RGB 欧式距离做 ΔE 近似（CIE76 需要 LAB，工程上 RGB ED 已能反映粗粒度差异）。
    返回在 0-1 颜色空间下的平均欧式距离。
    """
    rgb_pred = pred[..., :3]
    rgb_tgt = target[..., :3]
    de = (rgb_pred - rgb_tgt).pow(2).sum(dim=-1).sqrt()  # [B, N, 3]
    mask = valid * node_mask.unsqueeze(-1)
    return float((de * mask).sum() / mask.sum().clamp(min=1.0))


def style_cls_accuracy(
    pred_logits: torch.Tensor,  # [B, N, C]
    target: torch.Tensor,       # [B, N]
    node_mask: torch.Tensor,    # [B, N]
) -> float:
    pred = pred_logits.argmax(dim=-1)
    correct = (pred == target).float() * node_mask.float()
    return float(correct.sum() / node_mask.sum().clamp(min=1.0))


# ──────────────────────────── 整合：给定一个 batch 的输出，计算所有指标 ────────────────────────────

def compute_full_metrics(
    outputs: dict,
    target_types: torch.Tensor,
    parents: torch.Tensor,
    node_mask: torch.Tensor,
    style_targets: dict | None = None,
) -> dict:
    """供训练脚本/评测脚本汇总调用。返回 scalar dict。"""
    res = {}
    res["type_acc"] = type_accuracy(outputs["type_logits"], target_types, node_mask)

    # 结构 F1：每个样本算一次
    B = node_mask.shape[0]
    f1s, ps, rs, teds = [], [], [], []
    for b in range(B):
        pred_p = predicted_parents(
            outputs["parent_logits"][b].cpu(),
            outputs["candidate_mask"][b].cpu(),
            outputs["type_indices"][b].cpu(),
            node_mask[b].cpu(),
        )
        gold_p = parents[b].cpu().tolist()
        m = node_mask[b].cpu()
        f = parent_f1(pred_p, gold_p, m)
        ps.append(f["precision"]); rs.append(f["recall"]); f1s.append(f["f1"])
        teds.append(tree_edit_distance(pred_p, gold_p))
    res["parent_p"] = sum(ps) / max(1, len(ps))
    res["parent_r"] = sum(rs) / max(1, len(rs))
    res["parent_f1"] = sum(f1s) / max(1, len(f1s))
    res["tree_edit"] = sum(teds) / max(1, len(teds))

    # 样式
    if style_targets and "styles" in outputs:
        ps_out = outputs["styles"]
        if "reg" in ps_out:
            res["style_reg_mae"] = style_reg_mae(
                ps_out["reg"].cpu(),
                style_targets["reg"], style_targets["reg_valid"], node_mask.cpu(),
            )
        if "color" in ps_out:
            res["color_de"] = color_delta_e(
                ps_out["color"].cpu(),
                style_targets["color"], style_targets["color_valid"], node_mask.cpu(),
            )
        for k in ("textAlign", "display"):
            key = f"cls_{k}"
            if key in ps_out and key in style_targets:
                res[f"{key}_acc"] = style_cls_accuracy(
                    ps_out[key].cpu(), style_targets[key], node_mask.cpu(),
                )
    return res
