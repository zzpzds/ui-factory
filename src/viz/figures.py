"""
论文用可视化（依赖 matplotlib，不引 graphviz/cairo 等系统库）。

提供：
    plot_loss_curves        训练曲线（loss + 关键指标）
    plot_confusion_matrix   节点类型混淆矩阵
    plot_attention_heatmap  对齐 attention 热图
    plot_tree_compare       GT vs Pred 树并排（基于 ASCII / nx-as-mpl）
    plot_struct_depth_hist  结构深度分布
"""
from __future__ import annotations
import os
import json
import math
from typing import Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ──────────────────────────── 1. 训练曲线 ────────────────────────────

def plot_loss_curves(log_path: str, out_path: str, metrics: list[str] | None = None):
    """
    解析 train.log.jsonl，画训练/验证曲线。
    """
    train, val = [], []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("phase") == "train":
                train.append(rec)
            elif rec.get("phase") == "val":
                val.append(rec)

    keys = metrics or ["total", "type", "struct", "align_bce"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3))
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        if train:
            x_t = [r["epoch"] for r in train if k in r]
            y_t = [r[k] for r in train if k in r]
            ax.plot(x_t, y_t, "o-", label="train", linewidth=1.5)
        if val:
            x_v = [r["epoch"] for r in val if k in r]
            y_v = [r[k] for r in val if k in r]
            ax.plot(x_v, y_v, "s--", label="val", linewidth=1.5)
        ax.set_xlabel("epoch"); ax.set_ylabel(k); ax.set_title(k)
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    _ensure_dir(out_path); plt.savefig(out_path, dpi=130); plt.close()


# ──────────────────────────── 2. 混淆矩阵 ────────────────────────────

def plot_confusion_matrix(cm: torch.Tensor, labels: list[str], out_path: str, title: str = "Type CM"):
    cm_np = cm.cpu().numpy().astype(float)
    row_sum = cm_np.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm_np / row_sum

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Pred"); ax.set_ylabel("GT"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            if v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    _ensure_dir(out_path); plt.savefig(out_path, dpi=130); plt.close()


# ──────────────────────────── 3. Attention 热图 ────────────────────────────

def plot_attention_heatmap(
    image: Image.Image,
    attn: torch.Tensor,        # [N, P]，softmax 或 sigmoid 后
    node_idx_list: list[int],
    node_labels: list[str],
    out_path: str,
    grid_size: int = 14,
    target_size: int = 224,
):
    """
    对每个选中节点画一张子图：原图 + patch attention heatmap 叠加。
    """
    fig, axes = plt.subplots(1, len(node_idx_list), figsize=(4 * len(node_idx_list), 4))
    if len(node_idx_list) == 1:
        axes = [axes]

    img_resized = image.convert("RGB").resize((target_size, target_size))
    img_np = np.asarray(img_resized)

    for ax, n_idx, label in zip(axes, node_idx_list, node_labels):
        # patch attn → 14×14 grid → upsample 到 224×224
        a = attn[n_idx].cpu().numpy().reshape(grid_size, grid_size)
        a_full = np.kron(a, np.ones((target_size // grid_size, target_size // grid_size)))
        a_full = a_full / (a_full.max() + 1e-8)
        ax.imshow(img_np)
        ax.imshow(a_full, cmap="hot", alpha=0.45)
        ax.set_title(f"#{n_idx} {label[:40]}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    _ensure_dir(out_path); plt.savefig(out_path, dpi=130); plt.close()


# ──────────────────────────── 4. 树并排（matplotlib） ────────────────────────────

def _layout_tree(parents: list[int]) -> dict[int, tuple[float, float]]:
    """
    简易树排版：用 BFS 计算 depth，按层水平均分 x。
    返回 node_idx → (x, y)。
    """
    N = len(parents)
    children: dict[int, list[int]] = {}
    roots: list[int] = []
    for i, p in enumerate(parents):
        if p < 0 or p >= N:
            roots.append(i)
        else:
            children.setdefault(p, []).append(i)

    depth = {i: 0 for i in roots}
    order = list(roots)
    by_depth: dict[int, list[int]] = {0: list(roots)}
    while order:
        nxt = []
        for n in order:
            for c in children.get(n, []):
                depth[c] = depth[n] + 1
                nxt.append(c)
                by_depth.setdefault(depth[c], []).append(c)
        order = nxt

    pos: dict[int, tuple[float, float]] = {}
    for d, layer in by_depth.items():
        for i, n in enumerate(layer):
            x = (i + 1) / (len(layer) + 1)
            pos[n] = (x, -d)
    return pos


def plot_tree_compare(
    pred_parents: list[int],
    gold_parents: list[int],
    node_labels: list[str],
    out_path: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, parents, title in zip(axes, [gold_parents, pred_parents], ["GT", "Pred"]):
        pos = _layout_tree(parents)
        # 边
        for i, p in enumerate(parents):
            if i in pos and p in pos:
                xs = [pos[p][0], pos[i][0]]
                ys = [pos[p][1], pos[i][1]]
                ax.plot(xs, ys, "-", color="gray", linewidth=0.6, alpha=0.6)
        # 节点
        for i, (x, y) in pos.items():
            label = node_labels[i] if i < len(node_labels) else str(i)
            short = label.split(">")[0][:20]
            color = "#4c8" if (parents[i] == gold_parents[i]) else "#e76"
            ax.scatter([x], [y], s=80, c=color, zorder=3)
            ax.text(x, y - 0.25, short, fontsize=6, ha="center", rotation=20)
        ax.set_title(f"{title} ({sum(1 for p in parents if p < 0)} roots)")
        ax.axis("off")
    plt.tight_layout()
    _ensure_dir(out_path); plt.savefig(out_path, dpi=130); plt.close()


# ──────────────────────────── 5. 深度分布 ────────────────────────────

def _depth_of(parents: list[int]) -> list[int]:
    N = len(parents)
    d = [0] * N
    for _ in range(N):
        changed = False
        for i in range(N):
            p = parents[i]
            if 0 <= p < N:
                new_d = d[p] + 1
                if new_d != d[i]:
                    d[i] = new_d; changed = True
        if not changed:
            break
    return d


def plot_struct_depth_hist(pred_parents_list: list[list[int]], gold_parents_list: list[list[int]], out_path: str):
    pred_d, gold_d = [], []
    for p in pred_parents_list:
        pred_d.extend(_depth_of(p))
    for g in gold_parents_list:
        gold_d.extend(_depth_of(g))
    bins = range(0, max(max(pred_d, default=0), max(gold_d, default=0)) + 2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([gold_d, pred_d], bins=bins, label=["GT", "Pred"], alpha=0.6)
    ax.set_xlabel("depth"); ax.set_ylabel("# nodes"); ax.legend(); ax.set_title("Tree depth distribution")
    plt.tight_layout()
    _ensure_dir(out_path); plt.savefig(out_path, dpi=130); plt.close()
