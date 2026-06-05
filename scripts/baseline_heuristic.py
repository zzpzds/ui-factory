"""
B1：启发式基线（对应 figma-html / Russo 等纯规则方法）。

只依赖 HTML 标签 + 边界框包含关系，不使用任何视觉模型——这是论文里要对比
"为什么需要视觉模态"的对照组。

流程：
1. 节点类型：直接 HTML tag → Figma type 映射（与本方案的 TAG_TO_NODE_TYPE 一致）
2. 父子关系：边界框最小覆盖父
3. 样式：从 inline style 正则提取（与本方案 _extract_color_from_html 一致）

输出：和模型推理同样格式的 figma JSON，复用 metrics.compute_full_metrics 评测。

用法：
    python scripts/baseline_heuristic.py --config configs/default.yaml --split val
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import WebpageDataset, webpage_collate_fn
from src.models.decoder import (
    NODE_TYPE_TO_IDX, IDX_TO_NODE_TYPE, NUM_NODE_TYPES,
    html_tag_to_node_type,
)
from src.eval.metrics import (
    type_accuracy, predicted_parents, parent_f1, tree_edit_distance,
)


def heuristic_predict(batch: dict) -> dict:
    """
    单样本（batch=1）的启发式预测：
    - type 来自 HTML tag
    - parent 来自最小覆盖父（box 包含关系，与原始 _build_tree 一样）
    """
    node_texts = batch["node_texts"][0]
    node_boxes = batch["node_boxes"][0]
    node_mask = batch["node_mask"][0]
    N = node_mask.shape[0]
    n_valid = int(node_mask.sum().item())

    # type
    type_indices = torch.zeros(N, dtype=torch.long)
    for i, t in enumerate(node_texts):
        tag = t.split(" ")[0].strip("<").strip(">")
        type_indices[i] = NODE_TYPE_TO_IDX.get(html_tag_to_node_type(tag), 0)

    # parent: 最小覆盖父 box 包含关系
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

    # 仿照模型 outputs 的格式（造一个伪 type_logits 让 metrics 复用）
    fake_type_logits = torch.zeros(1, N, NUM_NODE_TYPES)
    for i in range(N):
        fake_type_logits[0, i, type_indices[i]] = 10.0

    return {
        "type_indices": type_indices,
        "type_logits": fake_type_logits,
        "predicted_parents": parents,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--output_dir", default="outputs/B1-heuristic")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    full = WebpageDataset(
        data_dir=cfg["data"]["data_dir"],
        max_nodes=cfg["data"]["max_nodes"],
        iou_threshold=cfg["data"]["iou_threshold"],
    )
    n = len(full)
    n_train = int(n * cfg["data"]["train_ratio"])
    if args.split == "train":
        ds = Subset(full, list(range(n_train)))
    else:
        ds = Subset(full, list(range(n_train, n)))

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=webpage_collate_fn)

    type_correct, type_total = 0.0, 0.0
    f1s, ps, rs, teds = [], [], [], []

    from src.training.trainer import build_type_targets
    for batch in loader:
        N = batch["node_mask"].shape[1]
        target_types = build_type_targets(batch["node_texts"], N)[0]
        out = heuristic_predict(batch)

        # type acc
        pred = out["type_indices"]
        m = batch["node_mask"][0]
        type_correct += float(((pred == target_types).float() * m).sum())
        type_total += float(m.sum())

        # parent F1 + ted
        gold = batch["parents"][0].cpu().tolist()
        pp = out["predicted_parents"]
        f = parent_f1(pp, gold, m)
        ps.append(f["precision"]); rs.append(f["recall"]); f1s.append(f["f1"])
        teds.append(tree_edit_distance(pp, gold))

    metrics = {
        "type_acc": type_correct / max(1.0, type_total),
        "parent_p": sum(ps) / max(1, len(ps)),
        "parent_r": sum(rs) / max(1, len(rs)),
        "parent_f1": sum(f1s) / max(1, len(f1s)),
        "tree_edit": sum(teds) / max(1, len(teds)),
    }
    print("=== 启发式基线（B1） ===")
    for k, v in metrics.items():
        print(f"  {k:14s} = {v:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_p = os.path.join(args.output_dir, f"metrics_{args.split}.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 → {out_p}")


if __name__ == "__main__":
    main()
