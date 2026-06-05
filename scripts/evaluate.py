"""
评测脚本：加载 checkpoint，在数据集上计算完整指标 + 生成论文用可视化。

用法：
    python scripts/evaluate.py --checkpoint outputs/M1-baseline/best.pt \
                               --config configs/default.yaml \
                               --split val \
                               --viz_dir outputs/M1-baseline/viz
"""
import os
import sys
import json
import argparse

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import WebpageDataset, webpage_collate_fn
from src.training.trainer import FigmaGenerationModel, build_type_targets
from src.eval.metrics import (
    compute_full_metrics, confusion_matrix, predicted_parents,
)
from src.viz.figures import (
    plot_loss_curves, plot_confusion_matrix,
    plot_attention_heatmap, plot_tree_compare, plot_struct_depth_hist,
)
from src.models.decoder import IDX_TO_NODE_TYPE, NUM_NODE_TYPES


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--viz_dir", default=None)
    p.add_argument("--n_attn_samples", type=int, default=3)
    p.add_argument("--structure_mode", choices=["model", "heuristic", "hybrid"], default="model")
    p.add_argument("--init_heuristic", action="store_true",
                   help="启发式 parents 作为迭代推理的初始化（让模型从合理起点优化）")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = get_device()
    print(f"[device] {device}")

    # 数据
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
        ds = Subset(full, list(range(n_train, n))) if n > n_train else full
    print(f"[data] split={args.split}  size={len(ds)}")

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=webpage_collate_fn)

    # 模型
    mc = cfg["model"]
    model = FigmaGenerationModel(
        dim=mc["dim"], align_heads=mc["align_heads"],
        ctx_layers=mc["ctx_layers"], ctx_heads=mc["ctx_heads"],
        struct_layers=mc["struct_layers"], struct_heads=mc["struct_heads"],
        freeze_visual=mc["freeze_visual"],
        codebert_unfreeze_last=mc["codebert_unfreeze_last"],
    ).to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    print(f"[model] loaded {args.checkpoint}")

    # 收集
    aggregate = {}
    cm_total = torch.zeros(NUM_NODE_TYPES, NUM_NODE_TYPES, dtype=torch.long)
    pred_parents_all, gold_parents_all = [], []
    sample_for_viz = None

    from src.models.decoder.figma_decoder import heuristic_parents_from_boxes
    from src.eval.metrics import parent_f1, tree_edit_distance

    for step, batch in enumerate(loader):
        target_types = build_type_targets(batch["node_texts"], batch["node_mask"].shape[1])
        # 迭代推理：解决 train/inference 在 depth/parents 上的 mismatch
        out = model.iterative_inference(batch, n_iters=3, init_with_heuristic=args.init_heuristic)
        m = compute_full_metrics(
            out, target_types.to(device), batch["parents"].to(device),
            batch["node_mask"].to(device), batch["styles"],
        )
        for k, v in m.items():
            aggregate[k] = aggregate.get(k, 0.0) + v
        cm_total += confusion_matrix(
            out["type_logits"], target_types.to(device), batch["node_mask"].to(device),
            num_classes=NUM_NODE_TYPES,
        )

        # 根据 structure_mode 决定使用哪种父链
        if args.structure_mode == "heuristic":
            pp = heuristic_parents_from_boxes(
                batch["node_boxes"][0].cpu(), batch["node_mask"][0].cpu(),
            )
        elif args.structure_mode == "hybrid":
            model_pp = out["predicted_parents"][0].cpu().tolist()
            heur_pp = heuristic_parents_from_boxes(
                batch["node_boxes"][0].cpu(), batch["node_mask"][0].cpu(),
            )
            probs = torch.softmax(out["parent_logits"][0].cpu(), dim=-1)
            max_prob = probs.max(dim=-1).values
            pp = [
                (int(model_pp[i]) if max_prob[i].item() >= 0.3 else int(heur_pp[i]))
                for i in range(len(model_pp))
            ]
        else:
            pp = out["predicted_parents"][0].cpu().tolist()

        gp = batch["parents"][0].cpu().tolist()
        pred_parents_all.append(pp); gold_parents_all.append(gp)

        # 用所选模式重新覆盖 parent_f1 / tree_edit
        f = parent_f1(pp, gp, batch["node_mask"][0])
        aggregate.setdefault("parent_p_mode", 0.0)
        aggregate["parent_p_mode"] += f["precision"]
        aggregate.setdefault("parent_r_mode", 0.0)
        aggregate["parent_r_mode"] += f["recall"]
        aggregate.setdefault("parent_f1_mode", 0.0)
        aggregate["parent_f1_mode"] += f["f1"]
        aggregate.setdefault("tree_edit_mode", 0.0)
        aggregate["tree_edit_mode"] += tree_edit_distance(pp, gp)

        if sample_for_viz is None:
            sample_for_viz = (batch, out, pp, gp)

    n_steps = max(1, len(loader))
    final = {k: v / n_steps for k, v in aggregate.items()}
    print("\n=== 指标 ===")
    for k, v in final.items():
        print(f"  {k:18s} = {v:.4f}")

    # 可视化
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)
        # 1) loss 曲线（如有 log）
        log_path = os.path.join(os.path.dirname(args.checkpoint), "train.log.jsonl")
        if os.path.exists(log_path):
            plot_loss_curves(
                log_path,
                os.path.join(args.viz_dir, "fig_loss_curves.png"),
                metrics=["total", "type", "struct", "align_bce"],
            )
        # 2) 混淆矩阵
        labels = [IDX_TO_NODE_TYPE[i] for i in range(NUM_NODE_TYPES)]
        plot_confusion_matrix(cm_total, labels, os.path.join(args.viz_dir, "fig_cm.png"))
        # 3) 树并排（用 viz 样本）
        if sample_for_viz:
            batch, out, pp, gp = sample_for_viz
            labels_one = batch["node_texts"][0]
            plot_tree_compare(pp, gp, labels_one, os.path.join(args.viz_dir, "fig_tree_compare.png"))
            # 4) attention heatmap
            attn = out["sim_logits"][0].sigmoid().cpu()  # 用监督通道更稳定
            n_valid = int(batch["node_mask"][0].sum().item())
            picked = list(range(min(args.n_attn_samples, n_valid)))
            plot_attention_heatmap(
                batch["image"][0], attn, picked,
                [labels_one[i] for i in picked],
                os.path.join(args.viz_dir, "fig_attention.png"),
            )
        # 5) 深度分布
        plot_struct_depth_hist(
            pred_parents_all, gold_parents_all,
            os.path.join(args.viz_dir, "fig_depth_hist.png"),
        )
        print(f"\n📊 可视化保存到 {args.viz_dir}")

    # 保存指标 JSON
    out_metrics = os.path.join(
        os.path.dirname(args.checkpoint),
        f"metrics_{args.split}_{args.structure_mode}.json"
    )
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"📊 指标 → {out_metrics}")


if __name__ == "__main__":
    main()
