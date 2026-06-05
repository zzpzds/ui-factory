"""
从训练日志和评测 JSON 自动生成最终的实验报告（M1-final.md）。

用法：
    python scripts/fill_report.py \
        --runs M1-baseline,A1a-no-align,A1b-bce-only,A4a-max50,A4b-max128 \
        --output docs/experiments/M1-final.md
"""
import os
import sys
import json
import argparse
from pathlib import Path


def load_log(run_dir: str) -> tuple[list, list]:
    p = Path(run_dir) / "train.log.jsonl"
    train, val = [], []
    if not p.exists():
        return train, val
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("phase") == "train":
                train.append(rec)
            elif rec.get("phase") == "val":
                val.append(rec)
    return train, val


def load_metrics(run_dir: str, split: str = "val") -> dict:
    p = Path(run_dir) / f"metrics_{split}.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fmt(v, prec=3):
    if v is None or v == "":
        return "—"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="comma-separated run names")
    p.add_argument("--outputs_dir", default="outputs")
    p.add_argument("--output", default="docs/experiments/M1-final.md")
    args = p.parse_args()

    runs = [r.strip() for r in args.runs.split(",")]
    rows = []
    for run in runs:
        run_dir = os.path.join(args.outputs_dir, run)
        train, val = load_log(run_dir)
        metrics = load_metrics(run_dir, "val")
        last_train = train[-1] if train else {}
        last_val = val[-1] if val else {}
        best_val = min(val, key=lambda r: r.get("total", 1e9)) if val else {}
        rows.append({
            "run": run,
            "epochs": len(val),
            "final_train_total": last_train.get("total"),
            "final_val_total": last_val.get("total"),
            "best_val_total": best_val.get("total"),
            "best_val_type_acc": best_val.get("type_acc"),
            "metrics": metrics,
        })

    # 渲染
    out_lines = []
    out_lines.append("# 实验最终报告（自动生成）\n")
    out_lines.append(f"runs: {', '.join(runs)}\n")
    out_lines.append("\n## 指标对比表\n")
    out_lines.append("| Run | epochs | best val total | best val type_acc | parent_f1 | tree_edit | style_reg_mae | color_de |")
    out_lines.append("|-----|--------|----------------|-------------------|-----------|-----------|---------------|----------|")
    for r in rows:
        m = r["metrics"]
        out_lines.append(
            f"| {r['run']} | {r['epochs']} | {fmt(r['best_val_total'])} | {fmt(r['best_val_type_acc'])} | "
            f"{fmt(m.get('parent_f1'))} | {fmt(m.get('tree_edit'))} | {fmt(m.get('style_reg_mae'))} | "
            f"{fmt(m.get('color_de'))} |"
        )

    out_lines.append("\n## 各 run 详细\n")
    for r in rows:
        out_lines.append(f"### {r['run']}\n")
        m = r["metrics"]
        out_lines.append("| 指标 | 数值 |")
        out_lines.append("|------|------|")
        for k in ("type_acc", "parent_p", "parent_r", "parent_f1", "tree_edit",
                 "style_reg_mae", "color_de", "cls_textAlign_acc", "cls_display_acc"):
            if k in m:
                out_lines.append(f"| {k} | {fmt(m[k])} |")
        out_lines.append("")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"📊 报告 → {args.output}")


if __name__ == "__main__":
    main()
