"""用一个现有渲染样本跑通 Design Intent 最小闭环。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.design_intent.figma_export import export_figma_json
from src.design_intent.page_graph import page_graph_from_legacy_sample
from src.design_intent.schema import PageGraph
from src.design_intent.validation import validate_intent_ir, validate_page_graph
from src.design_intent.weak_supervision import build_weak_intent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", default="data/sampled/0000")
    parser.add_argument("--output_dir", default="outputs/intent-pilot")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    if not sample_dir.exists():
        candidates = sorted(Path("data/sampled").glob("*"))
        sample_dir = next((path for path in candidates if path.is_dir()), sample_dir)
    if not sample_dir.exists():
        raise FileNotFoundError("没有找到可用于 pilot 的渲染样本")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = sample_dir / "page_graph.json"
    graph = (
        PageGraph.load(graph_path)
        if graph_path.exists()
        else page_graph_from_legacy_sample(sample_dir)
    )
    graph_errors = validate_page_graph(graph)
    if graph_errors:
        raise ValueError("PageGraph 校验失败：\n" + "\n".join(graph_errors))
    graph.dump(output_dir / "page_graph.json")

    intent = build_weak_intent(graph)
    intent_errors = validate_intent_ir(intent, graph)
    if intent_errors:
        raise ValueError("DesignIntentIR 校验失败：\n" + "\n".join(intent_errors))
    intent.dump(output_dir / "weak_intent.json")

    with open(output_dir / "figma.json", "w", encoding="utf-8") as f:
        json.dump(export_figma_json(intent), f, ensure_ascii=False, indent=2)

    summary = {
        "sample_id": graph.sample_id,
        "nodes": len(graph.nodes),
        "elements": len(intent.elements),
        "groups": len(intent.groups),
        "layouts": len(intent.layouts),
        "style_tokens": len(intent.style_tokens),
        "tree_edges": len(intent.tree),
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
