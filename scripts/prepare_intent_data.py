"""把旧渲染数据迁移为 PageGraph，并生成弱监督 Intent IR。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.design_intent.page_graph import page_graph_from_legacy_sample
from src.design_intent.validation import validate_intent_ir, validate_page_graph
from src.design_intent.weak_supervision import build_weak_intent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/sampled")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--manifest", default=None, help="包含 sample_ids 的数据清单"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--rebuild_graph",
        action="store_true",
        help="即使 page_graph.json 已存在，也从旧张量文件重建",
    )
    args = parser.parse_args()

    sample_dirs = [
        path
        for path in sorted(Path(args.data_dir).iterdir())
        if path.is_dir()
        and (path / "nodes.pt").exists()
        and (path / "parents.pt").exists()
        and (path / "node_texts.json").exists()
        and (path / "styles.json").exists()
    ]
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as f:
            selected_ids = set(json.load(f)["sample_ids"])
        sample_dirs = [
            path for path in sample_dirs if path.name in selected_ids
        ]
    if args.limit > 0:
        sample_dirs = sample_dirs[:args.limit]

    completed = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    for sample_dir in sample_dirs:
        graph_path = sample_dir / "page_graph.json"
        intent_path = sample_dir / "weak_intent.json"
        if not args.force and graph_path.exists() and intent_path.exists():
            skipped += 1
            continue
        try:
            if graph_path.exists() and not args.rebuild_graph:
                from src.design_intent.schema import PageGraph

                graph = PageGraph.load(graph_path)
            else:
                graph = page_graph_from_legacy_sample(sample_dir)
            graph_errors = validate_page_graph(graph)
            if graph_errors:
                raise ValueError("; ".join(graph_errors))
            intent = build_weak_intent(graph)
            intent_errors = validate_intent_ir(intent, graph)
            if intent_errors:
                raise ValueError("; ".join(intent_errors))
            graph.dump(graph_path)
            intent.dump(intent_path)
            completed += 1
        except Exception as error:
            failures.append({"sample_id": sample_dir.name, "error": str(error)})

    summary = {
        "samples": len(sample_dirs),
        "completed": completed,
        "skipped": skipped,
        "failed": len(failures),
        "failures": failures,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
