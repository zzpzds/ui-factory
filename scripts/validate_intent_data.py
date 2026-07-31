"""批量校验页面图与 Design Intent 标注。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.design_intent.schema import DesignIntentIR, PageGraph
from src.design_intent.validation import validate_intent_ir, validate_page_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/sampled")
    parser.add_argument("--annotation_name", default="weak_intent.json")
    parser.add_argument(
        "--manifest", default=None, help="包含 sample_ids 的数据清单"
    )
    args = parser.parse_args()

    selected_ids = None
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as f:
            selected_ids = set(json.load(f)["sample_ids"])
    checked = 0
    failures = []
    for sample_dir in sorted(Path(args.data_dir).iterdir()):
        if selected_ids is not None and sample_dir.name not in selected_ids:
            continue
        graph_path = sample_dir / "page_graph.json"
        intent_path = sample_dir / args.annotation_name
        if not graph_path.exists() or not intent_path.exists():
            continue
        graph = PageGraph.load(graph_path)
        ir = DesignIntentIR.load(intent_path)
        errors = validate_page_graph(graph) + validate_intent_ir(ir, graph)
        checked += 1
        if errors:
            failures.append({"sample_id": sample_dir.name, "errors": errors})

    result = {
        "checked": checked,
        "failed": len(failures),
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
