"""评测规则教师/启发式设计意图恢复基线。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.design_intent.metrics import compute_intent_metrics
from src.design_intent.schema import DesignIntentIR, PageGraph
from src.design_intent.validation import validate_intent_ir
from src.design_intent.weak_supervision import build_weak_intent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/sampled")
    parser.add_argument("--gold_name", default="gold_intent.json")
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/intent-baseline.json")
    args = parser.parse_args()

    totals: dict[str, list[float]] = defaultdict(list)
    failures = []
    evaluated = 0
    per_sample: dict[str, dict[str, float]] = {}
    selected: set[str] | None = None
    if args.split_manifest:
        with open(args.split_manifest, encoding="utf-8") as f:
            manifest = json.load(f)
        selected = set(manifest["splits"][args.split])
    for sample_dir in sorted(Path(args.data_dir).iterdir()):
        if selected is not None and sample_dir.name not in selected:
            continue
        graph_path = sample_dir / "page_graph.json"
        gold_path = sample_dir / args.gold_name
        if not graph_path.exists() or not gold_path.exists():
            continue
        graph = PageGraph.load(graph_path)
        gold = DesignIntentIR.load(gold_path)
        predicted = build_weak_intent(graph)
        errors = validate_intent_ir(predicted, graph)
        if errors:
            failures.append({"sample_id": sample_dir.name, "errors": errors})
            continue
        metrics = compute_intent_metrics(predicted, gold, graph)
        per_sample[sample_dir.name] = metrics
        for name, value in metrics.items():
            totals[name].append(value)
        evaluated += 1

    result = {
        "baseline": "weak_supervision_v1",
        "gold_name": args.gold_name,
        "split": args.split if selected is not None else "all",
        "evaluated": evaluated,
        "metrics": {
            name: sum(values) / len(values) for name, values in totals.items()
        },
        "per_sample": per_sample,
        "failures": failures,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
