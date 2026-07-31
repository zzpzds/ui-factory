"""校验双人标注包并计算试标注一致性门槛。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.design_intent.metrics import compute_intent_metrics
from src.design_intent.schema import DesignIntentIR, PageGraph
from src.design_intent.validation import validate_intent_ir


def source_signature(ir: DesignIntentIR, group_id: str) -> tuple[int, ...]:
    elements = {element.id: element for element in ir.elements}
    group = next(group for group in ir.groups if group.id == group_id)
    return tuple(
        sorted(
            {
                node_id
                for element_id in group.source_element_ids
                for node_id in elements[element_id].source_node_ids
            }
        )
    )


def layout_modes(ir: DesignIntentIR) -> dict[tuple[int, ...], str]:
    group_ids = {group.id for group in ir.groups}
    return {
        source_signature(ir, layout.target_id): layout.mode
        for layout in ir.layouts
        if layout.target_id in group_ids
    }


def cohen_kappa(left: DesignIntentIR, right: DesignIntentIR) -> float:
    left_modes = layout_modes(left)
    right_modes = layout_modes(right)
    common = set(left_modes) & set(right_modes)
    if not common:
        return 0.0
    left_values = [left_modes[key] for key in common]
    right_values = [right_modes[key] for key in common]
    observed = sum(
        left_value == right_value
        for left_value, right_value in zip(left_values, right_values)
    ) / len(common)
    left_counts = Counter(left_values)
    right_counts = Counter(right_values)
    classes = set(left_counts) | set(right_counts)
    expected = sum(
        left_counts[value] / len(common) * right_counts[value] / len(common)
        for value in classes
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package_dir", default="data/annotations/intent_pilot_v1"
    )
    parser.add_argument(
        "--output", default="outputs/intent-pilot-500/agreement.json"
    )
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    assignment = json.loads(
        (package_dir / "assignment.json").read_text(encoding="utf-8")
    )
    per_sample = {}
    incomplete = []
    invalid = []
    totals: dict[str, list[float]] = defaultdict(list)
    for item in assignment["samples"]:
        sample_id = item["sample_id"]
        graph = PageGraph.load(item["page_graph"])
        left = DesignIntentIR.load(
            package_dir / "annotator_a" / f"{sample_id}.json"
        )
        right = DesignIntentIR.load(
            package_dir / "annotator_b" / f"{sample_id}.json"
        )
        if (
            left.provenance.get("status") != "complete"
            or right.provenance.get("status") != "complete"
        ):
            incomplete.append(sample_id)
            continue
        errors = validate_intent_ir(left, graph) + validate_intent_ir(
            right, graph
        )
        if errors:
            invalid.append({"sample_id": sample_id, "errors": errors})
            continue
        metrics = compute_intent_metrics(left, right, graph)
        metrics["layout_cohen_kappa"] = cohen_kappa(left, right)
        per_sample[sample_id] = metrics
        for name, value in metrics.items():
            totals[name].append(value)

    aggregate = {
        name: sum(values) / len(values) for name, values in totals.items()
    }
    thresholds = {
        "leaf_f1": (">=", 0.85),
        "group_bcubed_f1": (">=", 0.75),
        "parent_f1": (">=", 0.80),
        "layout_cohen_kappa": (">=", 0.75),
        "gap_normalized_mae": ("<=", 0.10),
        "padding_normalized_mae": ("<=", 0.10),
        "token_bcubed_f1": (">=", 0.70),
    }
    gates = {}
    for name, (operator, threshold) in thresholds.items():
        value = aggregate.get(name)
        gates[name] = (
            value is not None
            and (value >= threshold if operator == ">=" else value <= threshold)
        )
    result = {
        "assigned": len(assignment["samples"]),
        "completed_pairs": len(per_sample),
        "incomplete": incomplete,
        "invalid": invalid,
        "aggregate": aggregate,
        "thresholds": {
            name: {"operator": operator, "value": value}
            for name, (operator, value) in thresholds.items()
        },
        "gates": gates,
        "passed": (
            len(per_sample) == len(assignment["samples"])
            and not invalid
            and all(gates.values())
        ),
        "per_sample": per_sample,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if incomplete or invalid or not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
