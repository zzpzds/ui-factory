"""校验并汇总 AI 辅助、人工审核的 Pilot 金标准。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_annotation_agreement import cohen_kappa
from src.annotation.store import AnnotationStore
from src.design_intent.metrics import compute_intent_metrics
from src.design_intent.schema import DesignIntentIR, PageGraph


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_without_provenance(ir: DesignIntentIR) -> dict[str, Any]:
    payload = ir.to_dict()
    payload.pop("provenance", None)
    return payload


def comparison_metrics(
    reference: DesignIntentIR,
    gold: DesignIntentIR,
    graph: PageGraph,
) -> dict[str, float]:
    metrics = compute_intent_metrics(reference, gold, graph)
    metrics["layout_cohen_kappa"] = cohen_kappa(reference, gold)
    return metrics


def mean_metrics(values: dict[str, list[float]]) -> dict[str, float]:
    return {
        name: sum(items) / len(items)
        for name, items in sorted(values.items())
        if items
    }


def summarize(repo_root: Path, package_dir: Path) -> dict[str, Any]:
    assignment = json.loads(
        (package_dir / "assignment.json").read_text(encoding="utf-8")
    )
    human_totals: dict[str, list[float]] = defaultdict(list)
    ai_totals: dict[str, list[float]] = defaultdict(list)
    per_sample = {}
    errors = []
    exact_human = 0
    exact_ai = 0

    for item in assignment["samples"]:
        sample_id = str(item["sample_id"])
        graph_path = repo_root / item["page_graph"]
        gold_path = graph_path.parent / "gold_intent.json"
        human_path = package_dir / "annotator_a" / f"{sample_id}.json"
        ai_path = package_dir / "annotator_ai" / f"{sample_id}.json"
        record_path = package_dir / "adjudication" / f"{sample_id}.json"
        draft_path = package_dir / "gold_drafts" / f"{sample_id}.json"
        required = [graph_path, gold_path, human_path, ai_path, record_path, draft_path]
        missing = [str(path.relative_to(repo_root)) for path in required if not path.exists()]
        if missing:
            errors.append({"sample_id": sample_id, "errors": [f"缺少文件：{path}" for path in missing]})
            continue

        graph = PageGraph.load(graph_path)
        gold = DesignIntentIR.load(gold_path)
        human = DesignIntentIR.load(human_path)
        ai = DesignIntentIR.load(ai_path)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        sample_errors = AnnotationStore._submission_errors(gold, graph)
        if gold.provenance.get("label_source") != "human_ai_adjudicated_gold":
            sample_errors.append("gold label_source 非 human_ai_adjudicated_gold")
        if gold.provenance.get("ai_assistance_disclosed") is not True:
            sample_errors.append("gold 未披露 AI assistance")
        if record.get("status") != "reviewed":
            sample_errors.append("仲裁记录不是 reviewed")
        if not str(record.get("review", {}).get("notes", "")).strip():
            sample_errors.append("仲裁审核说明为空")
        if sample_errors:
            errors.append({"sample_id": sample_id, "errors": sample_errors})
            continue

        human_metrics = comparison_metrics(human, gold, graph)
        ai_metrics = comparison_metrics(ai, gold, graph)
        for name, value in human_metrics.items():
            human_totals[name].append(value)
        for name, value in ai_metrics.items():
            ai_totals[name].append(value)

        matches_human = content_without_provenance(gold) == content_without_provenance(human)
        matches_ai = content_without_provenance(gold) == content_without_provenance(ai)
        exact_human += int(matches_human)
        exact_ai += int(matches_ai)
        per_sample[sample_id] = {
            "reviewed_by": record["review"]["reviewed_by"],
            "reviewed_at": record["review"]["reviewed_at"],
            "gold_sha256": sha256(gold_path),
            "counts": {
                "elements": len(gold.elements),
                "groups": len(gold.groups),
                "layouts": len(gold.layouts),
                "tree_edges": len(gold.tree),
                "style_tokens": len(gold.style_tokens),
            },
            "exact_content_match": {"human": matches_human, "ai": matches_ai},
            "human_to_gold": human_metrics,
            "ai_to_gold": ai_metrics,
        }

    total = len(assignment["samples"])
    return {
        "schema_version": "1.0",
        "study_design": "AI 代理独立候选 + 真实人员逐页仲裁",
        "interpretation": "跨来源描述性比较，不是双人标注者间一致性",
        "assigned": total,
        "validated_gold": len(per_sample),
        "complete": len(per_sample) == total and not errors,
        "errors": errors,
        "exact_content_match_counts": {"human": exact_human, "ai": exact_ai},
        "aggregate": {
            "human_to_gold": mean_metrics(human_totals),
            "ai_to_gold": mean_metrics(ai_totals),
        },
        "per_sample": per_sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package_dir", default="data/annotations/intent_pilot_v1")
    parser.add_argument(
        "--output",
        default=(
            "data/annotations/intent_pilot_v1/adjudication/"
            "gold_summary.json"
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    result = summarize(repo_root, (repo_root / args.package_dir).resolve())
    output = (repo_root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
