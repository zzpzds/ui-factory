"""仅在人工完成仲裁记录后导出 AI 辅助的人审金标准。"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.prepare_human_ai_adjudication import sha256, write_json
from src.annotation.store import AnnotationStore
from src.design_intent.schema import DesignIntentIR, PageGraph


def review_errors(record: dict[str, Any], draft: DesignIntentIR) -> list[str]:
    errors = []
    review = record.get("review", {})
    provenance = draft.provenance
    if record.get("status") != "reviewed":
        errors.append("仲裁记录 status 必须为 reviewed")
    if not review.get("reviewed_by"):
        errors.append("仲裁记录缺少 reviewed_by")
    if review.get("reviewed_by") == "annotator_ai":
        errors.append("AI 代理不能作为金标准审核人")
    if not review.get("reviewed_at"):
        errors.append("仲裁记录缺少 reviewed_at")
    if not str(review.get("notes", "")).strip():
        errors.append("仲裁记录缺少审核说明 notes")

    if provenance.get("label_source") != "human_ai_adjudication_draft":
        errors.append("金标准草稿来源字段非法")
    if provenance.get("status") != "complete":
        errors.append("金标准草稿 status 必须为 complete")
    if provenance.get("adjudication_status") != "reviewed":
        errors.append("金标准草稿 adjudication_status 必须为 reviewed")
    if provenance.get("gold_finalized") is not False:
        errors.append("待导出草稿的 gold_finalized 必须为 false")
    if not provenance.get("reviewed_by"):
        errors.append("金标准草稿缺少 reviewed_by")
    if provenance.get("reviewed_by") != review.get("reviewed_by"):
        errors.append("仲裁记录与金标准草稿的审核人不一致")
    if provenance.get("reviewed_at") != review.get("reviewed_at"):
        errors.append("仲裁记录与金标准草稿的审核时间不一致")
    if not str(provenance.get("review_notes", "")).strip():
        errors.append("金标准草稿缺少 review_notes")
    return errors


def finalize_adjudication(
    repo_root: Path,
    package_dir: Path,
    adjudication_dir: Path,
    gold_drafts_dir: Path,
    output_dir: Path | None = None,
    sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    assignment = json.loads(
        (package_dir / "assignment.json").read_text(encoding="utf-8")
    )
    result: dict[str, Any] = {"exported": [], "existing": [], "errors": []}
    selected = sample_ids or {
        str(item["sample_id"]) for item in assignment["samples"]
    }

    for item in assignment["samples"]:
        sample_id = str(item["sample_id"])
        if sample_id not in selected:
            continue
        record_path = adjudication_dir / f"{sample_id}.json"
        draft_path = gold_drafts_dir / f"{sample_id}.json"
        if not record_path.exists() or not draft_path.exists():
            result["errors"].append(
                {"sample_id": sample_id, "errors": ["缺少仲裁记录或金标准草稿"]}
            )
            continue

        record = json.loads(record_path.read_text(encoding="utf-8"))
        draft = DesignIntentIR.load(draft_path)
        errors = review_errors(record, draft)

        human_path = package_dir / "annotator_a" / f"{sample_id}.json"
        ai_path = package_dir / "annotator_ai" / f"{sample_id}.json"
        if record.get("inputs", {}).get("human", {}).get("sha256") != sha256(
            human_path
        ):
            errors.append("人工输入哈希已变化")
        if record.get("inputs", {}).get("ai", {}).get("sha256") != sha256(ai_path):
            errors.append("AI 代理输入哈希已变化")
        if draft.provenance.get("human_annotation_sha256") != sha256(human_path):
            errors.append("金标准草稿记录的人工输入哈希不一致")
        if draft.provenance.get("ai_annotation_sha256") != sha256(ai_path):
            errors.append("金标准草稿记录的 AI 输入哈希不一致")

        graph_path = Path(item["page_graph"])
        if not graph_path.is_absolute():
            graph_path = repo_root / graph_path
        graph = PageGraph.load(graph_path)
        errors.extend(AnnotationStore._submission_errors(draft, graph))
        if errors:
            result["errors"].append(
                {"sample_id": sample_id, "errors": list(dict.fromkeys(errors))}
            )
            continue

        if output_dir is None:
            destination = graph_path.parent / "gold_intent.json"
        else:
            destination = output_dir / sample_id / "gold_intent.json"

        finalized = copy.deepcopy(draft.to_dict())
        finalized["provenance"] = {
            **finalized["provenance"],
            "label_source": "human_ai_adjudicated_gold",
            "adjudication_status": "finalized",
            "gold_finalized": True,
            "ai_assistance_disclosed": True,
            "adjudication_record": str(
                record_path.relative_to(repo_root)
                if record_path.is_relative_to(repo_root)
                else record_path
            ),
        }
        if destination.exists():
            existing = DesignIntentIR.load(destination).to_dict()
            if existing == finalized:
                result["existing"].append(sample_id)
                continue
            result["errors"].append(
                {
                    "sample_id": sample_id,
                    "errors": [f"目标已存在且内容不同：{destination}"],
                }
            )
            continue
        write_json(destination, finalized)
        result["exported"].append(sample_id)

    unknown = sorted(selected - {str(item["sample_id"]) for item in assignment["samples"]})
    for sample_id in unknown:
        result["errors"].append(
            {"sample_id": sample_id, "errors": ["样本不在 assignment 中"]}
        )
    result["requested"] = len(selected)
    result["complete"] = not result["errors"]
    result["gold_label_source"] = "human_ai_adjudicated_gold"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package_dir", default="data/annotations/intent_pilot_v1"
    )
    parser.add_argument(
        "--adjudication_dir",
        default="data/annotations/intent_pilot_v1/adjudication",
    )
    parser.add_argument(
        "--gold_drafts_dir",
        default="data/annotations/intent_pilot_v1/gold_drafts",
    )
    parser.add_argument(
        "--output_dir",
        help="可选；指定后写入 <output_dir>/<sample_id>/gold_intent.json",
    )
    parser.add_argument("--sample_id", action="append")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    result = finalize_adjudication(
        repo_root=repo_root,
        package_dir=(repo_root / args.package_dir).resolve(),
        adjudication_dir=(repo_root / args.adjudication_dir).resolve(),
        gold_drafts_dir=(repo_root / args.gold_drafts_dir).resolve(),
        output_dir=(
            (repo_root / args.output_dir).resolve() if args.output_dir else None
        ),
        sample_ids=set(args.sample_id) if args.sample_id else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
