"""替换正式参考集在全分辨率复核中发现的不合格样本。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_annotation_pilot import size_bin, structure_fingerprint
from scripts.create_formal_gold_package import (
    broken_image_proxy,
    content_safety_flags,
    primary_page_type,
    screenshot_quality,
    style_repeat_evidence,
)
from scripts.detect_near_duplicates import difference_hash, hamming_distance
from src.design_intent.schema import PageGraph


REPO_ROOT = Path(__file__).resolve().parents[1]
REPLACED_AT = "2026-08-31T00:00:00+08:00"
REPLACEMENTS = {
    "0144": {
        "new_sample_id": "0433",
        "reason": "全分辨率复核发现导航、正文与版权区域大面积错位重叠",
    },
    "0553": {
        "new_sample_id": "0584",
        "reason": "全分辨率复核发现页面正文持续推广赌场、老虎机与博彩策略",
    },
    "1365": {
        "new_sample_id": "1530",
        "reason": "全分辨率复核发现导航重复错位、内容碰撞且缺少稳定主体区域",
        "candidate_source": "eligible_render_pool",
        "token_rich_candidate": False,
    },
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _replacement_assignment(
    previous: dict[str, Any],
    new_id: str,
    selected: dict[str, Any],
) -> dict[str, Any]:
    return {
        **previous,
        "sample_id": new_id,
        "node_count": selected["node_count"],
        "size_bin": selected["size_bin"],
        "page_type": selected["page_type"],
        "structure_fingerprint": selected["structure_fingerprint"],
        "token_rich_candidate": selected["token_rich_candidate"],
        "status": selected.get("status", "annotation_required"),
        "screenshot": f"data/processed/{new_id}/screenshot.png",
        "page_graph": f"data/processed/{new_id}/page_graph.json",
    }


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(samples),
        "by_split": dict(sorted(Counter(
            item["split"] for item in samples
        ).items())),
        "by_size": dict(sorted(Counter(
            item["size_bin"] for item in samples
        ).items())),
        "by_page_type": dict(sorted(Counter(
            item["page_type"] for item in samples
        ).items())),
        "token_rich_candidates": sum(
            bool(item.get("token_rich_candidate")) for item in samples
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pool_candidate(
    sample_id: str,
    token_rich_candidate: bool,
    selected: dict[str, dict[str, Any]],
    replaced_id: str,
) -> dict[str, Any]:
    data_dir = REPO_ROOT / "data/processed"
    graph_path = data_dir / sample_id / "page_graph.json"
    screenshot_path = data_dir / sample_id / "screenshot.png"
    graph = PageGraph.load(graph_path)
    split_manifest = _load(REPO_ROOT / "data/intent_pilot_split.json")
    split_by_id = {
        value: split
        for split, values in split_manifest["splits"].items()
        for value in values
    }
    if sample_id not in split_by_id:
        raise ValueError(f"候选 {sample_id} 不在冻结 split 中")

    fingerprint = structure_fingerprint(graph)
    for item in selected.values():
        if item["sample_id"] != replaced_id and (
            item["structure_fingerprint"] == fingerprint
        ):
            raise ValueError(f"候选 {sample_id} 与 {item['sample_id']} 结构重复")
    candidate_hash = difference_hash(screenshot_path)
    for item in selected.values():
        if item["sample_id"] == replaced_id:
            continue
        distance = hamming_distance(
            candidate_hash,
            difference_hash(
                data_dir / item["sample_id"] / "screenshot.png"
            ),
        )
        if distance <= 12:
            raise ValueError(
                f"候选 {sample_id} 与 {item['sample_id']} 截图近重复：{distance}"
            )

    quality = screenshot_quality(screenshot_path)
    safety = content_safety_flags(graph)
    image_proxy = broken_image_proxy(graph)
    eligible = (
        8 <= len(graph.nodes) <= 180
        and quality["luminance_std"] >= 10
        and not safety
        and not image_proxy["mass_failure_suspected"]
    )
    if not eligible:
        raise ValueError(f"候选 {sample_id} 未通过自动质量门")
    return {
        "sample_id": sample_id,
        "split": split_by_id[sample_id],
        "node_count": len(graph.nodes),
        "size_bin": size_bin(len(graph.nodes)),
        "page_type": primary_page_type(graph),
        "structure_fingerprint": fingerprint,
        "duplicate_cluster": sample_id,
        "style_repeat_evidence": style_repeat_evidence(graph),
        "quality": quality,
        "asset_sha256": {
            "screenshot": _sha256(screenshot_path),
            "page_graph": _sha256(graph_path),
        },
        "content_safety_flags": safety,
        "broken_image_proxy": image_proxy,
        "eligible": True,
        "token_rich_candidate": token_rich_candidate,
        "selection_noise": None,
    }


def replace(package_dir: Path) -> dict[str, Any]:
    selection_path = package_dir / "selection_manifest.json"
    assignment_path = package_dir / "assignment.json"
    exclusions_path = package_dir / "visual_exclusions.json"
    selection = _load(selection_path)
    assignment = _load(assignment_path)
    exclusions = _load(exclusions_path)

    selected_by_id = {
        item["sample_id"]: item for item in selection["samples"]
    }
    reserve_by_id = {
        item["sample_id"]: item for item in selection["reserve_samples"]
    }
    assignment_by_id = {
        item["sample_id"]: item for item in assignment["samples"]
    }
    audit = {
        item["old_sample_id"]: item
        for item in selection.get("quality_replacements", [])
    }

    for old_id, config in REPLACEMENTS.items():
        new_id = config["new_sample_id"]
        if old_id not in selected_by_id and new_id in selected_by_id:
            new_selected = selected_by_id[new_id]
            if old_id in assignment_by_id:
                previous = assignment_by_id.pop(old_id)
                assignment_by_id.setdefault(
                    new_id,
                    _replacement_assignment(previous, new_id, new_selected),
                )
            elif new_id not in assignment_by_id:
                raise ValueError(
                    f"替换 {old_id} -> {new_id} 后 assignment 缺失两端样本"
                )
        else:
            if old_id not in selected_by_id:
                raise ValueError(f"无法执行替换 {old_id} -> {new_id}")

            old_selected = selected_by_id.pop(old_id)
            if new_id in reserve_by_id:
                new_selected = dict(reserve_by_id.pop(new_id))
            elif config.get("candidate_source") == "eligible_render_pool":
                new_selected = _pool_candidate(
                    new_id,
                    bool(config.get("token_rich_candidate")),
                    {**selected_by_id, old_id: old_selected},
                    old_id,
                )
            else:
                raise ValueError(f"无法执行替换 {old_id} -> {new_id}")
            if old_selected["split"] != new_selected["split"]:
                raise ValueError(f"替换 {old_id} -> {new_id} 会改变 split")
            new_selected["status"] = "annotation_required"
            selected_by_id[new_id] = new_selected

            old_assignment = assignment_by_id.pop(old_id)
            assignment_by_id[new_id] = _replacement_assignment(
                old_assignment, new_id, new_selected
            )

        exclusion = {
            "sample_id": old_id,
            "reason": config["reason"],
            "excluded_at": REPLACED_AT,
            "review_method": "full_resolution_ai_visual_semantic_review",
        }
        if not any(
            item["sample_id"] == old_id
            for item in selection["visual_exclusions"]
        ):
            selection["visual_exclusions"].append(exclusion)
        if not any(
            item["sample_id"] == old_id for item in exclusions["samples"]
        ):
            exclusions["samples"].append(exclusion)
        audit[old_id] = {
            "old_sample_id": old_id,
            "new_sample_id": new_id,
            "split": new_selected["split"],
            "reason": config["reason"],
            "replaced_at": REPLACED_AT,
            "reviewer_kind": "ai_visual_semantic_reviewer",
        }

    selection["samples"] = sorted(
        selected_by_id.values(), key=lambda item: item["sample_id"]
    )
    selection["purpose"] = "design_intent_reference_v1_sampling"
    selection["reserve_samples"] = sorted(
        reserve_by_id.values(), key=lambda item: item["sample_id"]
    )
    selection["quality_replacements"] = sorted(
        audit.values(), key=lambda item: item["old_sample_id"]
    )
    selection["summary"] = _summary(selection["samples"])
    selection["reserve_summary"] = _summary(selection["reserve_samples"])
    selection["splits"] = {
        split: sorted(
            item["sample_id"]
            for item in selection["samples"]
            if item["split"] == split
        )
        for split in ("train", "validation", "test")
    }
    selection["annotation_required_ids"] = sorted(
        item["sample_id"]
        for item in selection["samples"]
        if item["status"] == "annotation_required"
    )
    selection["visual_review"].update({
        "additional_full_resolution_review": REPLACED_AT,
        "additional_rejections": sorted(REPLACEMENTS),
    })

    assignment["samples"] = sorted(
        assignment_by_id.values(), key=lambda item: item["sample_id"]
    )
    exclusions["reviewed_date"] = "2026-08-31"
    exclusions["review_method"] = (
        "候选联系表、原始截图与 AI 全分辨率视觉语义复核"
    )

    _write(selection_path, selection)
    _write(assignment_path, assignment)
    _write(exclusions_path, exclusions)
    replacement_manifest = {
        "schema_version": "1.0",
        "purpose": "formal_reference_quality_replacement",
        "replaced_at": REPLACED_AT,
        "replacements": selection["quality_replacements"],
        "selected_samples_after_replacement": len(selection["samples"]),
        "reserve_samples_after_replacement": len(selection["reserve_samples"]),
    }
    _write(package_dir / "quality_replacement_manifest.json", replacement_manifest)
    return replacement_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package_dir", default="data/annotations/intent_gold_v1"
    )
    args = parser.parse_args()
    result = replace((REPO_ROOT / args.package_dir).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
