"""生成 10 页设计师裁决 Gold + 50 页 AI 多视角参考标注。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotation.store import AnnotationStore
from src.design_intent.grouping import normalize_group_source_elements
from src.design_intent.schema import (
    BBox,
    DesignGroup,
    DesignIntentIR,
    LayoutConstraint,
    PageGraph,
    StyleToken,
    TreeEdge,
)
from src.design_intent.weak_supervision import build_weak_intent

from scripts.prepare_ai_assisted_gold import build_ai_prelabel


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARED_AT = "2026-08-31T00:00:00+08:00"
GENERATOR = "codex_multiview_reference_v1"
TOKEN_GENERATOR = "codex_compact_style_tokens_v1"
TOKEN_MIN_MEMBERS = 3
TOKEN_KIND_CAPS = {"COLOR": 4, "TEXT": 4, "RADIUS": 2, "SPACING": 2}
BRIEF_HUMAN_REVIEW_NOTE_SAMPLE_IDS = {
    "0001", "0020", "0052", "0319", "0419", "1094", "1429",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_candidate_is_current(
    annotation: DesignIntentIR,
    sample_id: str,
    graph_path: Path,
    screenshot_path: Path,
    expected: DesignIntentIR | None = None,
) -> bool:
    provenance = annotation.provenance
    provenance_matches = (
        provenance.get("source_sample_id") == sample_id
        and provenance.get("label_source") == "ai_preannotation"
        and provenance.get("page_graph_sha256") == _sha256(graph_path)
        and provenance.get("screenshot_sha256") == _sha256(screenshot_path)
    )
    return provenance_matches and (
        expected is None or annotation.to_dict() == expected.to_dict()
    )


def _remove_stale_json_files(directory: Path, allowed_stems: set[str]) -> None:
    for path in directory.glob("*.json"):
        if path.stem not in allowed_stems:
            path.unlink()


def _relative(path: Path, repo_root: Path = REPO_ROOT) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    return float(match.group(0)) if match else default


def _color(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    rgba = [round(float(channel), 3) for channel in value[:4]]
    if len(rgba) == 3:
        rgba.append(1.0)
    return rgba if rgba[3] > 0.05 else None


def _candidate_key(kind: str, value: dict[str, Any]) -> str:
    return f"{kind}|{json.dumps(value, ensure_ascii=False, sort_keys=True)}"


def attach_reference_tokens(annotation: DesignIntentIR) -> None:
    """自动保留高复用样式，并限制每页 Token 数量。"""
    buckets: dict[str, dict[str, Any]] = {}

    def add(
        kind: str, value: dict[str, Any], member_id: str, label: str
    ) -> None:
        key = _candidate_key(kind, value)
        candidate = buckets.setdefault(
            key,
            {"kind": kind, "value": value, "label": label, "members": set()},
        )
        candidate["members"].add(member_id)

    for entity in [*annotation.elements, *annotation.groups]:
        style = entity.style or {}
        background = _color(style.get("background_color"))
        if background:
            add(
                "COLOR",
                {"property": "background", "rgba": background},
                entity.id,
                "背景色",
            )

        is_textual = hasattr(entity, "type") and (
            getattr(entity, "type", "") in {"TEXT", "BUTTON_VISUAL"}
            or bool(getattr(entity, "text", ""))
        )
        foreground = _color(style.get("text_color"))
        if is_textual and foreground:
            add(
                "COLOR",
                {"property": "foreground", "rgba": foreground},
                entity.id,
                "文字色",
            )

        border = _color(style.get("border_color"))
        if border and _number(style.get("border_width")) > 0:
            add(
                "COLOR",
                {"property": "border", "rgba": border},
                entity.id,
                "边框色",
            )

        if is_textual:
            font_size = round(_number(style.get("font_size"), 16.0), 1)
            font_weight = int(
                round(_number(style.get("font_weight"), 400.0) / 100) * 100
            )
            add(
                "TEXT",
                {"font_size": font_size, "font_weight": font_weight},
                entity.id,
                "文字样式",
            )

        radius = round(_number(style.get("border_radius")), 1)
        if radius > 0:
            add("RADIUS", {"radius": radius}, entity.id, "圆角")

    for layout in annotation.layouts:
        gap = round(float(layout.gap), 1)
        if gap > 0:
            add(
                "SPACING",
                {"property": "gap", "spacing": gap},
                layout.target_id,
                "项目间距",
            )
        padding = [round(float(value), 1) for value in layout.padding]
        if padding and padding[0] > 0 and all(
            value == padding[0] for value in padding
        ):
            add(
                "SPACING",
                {"property": "padding", "spacing": padding[0]},
                layout.target_id,
                "内边距",
            )

    candidates = [
        candidate
        for candidate in buckets.values()
        if len(candidate["members"]) >= TOKEN_MIN_MEMBERS
    ]
    candidates.sort(
        key=lambda item: (
            list(TOKEN_KIND_CAPS).index(item["kind"]),
            -len(item["members"]),
            json.dumps(item["value"], ensure_ascii=False, sort_keys=True),
        )
    )

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        kind = candidate["kind"]
        if counts[kind] >= TOKEN_KIND_CAPS[kind]:
            continue
        counts[kind] += 1
        selected.append(candidate)

    annotation.style_tokens = []
    per_kind_index: dict[str, int] = defaultdict(int)
    for candidate in selected:
        kind = candidate["kind"]
        per_kind_index[kind] += 1
        index = per_kind_index[kind]
        annotation.style_tokens.append(StyleToken(
            id=f"token_{kind.lower()}_{index:02d}",
            kind=kind,
            value=candidate["value"],
            member_ids=sorted(candidate["members"]),
            name=f"{candidate['label']}/{index:02d}",
            confidence=0.9,
        ))

    for element in annotation.elements:
        element.style_token_refs = sorted(
            token.id
            for token in annotation.style_tokens
            if element.id in token.member_ids
        )


def _spatial_layout(boxes: list[BBox]) -> str:
    if len(boxes) < 2:
        return "FREE"
    x_centers = [box.x + box.width / 2 for box in boxes]
    y_centers = [box.y + box.height / 2 for box in boxes]
    x_spread = max(x_centers) - min(x_centers)
    y_spread = max(y_centers) - min(y_centers)
    avg_width = sum(box.width for box in boxes) / len(boxes)
    avg_height = sum(box.height for box in boxes) / len(boxes)
    if x_spread > y_spread * 1.5 and y_spread <= avg_height:
        return "HORIZONTAL"
    if y_spread > x_spread * 1.5 and x_spread <= avg_width:
        return "VERTICAL"
    return "FREE"


def _layout_gap(boxes: list[BBox], mode: str) -> float:
    if len(boxes) < 2 or mode not in {"HORIZONTAL", "VERTICAL"}:
        return 0.0
    if mode == "HORIZONTAL":
        ordered = sorted(boxes, key=lambda box: box.x)
        gaps = [
            max(0.0, right.x - left.x2)
            for left, right in zip(ordered, ordered[1:])
        ]
    else:
        ordered = sorted(boxes, key=lambda box: box.y)
        gaps = [
            max(0.0, lower.y - upper.y2)
            for upper, lower in zip(ordered, ordered[1:])
        ]
    return round(sum(gaps) / len(gaps), 3)


def apply_visual_correction_operations(
    annotation: DesignIntentIR,
    operations: list[dict[str, Any]],
) -> None:
    """将截图审阅确认的显式分组修正应用到结构候选。"""
    for operation in operations:
        if operation.get("action") != "create_group":
            raise ValueError(f"不支持的视觉修正操作：{operation.get('action')}")

        entity_by_id = {
            entity.id: entity
            for entity in [*annotation.elements, *annotation.groups]
        }
        group_id = str(operation["id"])
        child_ids = [str(value) for value in operation["child_ids"]]
        parent_id = str(operation.get("parent_id", "page_root"))
        if group_id in entity_by_id:
            raise ValueError(f"视觉修正分组 ID 已存在：{group_id}")
        if len(child_ids) < 2 or len(set(child_ids)) != len(child_ids):
            raise ValueError(f"视觉修正分组 {group_id} 至少需要两个不同子实体")
        missing = sorted(set(child_ids) - set(entity_by_id))
        if missing:
            raise ValueError(f"视觉修正分组 {group_id} 引用未知实体：{missing}")
        group_ids = {group.id for group in annotation.groups}
        if parent_id != "page_root" and parent_id not in group_ids:
            raise ValueError(f"视觉修正分组 {group_id} 的父组不存在：{parent_id}")
        if parent_id in child_ids:
            raise ValueError(f"视觉修正分组 {group_id} 不能包含其父组")

        child_boxes = [entity_by_id[child_id].bbox for child_id in child_ids]
        group = DesignGroup(
            id=group_id,
            source_element_ids=[],
            role=str(operation.get("role", "CONTAINER")),
            bbox=BBox.union(child_boxes),
            name=str(operation.get("name", group_id)),
            confidence=0.9,
        )
        parent_edges = [
            edge for edge in annotation.tree
            if edge.parent_id == parent_id and edge.child_id in child_ids
        ]
        parent_order = min(
            (edge.order for edge in parent_edges),
            default=max(
                (
                    edge.order for edge in annotation.tree
                    if edge.parent_id == parent_id
                ),
                default=-1,
            ) + 1,
        )
        annotation.tree = [
            edge for edge in annotation.tree if edge.child_id not in child_ids
        ]
        annotation.groups.append(group)
        annotation.tree.append(TreeEdge(
            parent_id=parent_id,
            child_id=group_id,
            order=parent_order,
            confidence=0.9,
        ))
        annotation.tree.extend(
            TreeEdge(
                parent_id=group_id,
                child_id=child_id,
                order=index,
                confidence=0.9,
            )
            for index, child_id in enumerate(child_ids)
        )
        mode = _spatial_layout(child_boxes)
        annotation.layouts.append(LayoutConstraint(
            target_id=group_id,
            mode=mode,
            gap=_layout_gap(child_boxes, mode),
            padding=[0.0, 0.0, 0.0, 0.0],
            primary_align="START",
            cross_align="START",
            horizontal_resize=(
                "STRETCH" if mode in {"HORIZONTAL", "GRID"} else "FIXED"
            ),
            vertical_resize=(
                "HUG" if mode in {"HORIZONTAL", "VERTICAL"} else "FIXED"
            ),
            confidence=0.8,
        ))

    normalize_group_source_elements(annotation)


def build_corrected_token_preview(
    structure: DesignIntentIR,
    visual_review: dict[str, Any],
) -> DesignIntentIR:
    annotation = copy.deepcopy(structure)
    apply_visual_correction_operations(
        annotation,
        list(visual_review.get("correction_operations", [])),
    )
    attach_reference_tokens(annotation)
    return annotation


def build_expanded_structure_candidate(graph: PageGraph) -> DesignIntentIR:
    """保留通用容器候选，为视觉上层级不足的页面提供第二条结构路径。"""
    annotation = build_weak_intent(graph)
    annotation.style_tokens = []
    for element in annotation.elements:
        element.style_token_refs = []
    annotation.provenance = {
        "source_sample_id": graph.sample_id,
        "label_source": "expanded_pagegraph_structure_candidate",
        "annotator_kind": "deterministic_pagegraph_heuristic",
        "generator": "pagegraph_expanded_structure_v1",
        "status": "complete",
        "human_labels_viewed_for_target": False,
        "visual_input_used": False,
        "page_graph_used": True,
        "generated_at": PREPARED_AT,
        "limitations": [
            "该文件是高召回结构候选，不是人工标注或最终银标准。",
            "通用容器可能产生过度分组，必须结合截图审阅选择。",
        ],
    }
    return annotation


def _load_visual_reviews(
    review_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    reviews: dict[str, dict[str, Any]] = {}
    batches: list[dict[str, Any]] = []
    for path in sorted(review_dir.glob("*.json")):
        payload = _load_json(path)
        if payload.get("review_kind") != "ai_visual_semantic_audit":
            raise ValueError(f"{path.name} 不是视觉语义审阅记录")
        if payload.get("human_labels_viewed_for_targets") is not False:
            raise ValueError(f"{path.name} 未声明目标页人工标签隔离")
        batch = {
            "path": _relative(path),
            "sha256": _sha256(path),
            "reviewer_id": payload.get("reviewer_id"),
            "sample_count": len(payload.get("reviews", [])),
        }
        batches.append(batch)
        for item in payload.get("reviews", []):
            sample_id = str(item["sample_id"])
            if sample_id in reviews:
                raise ValueError(f"视觉审阅记录重复：{sample_id}")
            reviews[sample_id] = {
                **item,
                "reviewer_id": payload.get("reviewer_id"),
                "review_path": path,
            }
    return reviews, batches


def _token_candidate_payload(
    sample_id: str,
    annotation: DesignIntentIR,
    structure_sha256: str,
    visual_review_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "sample_id": sample_id,
        "generator": TOKEN_GENERATOR,
        "generated_at": PREPARED_AT,
        "source_structure_sha256": structure_sha256,
        "source_visual_review_sha256": visual_review_sha256,
        "correction_operations_applied": True,
        "selection_policy": {
            "minimum_members": TOKEN_MIN_MEMBERS,
            "kind_caps": TOKEN_KIND_CAPS,
            "maximum_tokens_per_page": sum(TOKEN_KIND_CAPS.values()),
        },
        "style_tokens": annotation.to_dict()["style_tokens"],
    }


def _build_ai_reference(
    sample_id: str,
    conservative_structure: DesignIntentIR,
    conservative_path: Path,
    expanded_structure: DesignIntentIR,
    expanded_path: Path,
    visual_review: dict[str, Any],
    token_path: Path,
    screenshot_path: Path,
    graph_path: Path,
    diagnostic_replay_human_gold_sample_ids: list[str],
    diagnostic_replay_artifact: dict[str, Any],
) -> DesignIntentIR:
    if visual_review.get("quality_status") != "accepted":
        raise ValueError(f"{sample_id} 未通过视觉质量审阅")
    if visual_review.get("decision") not in {
        "accept_structure_candidate", "accept_with_limitations",
    }:
        raise ValueError(f"{sample_id} 尚未形成可接受的视觉裁决")
    if visual_review.get("correction_directives"):
        raise ValueError(f"{sample_id} 含未执行的视觉修正指令")

    preferred = visual_review.get(
        "preferred_structure_candidate", "conservative"
    )
    if preferred not in {"conservative", "expanded"}:
        raise ValueError(f"{sample_id} 的结构候选选择非法：{preferred}")
    selected_structure = (
        expanded_structure if preferred == "expanded" else conservative_structure
    )
    annotation = build_corrected_token_preview(
        selected_structure, visual_review
    )
    review_path = Path(visual_review["review_path"])
    annotation.provenance = {
        "source_sample_id": sample_id,
        "reference_tier": "ai_silver",
        "label_source": "ai_multiview_silver",
        "annotator": "codex_multiview_reference_v1",
        "annotator_kind": "ai_agent_pipeline",
        "status": "complete",
        "annotation_phase": "formal_reference_v1",
        "generator": GENERATOR,
        "generated_at": PREPARED_AT,
        "human_review_performed": False,
        "human_labels_viewed_for_target": False,
        "human_gold_used_for_generator_calibration": False,
        "human_gold_diagnostic_replay_performed": True,
        "generator_diagnostic_replay_sample_ids": (
            diagnostic_replay_human_gold_sample_ids
        ),
        "generator_diagnostic_replay_artifact": diagnostic_replay_artifact,
        "developer_human_gold_blinding_status": "not_guaranteed",
        "ai_treated_as_human_annotator": False,
        "visual_input_used": True,
        "page_graph_used": True,
        "selected_structure_candidate": preferred,
        "visual_review": {
            key: value
            for key, value in visual_review.items()
            if key != "review_path"
        },
        "review_roles": [
            {
                "role": "structure_candidate",
                "system": "deterministic_pagegraph_heuristic",
            },
            {
                "role": "visual_semantic_reviewer",
                "system": "OpenAI Codex",
                "reviewer_id": visual_review.get("reviewer_id"),
            },
            {
                "role": "multiview_adjudicator",
                "system": GENERATOR,
            },
        ],
        "pipeline_inputs": [
            {
                "kind": "pagegraph_structure_candidate",
                "path": _relative(conservative_path),
                "sha256": _sha256(conservative_path),
            },
            {
                "kind": "expanded_pagegraph_structure_candidate",
                "path": _relative(expanded_path),
                "sha256": _sha256(expanded_path),
            },
            {
                "kind": "screenshot_visual_semantic_review",
                "path": _relative(review_path),
                "sha256": _sha256(review_path),
            },
            {
                "kind": "computed_style_token_candidate",
                "path": _relative(token_path),
                "sha256": _sha256(token_path),
            },
        ],
        "source_assets": {
            "screenshot_path": _relative(screenshot_path),
            "screenshot_sha256": _sha256(screenshot_path),
            "page_graph_path": _relative(graph_path),
            "page_graph_sha256": _sha256(graph_path),
        },
        "limitations": [
            "该文件是 AI 多视角参考标注，不是人工标注。",
            "视觉角色审阅整体构图与语义，不等同于第二份完整独立 IR 标注。",
            "父子层级主要继承 PageGraph 候选，既有诊断性回放显示该维度风险较高。",
            "8 页 Human Gold 只做固定规则的诊断性回放，未用于选择或修改阈值。",
            "作者开发后续流程时可能已接触 Human Gold，不能声称开发过程完全盲法。",
            "各角色属于同一 Codex 工作流，不能解释为跨模型一致性。",
        ],
    }
    return annotation


def _prepare_diagnostic_replay_artifact(
    path: Path,
    diagnostic_sample_ids: list[str],
    manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError("缺少 AI 结构候选诊断性回放记录")
    historical = _load_json(path)
    calibration = historical.get("calibration", {})
    if calibration.get("scope") != "existing_locked_train_validation_only":
        raise ValueError("诊断性回放 scope 不符合冻结协议")
    if calibration.get("test_gold_used") is not False:
        raise ValueError("诊断性回放错误使用了 Human Gold test")
    replayed_ids = sorted(str(value) for value in calibration.get(
        "per_sample", {}
    ))
    if replayed_ids != diagnostic_sample_ids:
        raise ValueError(
            "诊断性回放样本与 assignment 不一致："
            f"expected={diagnostic_sample_ids}, actual={replayed_ids}"
        )
    if calibration.get("sample_count") != len(diagnostic_sample_ids):
        raise ValueError("诊断性回放 sample_count 不一致")

    planned_assisted = historical.pop(
        "human_assisted_samples",
        historical.get("human_assisted_samples_planned", 0),
    )
    planned_blind = historical.pop(
        "blind_control_samples",
        historical.get("blind_control_sample_ids_planned", []),
    )
    historical.update({
        "human_assisted_samples_planned": planned_assisted,
        "blind_control_sample_ids_planned": planned_blind,
        "human_reviews_completed": 0,
        "blind_control_annotations_completed": 0,
        "human_study_results_available": False,
        "status": "superseded",
        "superseded_at": PREPARED_AT,
        "superseded_by": _relative(manifest_path, repo_root),
        "calibration_interpretation": "diagnostic_replay_only",
        "generator_threshold_selection_used_human_gold": False,
        "interpretation_boundary": [
            "45 页 AI 辅助和 5 页盲标是历史计划分配，未执行真人复核。",
            "calibration 字段只记录固定规则在 8 页既有 Human Gold 上的诊断性回放。",
            "该回放没有选择、更新或冻结生成器阈值。",
            "本清单不包含人工研究结果，不能估计锚定效应或人工修改幅度。",
        ],
    })
    _write_json(path, historical)
    aggregate = calibration.get("aggregate", {})
    return {
        "path": _relative(path, repo_root),
        "sha256": _sha256(path),
        "scope": "fixed_rule_diagnostic_replay",
        "reported_structure_metrics": {
            "group_pair_f1": aggregate.get("group_pair_f1"),
            "group_match_f1": aggregate.get("group_match_f1"),
            "parent_f1": aggregate.get("parent_f1"),
        },
        "token_metrics_interpretable": False,
    }


def prepare(
    repo_root: Path,
    package_dir: Path,
    data_dir: Path,
) -> dict[str, Any]:
    assignment_path = package_dir / "assignment.json"
    assignment = _load_json(assignment_path)
    human_ids = sorted(str(value) for value in assignment["locked_sample_ids"])
    all_ids = [str(item["sample_id"]) for item in assignment["samples"]]
    ai_ids = sorted(set(all_ids) - set(human_ids))
    if len(human_ids) != 10 or len(ai_ids) != 50:
        raise ValueError("正式参考集必须保持 10 页设计师裁决 Gold + 50 页 AI")
    split_by_id = {
        str(item["sample_id"]): str(item["split"])
        for item in assignment["samples"]
    }
    non_replayed_human_ids = sorted(
        sample_id for sample_id in human_ids
        if split_by_id[sample_id] == "test"
    )
    diagnostic_replay_human_ids = sorted(
        set(human_ids) - set(non_replayed_human_ids)
    )
    if len(diagnostic_replay_human_ids) != 8 or len(non_replayed_human_ids) != 2:
        raise ValueError("Human Gold 诊断性回放/未回放集合必须保持 8/2")

    manifest_path = package_dir / "ai_reference_manifest.json"
    diagnostic_replay_artifact = _prepare_diagnostic_replay_artifact(
        package_dir / "ai_assistance_manifest.json",
        diagnostic_replay_human_ids,
        manifest_path,
        repo_root,
    )

    review_dir = package_dir / "ai_visual_reviews"
    reviews, review_batches = _load_visual_reviews(review_dir)
    missing_reviews = sorted(set(ai_ids) - set(reviews))
    extra_reviews = sorted(set(reviews) - set(ai_ids))
    if missing_reviews:
        raise ValueError(
            f"视觉审阅集合不完整：missing={missing_reviews}, extra={extra_reviews}"
        )

    reference_dir = package_dir / "reference"
    token_dir = package_dir / "ai_token_candidates"
    expanded_dir = package_dir / "annotator_ai_structure_expanded"
    structure_dir = package_dir / "annotator_ai_prelabel"
    reference_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)
    expanded_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_json_files(reference_dir, set(all_ids))
    _remove_stale_json_files(token_dir, set(ai_ids))
    _remove_stale_json_files(expanded_dir, set(ai_ids))
    _remove_stale_json_files(structure_dir, set(ai_ids))

    human_reports: list[dict[str, Any]] = []
    for sample_id in human_ids:
        source = data_dir / sample_id / "gold_intent.json"
        annotation = DesignIntentIR.load(source)
        if annotation.provenance.get("label_source") != "human_ai_adjudicated_gold":
            raise ValueError(f"{sample_id} 不是既有人工复核 Gold")
        reviewer_public_name_removed = bool(
            annotation.provenance.pop("reviewed_by", None)
        )
        annotation.provenance.update({
            "reference_tier": "human_gold",
            "human_reviewer_count": 1,
            "review_process": "single_human_ai_assisted_finalization",
            "reviewer_id": "human_reviewer_01",
            "reviewer_independent_of_base_annotator": False,
            "reviewer_qualification_documented": False,
            "reviewer_relationship_to_author": "not_documented",
            "reviewer_conflict_of_interest_status": "not_documented",
            "reviewer_public_name_removed": reviewer_public_name_removed,
            "public_name_consent_status": "not_documented",
            "review_notes_detail": (
                "brief_summary_only"
                if sample_id in BRIEF_HUMAN_REVIEW_NOTE_SAMPLE_IDS
                else "specific_change_summary"
            ),
            "token_labels_available": False,
            "token_review": {
                "status": "not_collected",
                "positive_labels_available": False,
                "explicit_negative_labels_available": False,
            },
            "reference_packaged_at": PREPARED_AT,
        })
        reference_path = reference_dir / f"{sample_id}.json"
        _write_json(reference_path, annotation.to_dict())
        human_reports.append({
            "sample_id": sample_id,
            "source_path": _relative(source, repo_root),
            "source_sha256": _sha256(source),
            "reference_path": _relative(reference_path, repo_root),
            "reference_sha256": _sha256(reference_path),
            "token_labels_available": False,
        })

    sample_reports: list[dict[str, Any]] = []
    for sample_id in ai_ids:
        graph_path = data_dir / sample_id / "page_graph.json"
        screenshot_path = data_dir / sample_id / "screenshot.png"
        graph = PageGraph.load(graph_path)
        structure_path = structure_dir / f"{sample_id}.json"
        expected_structure = build_ai_prelabel(graph)
        expected_structure.provenance["page_graph_sha256"] = _sha256(graph_path)
        expected_structure.provenance["screenshot_sha256"] = _sha256(
            screenshot_path
        )
        structure: DesignIntentIR | None = None
        if structure_path.exists():
            try:
                existing = DesignIntentIR.load(structure_path)
            except (KeyError, TypeError, ValueError):
                existing = None
            if existing is not None and _source_candidate_is_current(
                existing,
                sample_id,
                graph_path,
                screenshot_path,
                expected=expected_structure,
            ):
                structure = existing
        if structure is None:
            _write_json(structure_path, expected_structure.to_dict())
            structure = expected_structure

        expanded_path = expanded_dir / f"{sample_id}.json"
        expanded = build_expanded_structure_candidate(graph)
        expanded.provenance["page_graph_sha256"] = _sha256(graph_path)
        _write_json(expanded_path, expanded.to_dict())

        preferred = reviews[sample_id].get(
            "preferred_structure_candidate", "conservative"
        )
        selected_structure = expanded if preferred == "expanded" else structure
        selected_structure_path = (
            expanded_path if preferred == "expanded" else structure_path
        )

        token_preview = build_corrected_token_preview(
            selected_structure, reviews[sample_id]
        )
        token_path = token_dir / f"{sample_id}.json"
        _write_json(
            token_path,
            _token_candidate_payload(
                sample_id,
                token_preview,
                _sha256(selected_structure_path),
                _sha256(Path(reviews[sample_id]["review_path"])),
            ),
        )
        annotation = _build_ai_reference(
            sample_id,
            structure,
            structure_path,
            expanded,
            expanded_path,
            reviews[sample_id],
            token_path,
            screenshot_path,
            graph_path,
            diagnostic_replay_human_ids,
            diagnostic_replay_artifact,
        )
        errors = AnnotationStore._submission_errors(annotation, graph)
        if errors:
            raise ValueError(f"{sample_id} 参考标注校验失败：{errors}")
        reference_path = reference_dir / f"{sample_id}.json"
        _write_json(reference_path, annotation.to_dict())
        sample_reports.append({
            "sample_id": sample_id,
            "quality_status": reviews[sample_id]["quality_status"],
            "visual_decision": reviews[sample_id]["decision"],
            "structure_alignment": reviews[sample_id]["structure_alignment"],
            "hierarchy_risk": reviews[sample_id]["hierarchy_risk"],
            "selected_structure_candidate": preferred,
            "reference_path": _relative(reference_path),
            "reference_sha256": _sha256(reference_path),
            "elements": len(annotation.elements),
            "groups": len(annotation.groups),
            "style_tokens": len(annotation.style_tokens),
            "validation_errors": [],
        })

    human_set = set(human_ids)
    for item in assignment["samples"]:
        if item["sample_id"] in human_set:
            item["status"] = "existing_human_adjudicated_gold"
            item["annotation_condition"] = "existing_human_adjudicated_gold"
        else:
            item["status"] = "ai_reference_complete"
            item["annotation_condition"] = "ai_multiview_silver"

    assignment.update({
        "purpose": "design_intent_reference_v1",
        "annotators": ["reference"],
        "human_gold_sample_ids": human_ids,
        "ai_silver_sample_ids": ai_ids,
        "read_only_sample_ids": sorted(all_ids),
        "token_annotation_mode": "automatic_reference",
        "annotation_workflow": {
            "mode": "ai_multiview_reference_generation",
            "generator": GENERATOR,
            "prepared_at": PREPARED_AT,
            "canonical_reference_dir": _relative(reference_dir, repo_root),
            "structure_candidate_dir": _relative(
                package_dir / "annotator_ai_prelabel", repo_root
            ),
            "visual_review_dir": _relative(review_dir, repo_root),
            "token_candidate_dir": _relative(token_dir, repo_root),
            "expanded_structure_candidate_dir": _relative(
                expanded_dir, repo_root
            ),
            "ai_silver_sample_count": len(ai_ids),
            "human_gold_sample_count": len(human_ids),
            "diagnostic_replay_human_gold_sample_ids": (
                diagnostic_replay_human_ids
            ),
            "non_replayed_human_gold_sample_ids": non_replayed_human_ids,
            "generator_threshold_selection_used_human_gold": False,
            "developer_human_gold_blinding_status": "not_guaranteed",
            "rule_teacher_candidate_used": True,
            "human_gold_token_labels_available": False,
            "human_gold_token_metrics_reportable": False,
            "require_human_review_confirmation": False,
            "ai_labels_treated_as_human": False,
        },
        "human_gold_review_provenance": {
            "human_reviewer_count": 1,
            "review_process": "single_human_ai_assisted_finalization",
            "reviewer_id": "human_reviewer_01",
            "reviewer_independent_of_base_annotator": False,
            "reviewer_qualification_documented": False,
            "reviewer_relationship_to_author": "not_documented",
            "reviewer_conflict_of_interest_status": "not_documented",
            "public_name_consent_status": "not_documented",
            "brief_review_notes_samples": len(
                BRIEF_HUMAN_REVIEW_NOTE_SAMPLE_IDS
            ),
        },
    })
    original_weak_labels_visible = assignment.pop("weak_labels_visible", False)
    assignment["human_gold_original_assignment_context"] = {
        "weak_labels_visible": original_weak_labels_visible,
        "scope": "original_human_annotation_only",
    }
    _write_json(assignment_path, assignment)

    selection_path = package_dir / "selection_manifest.json"
    selection = _load_json(selection_path)
    for item in selection["samples"]:
        sample_id = str(item["sample_id"])
        if "selection_status" not in item:
            item["selection_status"] = item["status"]
        item["status"] = (
            "existing_human_adjudicated_gold"
            if sample_id in human_set
            else "ai_reference_complete"
        )
    selection.update({
        "annotation_required_ids": [],
        "reference_complete_ids": sorted(all_ids),
        "human_gold_sample_ids": human_ids,
        "ai_silver_sample_ids": ai_ids,
    })
    _write_json(selection_path, selection)

    manifest = {
        "schema_version": "1.0",
        "purpose": "ai_multiview_silver_annotation",
        "prepared_at": PREPARED_AT,
        "generator": GENERATOR,
        "canonical_reference_dir": _relative(reference_dir, repo_root),
        "ai_silver_samples": len(ai_ids),
        "human_gold_samples": len(human_ids),
        "diagnostic_replay_human_gold_sample_ids": (
            diagnostic_replay_human_ids
        ),
        "non_replayed_human_gold_sample_ids": non_replayed_human_ids,
        "generator_threshold_selection_used_human_gold": False,
        "developer_human_gold_blinding_status": "not_guaranteed",
        "diagnostic_replay_artifact": diagnostic_replay_artifact,
        "human_gold_token_labels_available": False,
        "human_gold_token_metrics_reportable": False,
        "human_gold_review_provenance": {
            "human_reviewer_count": 1,
            "review_process": "single_human_ai_assisted_finalization",
            "reviewer_id": "human_reviewer_01",
            "reviewer_independent_of_base_annotator": False,
            "reviewer_qualification_documented": False,
            "reviewer_relationship_to_author": "not_documented",
            "reviewer_conflict_of_interest_status": "not_documented",
            "public_name_consent_status": "not_documented",
            "brief_review_notes_samples": len(
                BRIEF_HUMAN_REVIEW_NOTE_SAMPLE_IDS
            ),
        },
        "human_review_performed": False,
        "ai_labels_treated_as_human": False,
        "visual_review_batches": review_batches,
        "visual_review_reproducibility": {
            "supported_scope": "frozen_artifact_replay",
            "end_to_end_regeneration_supported": False,
            "model_version_recorded": False,
            "prompt_hash_recorded": False,
            "sampling_parameters_recorded": False,
            "run_session_identifier_recorded": False,
            "limitation": (
                "冻结 JSON 之后的确定性组装可复现；当时的 AI 视觉判断会话"
                "缺少模型版本、提示词哈希和采样参数，不能端到端重生成。"
            ),
        },
        "excluded_visual_review_sample_ids": extra_reviews,
        "token_policy": {
            "generator": TOKEN_GENERATOR,
            "minimum_members": TOKEN_MIN_MEMBERS,
            "kind_caps": TOKEN_KIND_CAPS,
            "maximum_tokens_per_page": sum(TOKEN_KIND_CAPS.values()),
            "evaluation_scope": "ai_silver_proxy_consistency_only",
        },
        "human_gold_files": human_reports,
        "samples": sample_reports,
        "interpretation_boundary": [
            "50 页是 AI 多视角参考标注，不得表述为人工 Gold。",
            "10 页由同一名人工标注者以人工底稿结合 AI 差异提示后定稿；资历与作者关系未归档，不是独立裁决或双人盲标。",
            "混合测试集指标必须按标签来源分层报告。",
            "AI 视觉审阅不是第二份完整独立 IR，不能据此报告标注者间一致性。",
            "Human Gold 未提供可审计的 Token 正例或显式负例，不报告其 Token F1。",
            "AI Silver Token 只衡量与 computed-style 自动聚类代理的一致性。",
            "8 页 Human Gold 仅用于固定规则的诊断性回放，不是阈值选择集。",
        ],
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package_dir", default="data/annotations/intent_gold_v1"
    )
    parser.add_argument("--data_dir", default="data/processed")
    args = parser.parse_args()
    manifest = prepare(
        REPO_ROOT,
        (REPO_ROOT / args.package_dir).resolve(),
        (REPO_ROOT / args.data_dir).resolve(),
    )
    print(json.dumps({
        "ai_silver_samples": manifest["ai_silver_samples"],
        "human_gold_samples": manifest["human_gold_samples"],
        "human_review_performed": manifest["human_review_performed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
