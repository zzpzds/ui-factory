"""为正式 Gold 包生成可审计的 AI 预标注，并安全初始化人工校正草稿。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotation.store import AnnotationStore
from src.design_intent.grouping import normalize_group_source_elements
from src.design_intent.metrics import compute_intent_metrics
from src.design_intent.schema import DesignIntentIR, PageGraph, TreeEdge
from src.design_intent.weak_supervision import build_weak_intent


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARED_AT = "2026-08-14T00:00:00+08:00"
GENERATOR = "codex_pagegraph_preannotation_v1"
ELEMENT_CONFIDENCE_MIN = 0.75
GROUP_CONFIDENCE_MIN = 0.85
BLIND_CONTROL_TARGETS = {"train": 3, "validation": 1, "test": 1}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_score(seed: int, sample_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()[:8], "big"
    )


def select_blind_controls(
    samples: list[dict[str, Any]], seed: int
) -> list[str]:
    """按 split 抽取 10% 盲标页，并优先覆盖不同页面类型与尺寸。"""
    selected: list[dict[str, Any]] = []
    used_types: set[str] = set()
    used_sizes: set[str] = set()
    for split, target in BLIND_CONTROL_TARGETS.items():
        candidates = [item for item in samples if item["split"] == split]
        for _ in range(target):
            remaining = [item for item in candidates if item not in selected]
            if not remaining:
                raise ValueError(f"{split} 没有足够样本用于盲标对照")
            remaining.sort(
                key=lambda item: (
                    item.get("page_type") in used_types,
                    item.get("size_bin") in used_sizes,
                    not bool(item.get("token_rich_candidate")),
                    _stable_score(seed, item["sample_id"]),
                )
            )
            chosen = remaining[0]
            selected.append(chosen)
            used_types.add(str(chosen.get("page_type")))
            used_sizes.add(str(chosen.get("size_bin")))
    return sorted(item["sample_id"] for item in selected)


def _ancestors(node_id: int, graph: PageGraph) -> list[int]:
    node_by_id = {node.id: node for node in graph.nodes}
    result: list[int] = []
    current = node_by_id[node_id].parent_id
    seen: set[int] = set()
    while current is not None and current in node_by_id and current not in seen:
        result.append(current)
        seen.add(current)
        current = node_by_id[current].parent_id
    return result


def build_ai_prelabel(graph: PageGraph) -> DesignIntentIR:
    """保留高置信元素/语义容器，重建层级，并把 Token 留给人工候选审核。"""
    draft = build_weak_intent(graph)
    draft.elements = [
        element
        for element in draft.elements
        if element.confidence >= ELEMENT_CONFIDENCE_MIN
    ]
    retained_element_ids = {element.id for element in draft.elements}
    draft.groups = [
        group
        for group in draft.groups
        if group.confidence >= GROUP_CONFIDENCE_MIN
        and retained_element_ids.intersection(group.source_element_ids)
    ]
    retained_group_ids = {group.id for group in draft.groups}
    draft.layouts = [
        layout
        for layout in draft.layouts
        if layout.target_id in retained_group_ids
    ]
    draft.style_tokens = []
    for element in draft.elements:
        element.style_token_refs = []

    def rebuild_tree() -> dict[str, list[TreeEdge]]:
        group_by_node = {
            group.source_node_id: group.id
            for group in draft.groups
            if group.source_node_id is not None
        }
        boxes = {
            entity.id: entity.bbox
            for entity in [*draft.elements, *draft.groups]
        }
        tree: list[TreeEdge] = []
        for group in draft.groups:
            parent_id = "page_root"
            if group.source_node_id is not None:
                for ancestor_id in _ancestors(group.source_node_id, graph):
                    if ancestor_id in group_by_node:
                        parent_id = group_by_node[ancestor_id]
                        break
            tree.append(TreeEdge(
                parent_id, group.id, 0, group.confidence
            ))
        for element in draft.elements:
            parent_id = "page_root"
            for ancestor_id in _ancestors(element.source_node_ids[0], graph):
                if ancestor_id in group_by_node:
                    parent_id = group_by_node[ancestor_id]
                    break
            tree.append(TreeEdge(
                parent_id, element.id, 0, element.confidence
            ))
        children_by_parent: dict[str, list[TreeEdge]] = defaultdict(list)
        for edge in tree:
            children_by_parent[edge.parent_id].append(edge)
        for edges in children_by_parent.values():
            edges.sort(key=lambda edge: (
                boxes[edge.child_id].y,
                boxes[edge.child_id].x,
                edge.child_id,
            ))
            for order, edge in enumerate(edges):
                edge.order = order
        draft.tree = tree
        return children_by_parent

    # 单一原子 wrapper 不构成设计分组；删除后重新寻找最近的保留父组。
    while True:
        children_by_parent = rebuild_tree()
        group_ids = {group.id for group in draft.groups}
        invalid_group_ids = {
            group.id
            for group in draft.groups
            if not children_by_parent.get(group.id)
            or (
                len(children_by_parent[group.id]) == 1
                and children_by_parent[group.id][0].child_id not in group_ids
            )
        }
        if not invalid_group_ids:
            break
        draft.groups = [
            group for group in draft.groups if group.id not in invalid_group_ids
        ]
        draft.layouts = [
            layout
            for layout in draft.layouts
            if layout.target_id not in invalid_group_ids
        ]
    normalize_group_source_elements(draft)
    draft.provenance = {
        "source_sample_id": graph.sample_id,
        "label_source": "ai_preannotation",
        "annotator": "codex_ai_prelabel",
        "annotator_kind": "ai_preannotation",
        "generator": GENERATOR,
        "generator_version": "1.0",
        "status": "complete",
        "human_labels_viewed": False,
        "visual_input_used": False,
        "page_graph_used": True,
        "generated_at": PREPARED_AT,
        "element_confidence_min": ELEMENT_CONFIDENCE_MIN,
        "group_confidence_min": GROUP_CONFIDENCE_MIN,
        "limitations": [
            "该文件是 AI 辅助初稿，不是人工标注或最终 Gold。",
            "层级和语义分组必须由设计师逐项复核。",
            "Token 不直接预填，由人工候选审核产生。",
        ],
    }
    return draft


def _has_manual_work(annotation: DesignIntentIR) -> bool:
    return any(
        (
            annotation.elements,
            annotation.groups,
            annotation.layouts,
            annotation.tree,
            annotation.style_tokens,
            annotation.provenance.get("token_review"),
        )
    )


def _human_assisted_draft(
    ai: DesignIntentIR, ai_path: Path, package_dir: Path
) -> DesignIntentIR:
    draft = copy.deepcopy(ai)
    relative_ai_path = ai_path.relative_to(REPO_ROOT).as_posix()
    draft.provenance = {
        "source_sample_id": ai.provenance["source_sample_id"],
        "label_source": "ai_assisted_annotation_draft",
        "annotator": "annotator_a",
        "status": "draft",
        "annotation_phase": "formal_gold_ai_assisted",
        "ai_assistance_mode": "preannotation",
        "ai_assistance_disclosed": True,
        "ai_preannotation_visible": True,
        "ai_preannotation_path": relative_ai_path,
        "ai_preannotation_sha256": _sha256(ai_path),
        "ai_preannotation_generator": GENERATOR,
        "weak_labels_viewed": True,
        "human_review_confirmed": False,
        "human_reviewed_at": None,
        "token_review": {
            "status": "pending",
            "accepted_candidate_keys": [],
            "ignored_candidate_keys": [],
            "reviewed_candidate_keys": [],
        },
    }
    return draft


def _blind_control_draft(
    existing: DesignIntentIR, ai_path: Path
) -> DesignIntentIR:
    draft = copy.deepcopy(existing)
    draft.provenance = {
        **draft.provenance,
        "label_source": "human_annotation",
        "status": "draft",
        "annotation_phase": "formal_gold_blind_control",
        "ai_assistance_mode": "blind_control",
        "ai_assistance_disclosed": True,
        "ai_preannotation_visible": False,
        "ai_preannotation_path": ai_path.relative_to(REPO_ROOT).as_posix(),
        "ai_preannotation_sha256": _sha256(ai_path),
        "weak_labels_viewed": False,
        "human_review_confirmed": False,
        "human_reviewed_at": None,
    }
    return draft


def calibration_report(
    assignment: dict[str, Any], data_dir: Path
) -> dict[str, Any]:
    """仅用既有 train/validation Gold 记录预标注回放表现，不读取 test Gold。"""
    locked = set(assignment.get("locked_sample_ids", []))
    totals: dict[str, list[float]] = defaultdict(list)
    per_sample: dict[str, Any] = {}
    for item in assignment["samples"]:
        if item["sample_id"] not in locked or item.get("split") == "test":
            continue
        graph = PageGraph.load(data_dir / item["sample_id"] / "page_graph.json")
        gold = DesignIntentIR.load(data_dir / item["sample_id"] / "gold_intent.json")
        predicted = build_ai_prelabel(graph)
        metrics = compute_intent_metrics(predicted, gold, graph)
        per_sample[item["sample_id"]] = metrics
        for name, value in metrics.items():
            totals[name].append(value)
    return {
        "scope": "existing_locked_train_validation_only",
        "test_gold_used": False,
        "sample_count": len(per_sample),
        "aggregate": {
            name: sum(values) / len(values) for name, values in totals.items()
        },
        "per_sample": per_sample,
    }


def prepare(
    repo_root: Path, package_dir: Path, data_dir: Path
) -> dict[str, Any]:
    assignment_path = package_dir / "assignment.json"
    assignment = _load_json(assignment_path)
    locked = set(assignment.get("locked_sample_ids", []))
    pending = [
        item for item in assignment["samples"] if item["sample_id"] not in locked
    ]
    existing_controls = assignment.get("annotation_workflow", {}).get(
        "blind_control_sample_ids"
    )
    blind_controls = sorted(existing_controls or select_blind_controls(
        pending, int(assignment.get("seed", 0)) + 17
    ))
    blind_set = set(blind_controls)
    ai_dir = package_dir / "annotator_ai_prelabel"
    human_dir = package_dir / "annotator_a"
    ai_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[str] = []
    for item in pending:
        sample_id = item["sample_id"]
        human_path = human_dir / f"{sample_id}.json"
        existing_human = DesignIntentIR.load(human_path)
        if (
            _has_manual_work(existing_human)
            and not existing_human.provenance.get("ai_assistance_mode")
        ):
            raise ValueError(f"{sample_id} 已有人工内容，拒绝 AI 预填覆盖")

        graph_path = data_dir / sample_id / "page_graph.json"
        screenshot_path = data_dir / sample_id / "screenshot.png"
        graph = PageGraph.load(graph_path)
        ai_path = ai_dir / f"{sample_id}.json"
        if ai_path.exists():
            ai = DesignIntentIR.load(ai_path)
            if ai.provenance.get("generator") != GENERATOR:
                raise ValueError(f"{sample_id} 已有不兼容的 AI 初稿，拒绝覆盖")
        else:
            ai = build_ai_prelabel(graph)
            ai.provenance["page_graph_sha256"] = _sha256(graph_path)
            ai.provenance["screenshot_sha256"] = _sha256(screenshot_path)
            ai.dump(ai_path)

        if not existing_human.provenance.get("ai_assistance_mode"):
            draft = (
                _blind_control_draft(existing_human, ai_path)
                if sample_id in blind_set
                else _human_assisted_draft(ai, ai_path, package_dir)
            )
            draft.dump(human_path)
        item["annotation_condition"] = (
            "blind_control" if sample_id in blind_set else "ai_assisted"
        )
        prepared.append(sample_id)

    for item in assignment["samples"]:
        if item["sample_id"] in locked:
            item["annotation_condition"] = "existing_independent_gold"
    assignment["annotation_workflow"] = {
        "mode": "ai_preannotation_then_human_correction",
        "generator": GENERATOR,
        "prepared_at": PREPARED_AT,
        "ai_prelabel_dir": ai_dir.relative_to(repo_root).as_posix(),
        "assisted_sample_count": len(pending) - len(blind_controls),
        "blind_control_sample_count": len(blind_controls),
        "blind_control_sample_ids": blind_controls,
        "require_human_review_confirmation": True,
        "test_labels_not_used_for_diagnostic_replay": True,
        "generator_threshold_selection_used_human_gold": False,
    }
    _write_json(assignment_path, assignment)

    manifest = {
        "schema_version": "1.0",
        "purpose": "ai_assisted_formal_gold_annotation",
        "prepared_at": PREPARED_AT,
        "generator": GENERATOR,
        "inputs": ["page_graph"],
        "visual_input_used_for_generation": False,
        "pending_samples": len(pending),
        "ai_prelabels_generated": len(prepared),
        "human_assisted_samples_planned": len(pending) - len(blind_controls),
        "blind_control_sample_ids_planned": blind_controls,
        "human_reviews_completed": 0,
        "blind_control_annotations_completed": 0,
        "human_study_results_available": False,
        "calibration": calibration_report(assignment, data_dir),
        "interpretation_boundary": [
            "45 页 AI 辅助和 5 页盲标是计划分配，未执行真人复核。",
            "calibration 字段是固定规则的诊断性回放，不是阈值选择记录。",
            "该计划没有产生锚定效应或人工修改幅度的研究结果。",
            "test Human Gold 未参与诊断性回放。",
        ],
        "status": "planned_not_executed",
    }
    _write_json(package_dir / "ai_assistance_manifest.json", manifest)
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
        "generated": manifest["ai_prelabels_generated"],
        "assisted_planned": manifest["human_assisted_samples_planned"],
        "blind_controls_planned": manifest["blind_control_sample_ids_planned"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
