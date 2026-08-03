"""为已完成的人机标注对生成可审计的仲裁记录和金标准草稿。"""

from __future__ import annotations

import argparse
import copy
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
from src.design_intent.schema import BBox, DesignIntentIR, PageGraph


ROOT_ID = "page_root"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_signature(ir: DesignIntentIR, entity_id: str) -> tuple[int, ...]:
    elements = {element.id: element for element in ir.elements}
    if entity_id in elements:
        return tuple(sorted(elements[entity_id].source_node_ids))

    groups = {group.id: group for group in ir.groups}
    if entity_id in groups:
        return tuple(
            sorted(
                {
                    node_id
                    for element_id in groups[entity_id].source_element_ids
                    for node_id in elements[element_id].source_node_ids
                }
            )
        )
    return ()


def entity_kind(ir: DesignIntentIR, entity_id: str) -> str:
    if entity_id == ROOT_ID:
        return "root"
    if any(element.id == entity_id for element in ir.elements):
        return "element"
    if any(group.id == entity_id for group in ir.groups):
        return "group"
    raise KeyError(f"未知实体：{entity_id}")


def entity_key(ir: DesignIntentIR, entity_id: str) -> tuple[str, tuple[int, ...]]:
    kind = entity_kind(ir, entity_id)
    return kind, () if kind == "root" else source_signature(ir, entity_id)


def key_payload(key: tuple[str, tuple[int, ...]]) -> dict[str, Any]:
    return {"kind": key[0], "source_node_ids": list(key[1])}


def bbox_payload(bbox: BBox) -> dict[str, float]:
    return {
        "x": bbox.x,
        "y": bbox.y,
        "width": bbox.width,
        "height": bbox.height,
    }


def bbox_iou(left: BBox, right: BBox) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.area + right.area - intersection
    return intersection / union if union > 0 else 0.0


def element_payload(element: Any) -> dict[str, Any]:
    return {
        "id": element.id,
        "type": element.type,
        "name": element.name,
        "text": element.text[:120],
        "bbox": bbox_payload(element.bbox),
    }


def group_payload(group: Any) -> dict[str, Any]:
    return {
        "id": group.id,
        "role": group.role,
        "name": group.name,
        "bbox": bbox_payload(group.bbox),
    }


def element_differences(
    human: DesignIntentIR, ai: DesignIntentIR
) -> dict[str, Any]:
    human_by_signature = {
        tuple(sorted(element.source_node_ids)): element for element in human.elements
    }
    ai_by_signature = {
        tuple(sorted(element.source_node_ids)): element for element in ai.elements
    }
    common = sorted(set(human_by_signature) & set(ai_by_signature))
    changed = []
    for signature in common:
        human_element = human_by_signature[signature]
        ai_element = ai_by_signature[signature]
        iou = bbox_iou(human_element.bbox, ai_element.bbox)
        if human_element.type != ai_element.type or iou < 0.95:
            changed.append(
                {
                    "source_node_ids": list(signature),
                    "type_disagrees": human_element.type != ai_element.type,
                    "bbox_iou": iou,
                    "human": element_payload(human_element),
                    "ai": element_payload(ai_element),
                }
            )
    return {
        "only_human": [
            {
                "source_node_ids": list(signature),
                "human": element_payload(human_by_signature[signature]),
            }
            for signature in sorted(set(human_by_signature) - set(ai_by_signature))
        ],
        "only_ai": [
            {
                "source_node_ids": list(signature),
                "ai": element_payload(ai_by_signature[signature]),
            }
            for signature in sorted(set(ai_by_signature) - set(human_by_signature))
        ],
        "matched_but_changed": changed,
    }


def groups_by_signature(ir: DesignIntentIR) -> dict[tuple[int, ...], list[Any]]:
    result: dict[tuple[int, ...], list[Any]] = {}
    for group in ir.groups:
        result.setdefault(source_signature(ir, group.id), []).append(group)
    for groups in result.values():
        groups.sort(key=lambda group: (group.role, group.name, group.id))
    return result


def group_differences(human: DesignIntentIR, ai: DesignIntentIR) -> list[dict[str, Any]]:
    human_groups = groups_by_signature(human)
    ai_groups = groups_by_signature(ai)
    result = []
    for signature in sorted(set(human_groups) | set(ai_groups)):
        left = human_groups.get(signature, [])
        right = ai_groups.get(signature, [])
        left_roles = [group.role for group in left]
        right_roles = [group.role for group in right]
        if left_roles == right_roles:
            continue
        result.append(
            {
                "source_node_ids": list(signature),
                "human": [group_payload(group) for group in left],
                "ai": [group_payload(group) for group in right],
            }
        )
    return result


def parent_map(
    ir: DesignIntentIR,
) -> dict[tuple[str, tuple[int, ...]], list[tuple[str, tuple[int, ...]]]]:
    result: dict[
        tuple[str, tuple[int, ...]], list[tuple[str, tuple[int, ...]]]
    ] = {}
    for edge in ir.tree:
        child = entity_key(ir, edge.child_id)
        parent = entity_key(ir, edge.parent_id)
        result.setdefault(child, []).append(parent)
    for parents in result.values():
        parents.sort()
    return result


def tree_differences(human: DesignIntentIR, ai: DesignIntentIR) -> list[dict[str, Any]]:
    human_parents = parent_map(human)
    ai_parents = parent_map(ai)
    result = []
    for child in sorted(set(human_parents) | set(ai_parents)):
        left = human_parents.get(child, [])
        right = ai_parents.get(child, [])
        if left == right:
            continue
        result.append(
            {
                "child": key_payload(child),
                "human_parents": [key_payload(parent) for parent in left],
                "ai_parents": [key_payload(parent) for parent in right],
            }
        )
    return result


def layouts_by_signature(ir: DesignIntentIR) -> dict[tuple[int, ...], list[Any]]:
    groups = {group.id for group in ir.groups}
    result: dict[tuple[int, ...], list[Any]] = {}
    for layout in ir.layouts:
        if layout.target_id in groups:
            result.setdefault(source_signature(ir, layout.target_id), []).append(layout)
    for layouts in result.values():
        layouts.sort(key=lambda layout: layout.target_id)
    return result


def layout_payload(layout: Any) -> dict[str, Any]:
    return {
        "target_id": layout.target_id,
        "mode": layout.mode,
        "gap": layout.gap,
        "padding": layout.padding,
        "primary_align": layout.primary_align,
        "cross_align": layout.cross_align,
        "horizontal_resize": layout.horizontal_resize,
        "vertical_resize": layout.vertical_resize,
    }


def layout_differences(human: DesignIntentIR, ai: DesignIntentIR) -> list[dict[str, Any]]:
    human_layouts = layouts_by_signature(human)
    ai_layouts = layouts_by_signature(ai)
    result = []
    for signature in sorted(set(human_layouts) | set(ai_layouts)):
        left = [layout_payload(layout) for layout in human_layouts.get(signature, [])]
        right = [layout_payload(layout) for layout in ai_layouts.get(signature, [])]
        left_values = [
            {key: value for key, value in item.items() if key != "target_id"}
            for item in left
        ]
        right_values = [
            {key: value for key, value in item.items() if key != "target_id"}
            for item in right
        ]
        if left_values == right_values:
            continue
        result.append(
            {
                "source_node_ids": list(signature),
                "human": left,
                "ai": right,
            }
        )
    return result


def token_key(ir: DesignIntentIR, token: Any) -> tuple[str, tuple[Any, ...]]:
    members = tuple(sorted(entity_key(ir, member_id) for member_id in token.member_ids))
    return token.kind, members


def token_key_payload(key: tuple[str, tuple[Any, ...]]) -> dict[str, Any]:
    return {
        "kind": key[0],
        "members": [key_payload(member) for member in key[1]],
    }


def token_differences(human: DesignIntentIR, ai: DesignIntentIR) -> dict[str, Any]:
    human_keys = {token_key(human, token) for token in human.style_tokens}
    ai_keys = {token_key(ai, token) for token in ai.style_tokens}
    return {
        "only_human": [token_key_payload(key) for key in sorted(human_keys - ai_keys)],
        "only_ai": [token_key_payload(key) for key in sorted(ai_keys - human_keys)],
    }


def count_differences(differences: dict[str, Any]) -> dict[str, int]:
    elements = differences["elements"]
    tokens = differences["tokens"]
    return {
        "elements_only_human": len(elements["only_human"]),
        "elements_only_ai": len(elements["only_ai"]),
        "elements_matched_but_changed": len(elements["matched_but_changed"]),
        "group_signature_disagreements": len(differences["groups"]),
        "tree_parent_disagreements": len(differences["tree"]),
        "layout_disagreements": len(differences["layouts"]),
        "tokens_only_human": len(tokens["only_human"]),
        "tokens_only_ai": len(tokens["only_ai"]),
    }


def build_record(
    sample_id: str,
    graph: PageGraph,
    human: DesignIntentIR,
    ai: DesignIntentIR,
    human_path: Path,
    ai_path: Path,
    package_dir: Path,
) -> dict[str, Any]:
    differences = {
        "elements": element_differences(human, ai),
        "groups": group_differences(human, ai),
        "tree": tree_differences(human, ai),
        "layouts": layout_differences(human, ai),
        "tokens": token_differences(human, ai),
    }
    metrics = compute_intent_metrics(ai, human, graph)
    metrics["layout_cohen_kappa"] = cohen_kappa(ai, human)
    return {
        "schema_version": "1.0",
        "comparison_kind": "human-ai",
        "sample_id": sample_id,
        "status": "pending",
        "inputs": {
            "human": {
                "annotator": "annotator_a",
                "path": str(human_path.relative_to(package_dir.parent.parent.parent)),
                "sha256": sha256(human_path),
            },
            "ai": {
                "annotator": "annotator_ai",
                "path": str(ai_path.relative_to(package_dir.parent.parent.parent)),
                "sha256": sha256(ai_path),
                "must_not_be_reported_as_human": True,
            },
        },
        "metrics": metrics,
        "summary": count_differences(differences),
        "differences": differences,
        "review": {
            "reviewed_by": None,
            "reviewed_at": None,
            "notes": "",
            "required_checks": [
                "逐项核对原子元素的合并、拆分与类型",
                "逐项核对语义分组及父子层级",
                "逐项核对布局模式、间距与内边距",
                "确认样式 Token 是否代表应联动修改的设计变量",
                "在金标准草稿中记录采用人工、采用 AI 或重新标注的决定",
            ],
        },
    }


def build_gold_draft(
    human: DesignIntentIR, record: dict[str, Any]
) -> dict[str, Any]:
    draft = copy.deepcopy(human.to_dict())
    original_provenance = copy.deepcopy(draft.get("provenance", {}))
    draft["provenance"] = {
        "source_sample_id": record["sample_id"],
        "label_source": "human_ai_adjudication_draft",
        "status": "draft",
        "adjudication_status": "pending",
        "gold_finalized": False,
        "base_annotation": "annotator_a",
        "base_annotation_provenance": original_provenance,
        "human_annotation_sha256": record["inputs"]["human"]["sha256"],
        "ai_annotation_sha256": record["inputs"]["ai"]["sha256"],
        "ai_used_for_disagreement_review": True,
        "ai_treated_as_human_annotator": False,
        "reviewed_by": None,
        "reviewed_at": None,
        "review_notes": "",
    }
    return draft


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare_adjudication(
    repo_root: Path,
    package_dir: Path,
    adjudication_dir: Path,
    gold_drafts_dir: Path,
) -> dict[str, Any]:
    assignment = json.loads(
        (package_dir / "assignment.json").read_text(encoding="utf-8")
    )
    result: dict[str, Any] = {
        "comparison_kind": "human-ai",
        "prepared": [],
        "existing": [],
        "incomplete_human": [],
        "invalid": [],
    }

    for item in assignment["samples"]:
        sample_id = item["sample_id"]
        human_path = package_dir / "annotator_a" / f"{sample_id}.json"
        ai_path = package_dir / "annotator_ai" / f"{sample_id}.json"
        human = DesignIntentIR.load(human_path)
        ai = DesignIntentIR.load(ai_path)
        if human.provenance.get("status") != "complete":
            result["incomplete_human"].append(sample_id)
            continue
        if ai.provenance.get("status") != "complete":
            result["invalid"].append(
                {"sample_id": sample_id, "errors": ["AI 代理标注未完成"]}
            )
            continue
        if human.provenance.get("label_source") != "human_annotation":
            result["invalid"].append(
                {"sample_id": sample_id, "errors": ["人工输入来源字段非法"]}
            )
            continue
        if (
            ai.provenance.get("label_source") != "ai_proxy_annotation"
            or ai.provenance.get("human_labels_viewed") is not False
        ):
            result["invalid"].append(
                {"sample_id": sample_id, "errors": ["AI 代理盲态来源字段非法"]}
            )
            continue

        graph_path = Path(item["page_graph"])
        if not graph_path.is_absolute():
            graph_path = repo_root / graph_path
        graph = PageGraph.load(graph_path)
        errors = AnnotationStore._submission_errors(human, graph)
        errors += AnnotationStore._submission_errors(ai, graph)
        if errors:
            result["invalid"].append(
                {"sample_id": sample_id, "errors": list(dict.fromkeys(errors))}
            )
            continue

        record_path = adjudication_dir / f"{sample_id}.json"
        gold_path = gold_drafts_dir / f"{sample_id}.json"
        if record_path.exists() or gold_path.exists():
            if not record_path.exists() or not gold_path.exists():
                raise RuntimeError(f"{sample_id} 的仲裁记录与金标准草稿不成对")
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            hashes = existing.get("inputs", {})
            if (
                hashes.get("human", {}).get("sha256") != sha256(human_path)
                or hashes.get("ai", {}).get("sha256") != sha256(ai_path)
            ):
                raise RuntimeError(
                    f"{sample_id} 的输入已变化；为避免覆盖人工审核，需另行处理"
                )
            result["existing"].append(sample_id)
            continue

        record = build_record(
            sample_id,
            graph,
            human,
            ai,
            human_path,
            ai_path,
            package_dir,
        )
        write_json(gold_path, build_gold_draft(human, record))
        write_json(record_path, record)
        result["prepared"].append(sample_id)

    result["assigned"] = len(assignment["samples"])
    result["ready_pairs"] = len(result["prepared"]) + len(result["existing"])
    result["complete"] = (
        result["ready_pairs"] == result["assigned"] and not result["invalid"]
    )
    result["gold_status"] = "draft_only"
    result["training_exported"] = False
    write_json(adjudication_dir / "index.json", result)
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
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    result = prepare_adjudication(
        repo_root,
        (repo_root / args.package_dir).resolve(),
        (repo_root / args.adjudication_dir).resolve(),
        (repo_root / args.gold_drafts_dir).resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["invalid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
