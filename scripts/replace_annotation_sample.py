"""安全替换尚未开始标注的 Pilot 样本，并保留可审计记录。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.create_annotation_pilot import (
    empty_annotation,
    size_bin,
    structure_fingerprint,
)
from src.design_intent.schema import DesignIntentIR, PageGraph
from src.design_intent.validation import validate_page_graph


ANNOTATORS = ("annotator_a", "annotator_b")
AI_ANNOTATOR = "annotator_ai"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _is_empty_draft(ir: DesignIntentIR) -> bool:
    return (
        ir.provenance.get("status", "draft") == "draft"
        and not ir.elements
        and not ir.groups
        and not ir.layouts
        and not ir.tree
        and not ir.style_tokens
    )


def _duplicate_component(
    sample_id: str,
    duplicate_report: dict[str, Any],
) -> set[str]:
    adjacency: dict[str, set[str]] = {}
    for match in duplicate_report.get("matches", []):
        left, right = str(match["left"]), str(match["right"])
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    seen = {sample_id}
    pending = [sample_id]
    while pending:
        current = pending.pop()
        for neighbor in adjacency.get(current, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return seen


def replace_sample(
    package_dir: Path,
    data_dir: Path,
    old_sample_id: str,
    new_sample_id: str,
    reason: str,
    duplicate_report_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    data_dir = data_dir.resolve()
    assignment_path = package_dir / "assignment.json"
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    samples = assignment["samples"]
    index = next(
        (
            position
            for position, item in enumerate(samples)
            if item["sample_id"] == old_sample_id
        ),
        None,
    )
    if index is None:
        raise ValueError(f"assignment 中不存在样本 {old_sample_id}")
    if any(item["sample_id"] == new_sample_id for item in samples):
        raise ValueError(f"样本 {new_sample_id} 已在 assignment 中")

    old_annotations = {}
    for annotator in ANNOTATORS:
        path = package_dir / annotator / f"{old_sample_id}.json"
        ir = DesignIntentIR.load(path)
        if not _is_empty_draft(ir):
            raise ValueError(
                f"{annotator}/{old_sample_id} 已开始标注，禁止替换"
            )
        old_annotations[annotator] = path

    ai_annotation_path = package_dir / AI_ANNOTATOR / f"{old_sample_id}.json"
    ai_manifest_path = package_dir / AI_ANNOTATOR / "manifest.json"
    has_ai_track = ai_annotation_path.exists()
    new_ai_annotation_path = package_dir / AI_ANNOTATOR / f"{new_sample_id}.json"
    if has_ai_track and new_ai_annotation_path.exists():
        raise FileExistsError(f"新 AI 代理标注已存在：{new_ai_annotation_path}")

    graph_path = data_dir / new_sample_id / "page_graph.json"
    screenshot_path = data_dir / new_sample_id / "screenshot.png"
    if not graph_path.exists() or not screenshot_path.exists():
        raise FileNotFoundError(f"新样本 {new_sample_id} 缺少 PageGraph 或截图")
    graph = PageGraph.load(graph_path)
    graph_errors = validate_page_graph(graph)
    if graph_errors:
        raise ValueError(f"新样本 PageGraph 非法：{graph_errors}")
    if graph.sample_id != new_sample_id:
        raise ValueError("新样本目录名与 PageGraph sample_id 不一致")

    old_item = samples[index]
    new_size_bin = size_bin(len(graph.nodes))
    if new_size_bin != old_item["size_bin"]:
        raise ValueError(
            f"规模分层不一致：{old_item['size_bin']} -> {new_size_bin}"
        )
    new_fingerprint = structure_fingerprint(graph)
    retained = [
        item for item in samples if item["sample_id"] != old_sample_id
    ]
    if new_fingerprint in {
        item["structure_fingerprint"] for item in retained
    }:
        raise ValueError("新样本与保留样本的结构指纹重复")

    if duplicate_report_path is not None:
        duplicate_report = json.loads(
            duplicate_report_path.read_text(encoding="utf-8")
        )
        component = _duplicate_component(new_sample_id, duplicate_report)
        conflicts = sorted(
            component
            & {item["sample_id"] for item in retained}
        )
        if conflicts:
            raise ValueError(f"新样本与保留样本近重复：{conflicts}")

    new_item = {
        "sample_id": new_sample_id,
        "node_count": len(graph.nodes),
        "size_bin": new_size_bin,
        "structure_fingerprint": new_fingerprint,
        "screenshot": str(
            Path(os.path.relpath(screenshot_path, Path.cwd()))
        ),
        "page_graph": str(
            Path(os.path.relpath(graph_path, Path.cwd()))
        ),
    }
    event = {
        "old_sample_id": old_sample_id,
        "new_sample_id": new_sample_id,
        "reason": reason,
        "old_size_bin": old_item["size_bin"],
        "new_size_bin": new_size_bin,
        "old_node_count": old_item["node_count"],
        "new_node_count": len(graph.nodes),
    }
    result = {
        "dry_run": dry_run,
        "replacement": event,
        "new_sample": new_item,
    }
    if dry_run:
        return result

    archive_dir = (
        package_dir / "replaced" / f"{old_sample_id}_to_{new_sample_id}"
    )
    if archive_dir.exists():
        raise FileExistsError(f"归档目录已存在：{archive_dir}")
    new_annotation_paths = {
        annotator: package_dir / annotator / f"{new_sample_id}.json"
        for annotator in ANNOTATORS
    }
    existing_new_paths = [
        str(path) for path in new_annotation_paths.values() if path.exists()
    ]
    if existing_new_paths:
        raise FileExistsError(f"新标注模板已存在：{existing_new_paths}")
    archive_dir.mkdir(parents=True)

    for annotator in ANNOTATORS:
        new_path = new_annotation_paths[annotator]
        empty_annotation(graph, annotator).dump(new_path)
        annotator_archive = archive_dir / annotator
        annotator_archive.mkdir()
        os.replace(
            old_annotations[annotator],
            annotator_archive / f"{old_sample_id}.json",
        )

    if has_ai_track:
        ai_archive = archive_dir / AI_ANNOTATOR
        ai_archive.mkdir()
        os.replace(
            ai_annotation_path,
            ai_archive / f"{old_sample_id}.json",
        )
        if ai_manifest_path.exists():
            os.replace(ai_manifest_path, ai_archive / "manifest.json")

    samples[index] = new_item
    event["replaced_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    event["archive_dir"] = str(
        Path(os.path.relpath(archive_dir, Path.cwd()))
    )
    assignment.setdefault("replacement_history", []).append(event)
    ai_track = assignment.get("annotation_tracks", {}).get("ai_proxy")
    if has_ai_track and isinstance(ai_track, dict):
        ai_track["status"] = "replacement_pending"
        ai_track["pending_sample_id"] = new_sample_id
        ai_track["replaced_sample_id"] = old_sample_id
    _atomic_json(assignment_path, assignment)
    result["replacement"] = event
    result["ai_proxy_rebuild_required"] = has_ai_track
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package_dir", default="data/annotations/intent_pilot_v1"
    )
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--old_sample_id", required=True)
    parser.add_argument("--new_sample_id", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument(
        "--duplicate_report",
        default="outputs/intent-pilot-500/near-duplicate-report.json",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    result = replace_sample(
        package_dir=Path(args.package_dir),
        data_dir=Path(args.data_dir),
        old_sample_id=args.old_sample_id,
        new_sample_id=args.new_sample_id,
        reason=args.reason,
        duplicate_report_path=(
            Path(args.duplicate_report) if args.duplicate_report else None
        ),
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
