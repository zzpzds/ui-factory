"""按页面规模与结构多样性创建双人 Pilot 标注包。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.design_intent.schema import DesignIntentIR, PageGraph


def structure_fingerprint(graph: PageGraph) -> str:
    signature = "|".join(
        f"{node.tag}:{node.depth}:{min(node.child_count, 8)}"
        for node in graph.nodes
    )
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def size_bin(node_count: int) -> str:
    if node_count <= 50:
        return "small"
    if node_count <= 128:
        return "medium"
    return "large"


def empty_annotation(graph: PageGraph, annotator: str) -> DesignIntentIR:
    return DesignIntentIR(
        schema_version="1.0",
        canvas=graph.canvas,
        elements=[],
        groups=[],
        layouts=[],
        tree=[],
        style_tokens=[],
        provenance={
            "source_sample_id": graph.sample_id,
            "label_source": "human_annotation",
            "annotator": annotator,
            "status": "draft",
            "weak_labels_viewed": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument(
        "--pilot_manifest", default="data/intent_pilot_manifest.json"
    )
    parser.add_argument(
        "--output_dir", default="data/annotations/intent_pilot_v1"
    )
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--duplicate_report",
        default=None,
        help="近重复报告；提供后每个重复簇只保留最小 sample_id",
    )
    args = parser.parse_args()

    manifest = json.loads(
        Path(args.pilot_manifest).read_text(encoding="utf-8")
    )
    representatives: dict[str, str] = {}
    if args.duplicate_report:
        duplicate_report = json.loads(
            Path(args.duplicate_report).read_text(encoding="utf-8")
        )
        parent: dict[str, str] = {}

        def find(value: str) -> str:
            parent.setdefault(value, value)
            if parent[value] != value:
                parent[value] = find(parent[value])
            return parent[value]

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for match in duplicate_report["matches"]:
            union(match["left"], match["right"])
        clusters: dict[str, list[str]] = {}
        for sample_id in parent:
            clusters.setdefault(find(sample_id), []).append(sample_id)
        for members in clusters.values():
            representative = min(members)
            for sample_id in members:
                representatives[sample_id] = representative

    candidates = []
    for sample_id in manifest["sample_ids"]:
        if (
            sample_id in representatives
            and representatives[sample_id] != sample_id
        ):
            continue
        graph_path = Path(args.data_dir) / sample_id / "page_graph.json"
        if not graph_path.exists():
            continue
        graph = PageGraph.load(graph_path)
        candidates.append(
            {
                "sample_id": sample_id,
                "graph": graph,
                "node_count": len(graph.nodes),
                "size_bin": size_bin(len(graph.nodes)),
                "fingerprint": structure_fingerprint(graph),
            }
        )
    if len(candidates) < args.num_samples:
        raise ValueError("成功渲染样本不足以创建标注包。")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    targets = {
        "small": max(1, args.num_samples // 3),
        "medium": max(1, args.num_samples // 3),
        "large": max(1, args.num_samples // 3),
    }
    targets["medium"] += args.num_samples - sum(targets.values())
    selected = []
    used_fingerprints: set[str] = set()
    for bin_name in ("small", "medium", "large"):
        pool = [
            item
            for item in candidates
            if item["size_bin"] == bin_name
            and item["fingerprint"] not in used_fingerprints
        ]
        pool.sort(key=lambda item: item["node_count"])
        if len(pool) > targets[bin_name]:
            step = (len(pool) - 1) / max(targets[bin_name] - 1, 1)
            pool = [
                pool[round(index * step)]
                for index in range(targets[bin_name])
            ]
        for item in pool[: targets[bin_name]]:
            selected.append(item)
            used_fingerprints.add(item["fingerprint"])

    if len(selected) < args.num_samples:
        remaining = [
            item
            for item in candidates
            if item["sample_id"]
            not in {chosen["sample_id"] for chosen in selected}
            and item["fingerprint"] not in used_fingerprints
        ]
        selected.extend(remaining[: args.num_samples - len(selected)])
    selected = sorted(selected[: args.num_samples], key=lambda item: item["sample_id"])

    output_dir = Path(args.output_dir)
    for annotator in ("annotator_a", "annotator_b"):
        annotation_dir = output_dir / annotator
        annotation_dir.mkdir(parents=True, exist_ok=True)
        for item in selected:
            empty_annotation(item["graph"], annotator).dump(
                annotation_dir / f"{item['sample_id']}.json"
            )

    assignment = {
        "schema_version": "1.0",
        "seed": args.seed,
        "blind_double_annotation": True,
        "weak_labels_visible": False,
        "near_duplicate_policy": (
            "one_representative_per_cluster"
            if args.duplicate_report
            else "not_applied"
        ),
        "protocol": "docs/thesis/05_annotation_protocol.md",
        "samples": [
            {
                "sample_id": item["sample_id"],
                "node_count": item["node_count"],
                "size_bin": item["size_bin"],
                "structure_fingerprint": item["fingerprint"],
                "screenshot": str(
                    Path(args.data_dir) / item["sample_id"] / "screenshot.png"
                ),
                "page_graph": str(
                    Path(args.data_dir) / item["sample_id"] / "page_graph.json"
                ),
            }
            for item in selected
        ],
    }
    (output_dir / "assignment.json").write_text(
        json.dumps(assignment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(assignment, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
