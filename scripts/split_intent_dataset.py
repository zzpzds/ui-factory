"""按站点或页面结构指纹分组切分，避免模板泄漏。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.design_intent.schema import PageGraph


def structural_fingerprint(graph: PageGraph) -> str:
    signature = [
        (
            node.tag,
            node.parent_id,
            min(node.child_count, 8),
            str(node.computed_style.get("display", "block")),
        )
        for node in graph.nodes
    ]
    payload = json.dumps(signature, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def grouping_key(
    graph: PageGraph, sample_dir: Path | None = None
) -> str:
    source_meta = graph.metadata.get("source_meta", {})
    url = (
        graph.metadata.get("url")
        or source_meta.get("url")
        or source_meta.get("source_url")
    )
    if url:
        domain = urlparse(str(url)).netloc.lower()
        if domain:
            return f"domain:{domain}"
    if sample_dir is not None:
        meta_path = sample_dir / "meta.json"
        if meta_path.exists():
            try:
                source_hash = str(
                    json.loads(meta_path.read_text(encoding="utf-8")).get(
                        "hash", ""
                    )
                ).strip()
                if source_hash:
                    return f"source_hash:{source_hash}"
            except (OSError, ValueError, TypeError):
                pass
    return f"structure:{structural_fingerprint(graph)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/sampled")
    parser.add_argument(
        "--sample_manifest", default=None, help="包含 sample_ids 的数据清单"
    )
    parser.add_argument("--output", default="data/intent_split.json")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    groups: dict[str, list[str]] = {}
    fingerprints: dict[str, str] = {}
    selected_ids = None
    if args.sample_manifest:
        with open(args.sample_manifest, encoding="utf-8") as f:
            selected_ids = set(json.load(f)["sample_ids"])
    for sample_dir in sorted(Path(args.data_dir).iterdir()):
        if selected_ids is not None and sample_dir.name not in selected_ids:
            continue
        graph_path = sample_dir / "page_graph.json"
        if not graph_path.exists():
            continue
        graph = PageGraph.load(graph_path)
        key = grouping_key(graph, sample_dir)
        groups.setdefault(key, []).append(sample_dir.name)
        fingerprints[sample_dir.name] = structural_fingerprint(graph)

    group_items = list(groups.items())
    random.Random(args.seed).shuffle(group_items)
    total = sum(len(samples) for _, samples in group_items)
    train_target = total * args.train_ratio
    val_target = total * args.val_ratio
    split = {"train": [], "validation": [], "test": []}
    for _, samples in group_items:
        if len(split["train"]) < train_target:
            target = "train"
        elif len(split["validation"]) < val_target:
            target = "validation"
        else:
            target = "test"
        split[target].extend(samples)

    manifest = {
        "seed": args.seed,
        "grouping": (
            "domain_if_available_else_source_hash_else_"
            "exact_structure_fingerprint"
        ),
        "ratios": {
            "train": args.train_ratio,
            "validation": args.val_ratio,
            "test": 1.0 - args.train_ratio - args.val_ratio,
        },
        "splits": {key: sorted(values) for key, values in split.items()},
        "fingerprints": fingerprints,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {key: len(values) for key, values in split.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
