"""汇总 Pilot PageGraph 与弱标签的规模、截断和压缩统计。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean, median
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from src.design_intent.schema import DesignIntentIR, PageGraph


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def percentile(ratio: float) -> float:
        return float(ordered[round((len(ordered) - 1) * ratio)])

    return {
        "min": float(ordered[0]),
        "median": float(median(ordered)),
        "mean": float(mean(ordered)),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "max": float(ordered[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument(
        "--pilot_manifest", default="data/intent_pilot_manifest.json"
    )
    parser.add_argument(
        "--render_report", default="outputs/intent-pilot-500/render-report.json"
    )
    parser.add_argument(
        "--output", default="outputs/intent-pilot-500/pilot-analysis.json"
    )
    args = parser.parse_args()

    manifest = json.loads(
        Path(args.pilot_manifest).read_text(encoding="utf-8")
    )
    render_report = json.loads(
        Path(args.render_report).read_text(encoding="utf-8")
    )
    records = []
    blank_candidates = []
    wrong_dimensions = []
    for sample_id in manifest["sample_ids"]:
        sample_dir = Path(args.data_dir) / sample_id
        graph_path = sample_dir / "page_graph.json"
        intent_path = sample_dir / "weak_intent.json"
        if not graph_path.exists() or not intent_path.exists():
            continue
        graph = PageGraph.load(graph_path)
        intent = DesignIntentIR.load(intent_path)
        image = Image.open(sample_dir / "screenshot.png").convert("L")
        if image.size != (
            int(round(graph.canvas.width)),
            int(round(graph.canvas.height)),
        ):
            wrong_dimensions.append(sample_id)
        luminance_std = float(np.asarray(image, dtype=np.float32).std())
        if luminance_std < 2.0 or len(graph.nodes) < 5:
            blank_candidates.append(sample_id)
        entity_count = len(intent.elements) + len(intent.groups)
        records.append(
            {
                "sample_id": sample_id,
                "nodes": len(graph.nodes),
                "elements": len(intent.elements),
                "groups": len(intent.groups),
                "layouts": len(intent.layouts),
                "tokens": len(intent.style_tokens),
                "entities": entity_count,
                "compression": entity_count / max(len(graph.nodes), 1),
                "luminance_std": luminance_std,
            }
        )

    node_counts = [record["nodes"] for record in records]
    result = {
        "manifest_sha256": manifest["sample_ids_sha256"],
        "requested": manifest.get(
            "requested_samples", len(manifest["sample_ids"])
        ),
        "successful_manifest_samples": len(manifest["sample_ids"]),
        "render_success": render_report["success"],
        "render_failed": render_report["failed"],
        "weak_labels_analyzed": len(records),
        "render_success_rate": (
            render_report["success"] / max(render_report["requested"], 1)
        ),
        "node_count": summary(node_counts),
        "element_count": summary([record["elements"] for record in records]),
        "group_count": summary([record["groups"] for record in records]),
        "layout_count": summary([record["layouts"] for record in records]),
        "token_count": summary([record["tokens"] for record in records]),
        "entity_to_node_ratio": summary(
            [record["compression"] for record in records]
        ),
        "screenshot_luminance_std": summary(
            [record["luminance_std"] for record in records]
        ),
        "blank_candidates": blank_candidates,
        "wrong_screenshot_dimensions": wrong_dimensions,
        "truncation_rate": {
            str(limit): sum(count > limit for count in node_counts)
            / max(len(node_counts), 1)
            for limit in (50, 128, 192, 256)
        },
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
