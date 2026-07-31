"""为人工标注创建不含弱标签的空白 Design Intent IR。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.design_intent.schema import DesignIntentIR, PageGraph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page_graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotator", required=True)
    args = parser.parse_args()

    graph = PageGraph.load(args.page_graph)
    annotation = DesignIntentIR(
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
            "annotator": args.annotator,
            "status": "draft",
            "weak_labels_viewed": False,
        },
    )
    annotation.dump(Path(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
