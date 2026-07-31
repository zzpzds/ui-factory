"""按设计树重建人工标注中 group 的后代原子覆盖集合。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.design_intent.grouping import normalize_group_source_elements
from src.design_intent.schema import DesignIntentIR, PageGraph
from src.design_intent.validation import validate_intent_ir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package_dir", default="data/annotations/intent_pilot_v1"
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    assignment = json.loads(
        (package_dir / "assignment.json").read_text(encoding="utf-8")
    )
    changed_files = []
    invalid_files = []
    for item in assignment["samples"]:
        sample_id = item["sample_id"]
        graph = PageGraph.load(item["page_graph"])
        for annotator in ("annotator_a", "annotator_b"):
            path = package_dir / annotator / f"{sample_id}.json"
            ir = DesignIntentIR.load(path)
            changes = normalize_group_source_elements(ir)
            errors = validate_intent_ir(ir, graph)
            if errors:
                invalid_files.append(
                    {
                        "path": str(path),
                        "errors": errors,
                    }
                )
                continue
            if not changes:
                continue
            changed_files.append(
                {
                    "path": str(path),
                    "groups": {
                        group_id: {"before": before, "after": after}
                        for group_id, (before, after) in changes.items()
                    },
                }
            )
            if args.apply:
                temporary = path.with_suffix(".json.tmp")
                ir.dump(temporary)
                os.replace(temporary, path)

    result = {
        "mode": "apply" if args.apply else "dry_run",
        "changed_file_count": len(changed_files),
        "invalid_file_count": len(invalid_files),
        "changed_files": changed_files,
        "invalid_files": invalid_files,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if invalid_files:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
