"""根据渲染报告冻结实际成功的 Pilot 样本集合。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection_manifest", default="data/intent_pilot_manifest.json"
    )
    parser.add_argument(
        "--render_report", default="outputs/intent-pilot-500/render-report.json"
    )
    parser.add_argument(
        "--output", default="data/intent_pilot_success_manifest.json"
    )
    args = parser.parse_args()

    selection = json.loads(
        Path(args.selection_manifest).read_text(encoding="utf-8")
    )
    report = json.loads(Path(args.render_report).read_text(encoding="utf-8"))
    if report["requested"] != selection["selected_samples"]:
        raise ValueError("渲染报告与选择 manifest 的请求数不一致。")
    failed_ids = {item["sample_id"] for item in report["failures"]}
    successful_ids = [
        sample_id
        for sample_id in selection["sample_ids"]
        if sample_id not in failed_ids
    ]
    if len(successful_ids) != report["success"]:
        raise ValueError("成功样本数与渲染报告不一致。")
    digest = hashlib.sha256(
        "\n".join(successful_ids).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "purpose": "design_intent_pilot_successful_renders",
        "parent_manifest_sha256": selection["sample_ids_sha256"],
        "sample_ids_sha256": digest,
        "requested_samples": report["requested"],
        "successful_samples": report["success"],
        "failed_samples": report["failed"],
        "sample_ids": successful_ids,
        "failures": report["failures"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
