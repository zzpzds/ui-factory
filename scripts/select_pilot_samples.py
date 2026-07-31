"""从已有 HTML 样本中可复现地选择 Pilot 页面。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="data/intent_pilot_manifest.json"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    eligible = [
        sample_dir.name
        for sample_dir in sorted(data_dir.iterdir())
        if sample_dir.is_dir()
        and (sample_dir / "page.html").exists()
        and (sample_dir / "screenshot.png").exists()
    ]
    if len(eligible) < args.num_samples:
        raise ValueError(
            f"可用样本只有 {len(eligible)}，少于请求的 {args.num_samples}"
        )
    selected = sorted(
        random.Random(args.seed).sample(eligible, args.num_samples)
    )
    digest = hashlib.sha256(
        "\n".join(selected).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "purpose": "design_intent_pilot",
        "data_dir": str(data_dir),
        "seed": args.seed,
        "eligible_samples": len(eligible),
        "selected_samples": len(selected),
        "selection": "uniform_without_replacement",
        "sample_ids_sha256": digest,
        "sample_ids": selected,
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
