"""使用截图 dHash 检测跨切分近重复页面。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def difference_hash(path: Path, hash_size: int = 16) -> int:
    image = Image.open(path).convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = list(image.getdata())
    value = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for column in range(hash_size):
            value = (value << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--split_manifest", default="data/intent_split.json")
    parser.add_argument("--threshold", type=int, default=12)
    parser.add_argument(
        "--output", default="outputs/near-duplicate-report.json"
    )
    args = parser.parse_args()

    with open(args.split_manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    split_by_id = {
        sample_id: split_name
        for split_name, sample_ids in manifest["splits"].items()
        for sample_id in sample_ids
    }
    hashes = {
        sample_id: difference_hash(
            Path(args.data_dir) / sample_id / "screenshot.png"
        )
        for sample_id in split_by_id
    }

    ids = sorted(hashes)
    matches = []
    for index, left_id in enumerate(ids):
        for right_id in ids[index + 1 :]:
            distance = hamming_distance(hashes[left_id], hashes[right_id])
            if distance <= args.threshold:
                matches.append(
                    {
                        "left": left_id,
                        "right": right_id,
                        "distance": distance,
                        "left_split": split_by_id[left_id],
                        "right_split": split_by_id[right_id],
                        "cross_split": (
                            split_by_id[left_id] != split_by_id[right_id]
                        ),
                    }
                )

    result = {
        "method": "dhash_16x16",
        "threshold": args.threshold,
        "samples": len(ids),
        "matches": matches,
        "cross_split_matches": sum(
            int(match["cross_split"]) for match in matches
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
