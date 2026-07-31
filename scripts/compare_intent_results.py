"""对两组页面级结果执行 bootstrap CI、配对置换检验与 Holm 校正。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_ci(
    differences: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    indices = rng.integers(
        0, len(differences), size=(samples, len(differences))
    )
    means = differences[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def paired_permutation_p(
    differences: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> float:
    observed = abs(float(differences.mean()))
    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(samples, len(differences)),
    )
    permuted = np.abs((signs * differences).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (samples + 1))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (count - rank) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--permutation_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/intent-comparison.json")
    args = parser.parse_args()

    method = json.loads(Path(args.method).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    method_pages = method["per_sample"]
    baseline_pages = baseline["per_sample"]
    common_ids = sorted(set(method_pages) & set(baseline_pages))
    if len(common_ids) < 2:
        raise ValueError("配对统计至少需要 2 个共同页面。")
    metrics = sorted(
        set.intersection(
            *(set(method_pages[sample_id]) for sample_id in common_ids),
            *(set(baseline_pages[sample_id]) for sample_id in common_ids),
        )
    )

    rng = np.random.default_rng(args.seed)
    results = {}
    raw_p_values = {}
    for metric in metrics:
        method_values = np.array(
            [method_pages[sample_id][metric] for sample_id in common_ids]
        )
        baseline_values = np.array(
            [baseline_pages[sample_id][metric] for sample_id in common_ids]
        )
        differences = method_values - baseline_values
        std = float(differences.std(ddof=1))
        raw_p = paired_permutation_p(
            differences, args.permutation_samples, rng
        )
        raw_p_values[metric] = raw_p
        results[metric] = {
            "method_mean": float(method_values.mean()),
            "baseline_mean": float(baseline_values.mean()),
            "paired_difference": float(differences.mean()),
            "bootstrap_95_ci": bootstrap_ci(
                differences, args.bootstrap_samples, rng
            ),
            "cohen_dz": (
                float(differences.mean() / std) if std > 0 else None
            ),
            "permutation_p": raw_p,
        }

    adjusted = holm_adjust(raw_p_values)
    for metric, value in adjusted.items():
        results[metric]["holm_adjusted_p"] = value
    output = {
        "method": args.method,
        "baseline": args.baseline,
        "paired_pages": len(common_ids),
        "seed": args.seed,
        "metrics": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
