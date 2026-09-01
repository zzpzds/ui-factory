"""对两组页面级结果执行 bootstrap CI、配对置换检验与 Holm 校正。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from src.design_intent.evaluation import build_evaluation_report
from src.design_intent.metrics import METRIC_SPECS


REFERENCE_TIERS = ("human_gold", "ai_silver")
STRATUM_SCOPES = {
    "human_gold": "exploratory_human_reference",
    "ai_silver": "ai_proxy_consistency_only",
    "overall_mixed_descriptive": "mixed_descriptive_only",
}
INFERENCE_STATUSES = {
    "human_gold": "not_performed",
    "ai_silver": "exploratory_low_power",
    "overall_mixed_descriptive": "prohibited",
}
DATA_MANIFEST_HASHES = (
    "assignment",
    "selection_manifest",
    "ai_reference_manifest",
    "source_split",
)
REQUIRED_RUN_HASHES = (*DATA_MANIFEST_HASHES, "checkpoint")
SHA256_TAG_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


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


def _comparison_value(
    status: str,
    value: float | list[float] | None,
    reason_code: str | None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "value": value,
        "reason_code": reason_code,
        **extra,
    }


def _metric_rng(seed: int, metric_name: str, purpose: str) -> np.random.Generator:
    digest = hashlib.sha256(
        f"{seed}:{metric_name}:{purpose}".encode("utf-8")
    ).digest()
    derived = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return np.random.default_rng(derived)


def _tier_ids(report: dict[str, Any], tier: str) -> list[str]:
    return [
        sample_id
        for sample_id in report["dataset"]["expected_ids"]
        if report["per_sample"][sample_id]["reference_tier"] == tier
    ]


def _valid_ids(report: dict[str, Any], tier: str) -> list[str]:
    return [
        sample_id
        for sample_id in _tier_ids(report, tier)
        if report["per_sample"][sample_id]["status"] == "valid"
    ]


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须是 object。")
    return value


def _require_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} 必须是非空字符串。")
    return value


def _validate_run_audit(report_name: str, report: dict[str, Any]) -> dict[str, str]:
    run = _require_mapping(report.get("run"), f"{report_name}.run")
    for field in ("checkpoint", "split", "reference_package"):
        _require_nonempty_string(run.get(field), f"{report_name}.run.{field}")
    statistics_seed = run.get("statistics_seed")
    if isinstance(statistics_seed, bool) or not isinstance(statistics_seed, int):
        raise ValueError(f"{report_name}.run.statistics_seed 必须是整数。")
    bootstrap_samples = run.get("bootstrap_samples")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples <= 0
    ):
        raise ValueError(
            f"{report_name}.run.bootstrap_samples 必须是正整数。"
        )

    hashes = _require_mapping(run.get("hashes"), f"{report_name}.run.hashes")
    validated: dict[str, str] = {}
    for field in REQUIRED_RUN_HASHES:
        value = hashes.get(field)
        if not isinstance(value, str) or SHA256_TAG_PATTERN.fullmatch(value) is None:
            raise ValueError(
                f"{report_name}.run.hashes.{field} 不是合规 sha256。"
            )
        validated[field] = value
    return validated


def _validate_sample_metric(
    report_name: str,
    sample_id: str,
    tier: str,
    metric_name: str,
    metric: Any,
) -> None:
    path = f"{report_name}.per_sample.{sample_id}.metrics.{metric_name}"
    payload = _require_mapping(metric, path)
    status = payload.get("status")
    if status not in {"available", "unavailable"}:
        raise ValueError(f"{path}.status 非法。")
    if payload.get("scope") != STRATUM_SCOPES[tier]:
        raise ValueError(f"{path}.scope 与 reference_tier 不一致。")

    value = payload.get("value")
    reason = payload.get("reason_code")
    if status == "available":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{path}.value 必须是有限数值。")
        if reason is not None:
            raise ValueError(f"{path}.reason_code 在 available 时必须为 null。")
    else:
        if value is not None:
            raise ValueError(f"{path}.value 在 unavailable 时必须为 null。")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"{path}.reason_code 在 unavailable 时不能为空。")

    if tier not in METRIC_SPECS[metric_name].allowed_tiers and status != "unavailable":
        raise ValueError(f"{path}.status 与 reference_tier 不一致。")


def _validate_reference_report(
    report_name: str,
    report: dict[str, Any],
) -> dict[str, str]:
    if report.get("schema_version") != "intent-evaluation/v2":
        raise ValueError(f"{report_name} 不是 intent-evaluation/v2。")
    hashes = _validate_run_audit(report_name, report)

    dataset = _require_mapping(report.get("dataset"), f"{report_name}.dataset")
    expected_ids = dataset.get("expected_ids")
    if (
        not isinstance(expected_ids, list)
        or any(not isinstance(sample_id, str) for sample_id in expected_ids)
        or expected_ids != sorted(expected_ids)
        or len(expected_ids) != len(set(expected_ids))
    ):
        raise ValueError(
            f"{report_name}.dataset.expected_ids 必须是排序且无重复的字符串列表。"
        )
    per_sample = _require_mapping(
        report.get("per_sample"), f"{report_name}.per_sample"
    )
    if set(per_sample) != set(expected_ids):
        raise ValueError(
            f"{report_name}.per_sample keys 必须与 dataset.expected_ids 完全一致。"
        )

    tier_counts = {tier: 0 for tier in REFERENCE_TIERS}
    failed_records: list[dict[str, Any]] = []
    for sample_id in expected_ids:
        record_path = f"{report_name}.per_sample.{sample_id}"
        record = _require_mapping(per_sample[sample_id], record_path)
        if record.get("sample_id") != sample_id:
            raise ValueError(f"{record_path}.sample_id 与 key 不一致。")
        tier = record.get("reference_tier")
        if tier not in REFERENCE_TIERS:
            raise ValueError(f"{record_path}.reference_tier 非法。")
        tier_counts[tier] += 1
        status = record.get("status")
        if status not in {"valid", "failed"}:
            raise ValueError(f"{record_path}.status 非法。")
        failure = record.get("failure")
        if status == "valid" and failure is not None:
            raise ValueError(f"{record_path}.failure 与 valid status 不一致。")
        if status == "failed":
            failure_payload = _require_mapping(failure, f"{record_path}.failure")
            if failure_payload.get("stage") not in {"decode", "ir_validation"}:
                raise ValueError(f"{record_path}.failure.stage 非法。")
            for field in ("error_type", "error_summary"):
                _require_nonempty_string(
                    failure_payload.get(field), f"{record_path}.failure.{field}"
                )
            failed_records.append(record)

        metrics = _require_mapping(record.get("metrics"), f"{record_path}.metrics")
        if set(metrics) != set(METRIC_SPECS):
            raise ValueError(f"{record_path}.metric catalog 不完整。")
        for metric_name in METRIC_SPECS:
            _validate_sample_metric(
                report_name,
                sample_id,
                tier,
                metric_name,
                metrics[metric_name],
            )
            if status == "failed" and metrics[metric_name]["status"] != "unavailable":
                raise ValueError(
                    f"{record_path}.metrics.{metric_name} 与 failed status 不一致。"
                )

    counts_by_tier = dataset.get("counts_by_tier")
    if counts_by_tier != tier_counts:
        raise ValueError(
            f"{report_name}.dataset.counts_by_tier 与逐页 reference_tier 不一致。"
        )
    failures = report.get("failures")
    if failures != failed_records:
        raise ValueError(
            f"{report_name}.failures 必须与 failed 逐页记录完全一致。"
        )

    strata = _require_mapping(report.get("strata"), f"{report_name}.strata")
    if set(strata) != set(STRATUM_SCOPES):
        raise ValueError(f"{report_name}.strata catalog 不完整。")
    for stratum, scope in STRATUM_SCOPES.items():
        stratum_payload = _require_mapping(
            strata[stratum], f"{report_name}.strata.{stratum}"
        )
        if stratum_payload.get("scope") != scope:
            raise ValueError(f"{report_name}.strata.{stratum}.scope 非法。")
        stratum_records = [
            per_sample[sample_id]
            for sample_id in expected_ids
            if stratum == "overall_mixed_descriptive"
            or per_sample[sample_id]["reference_tier"] == stratum
        ]
        valid = sum(record["status"] == "valid" for record in stratum_records)
        expected_counts = {
            "expected": len(stratum_records),
            "valid": valid,
            "failed": len(stratum_records) - valid,
        }
        if stratum_payload.get("counts") != expected_counts:
            raise ValueError(f"{report_name}.strata.{stratum}.counts 不一致。")
        aggregate_metrics = _require_mapping(
            stratum_payload.get("aggregate_metrics"),
            f"{report_name}.strata.{stratum}.aggregate_metrics",
        )
        if set(aggregate_metrics) != set(METRIC_SPECS):
            raise ValueError(
                f"{report_name}.strata.{stratum}.metric catalog 不完整。"
            )
        inference = _require_mapping(
            stratum_payload.get("inference"),
            f"{report_name}.strata.{stratum}.inference",
        )
        if inference.get("status") != INFERENCE_STATUSES[stratum]:
            raise ValueError(f"{report_name}.strata.{stratum}.inference 非法。")

    run = report["run"]
    rebuilt = build_evaluation_report(
        [per_sample[sample_id] for sample_id in expected_ids],
        run=run,
        expected_ids=expected_ids,
        statistics_seed=run["statistics_seed"],
        bootstrap_samples=run["bootstrap_samples"],
    )
    for stratum in STRATUM_SCOPES:
        if (
            strata[stratum]["aggregate_metrics"]
            != rebuilt["strata"][stratum]["aggregate_metrics"]
        ):
            raise ValueError(
                f"{report_name}.strata.{stratum}.aggregate_metrics "
                "与逐页结果重算不一致。"
            )
    return hashes


def _empty_metric_comparison(
    metric_name: str,
    *,
    reason_code: str,
    method_available_ids: list[str],
    baseline_available_ids: list[str],
) -> dict[str, Any]:
    spec = METRIC_SPECS[metric_name]
    unavailable = _comparison_value("unavailable", None, reason_code)
    return {
        "status": "unavailable",
        "reason_code": reason_code,
        "family": spec.family,
        "direction": spec.direction,
        "holm_family": spec.holm_family,
        "method_available_ids": method_available_ids,
        "baseline_available_ids": baseline_available_ids,
        "paired_ids": [],
        "n_paired": 0,
        "method_mean": dict(unavailable),
        "baseline_mean": dict(unavailable),
        "mean_improvement": dict(unavailable),
        "bootstrap_mean_95_ci": dict(unavailable),
        "cohen_dz": dict(unavailable),
        "permutation_p": dict(unavailable),
        "holm_input_p": 1.0,
        "holm_adjusted_p": _comparison_value(
            "unavailable", None, reason_code, mechanical_value=1.0
        ),
    }


def _compare_metric(
    method: dict[str, Any],
    baseline: dict[str, Any],
    valid_ids: list[str],
    metric_name: str,
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, Any]:
    method_available = [
        sample_id
        for sample_id in valid_ids
        if method["per_sample"][sample_id]["metrics"][metric_name]["status"]
        == "available"
    ]
    baseline_available = [
        sample_id
        for sample_id in valid_ids
        if baseline["per_sample"][sample_id]["metrics"][metric_name]["status"]
        == "available"
    ]
    if method_available != baseline_available:
        return _empty_metric_comparison(
            metric_name,
            reason_code="metric_cohort_mismatch",
            method_available_ids=method_available,
            baseline_available_ids=baseline_available,
        )
    if not method_available:
        return _empty_metric_comparison(
            metric_name,
            reason_code="no_paired_metric_pages",
            method_available_ids=method_available,
            baseline_available_ids=baseline_available,
        )

    spec = METRIC_SPECS[metric_name]
    method_values = np.asarray(
        [
            method["per_sample"][sample_id]["metrics"][metric_name]["value"]
            for sample_id in method_available
        ],
        dtype=np.float64,
    )
    baseline_values = np.asarray(
        [
            baseline["per_sample"][sample_id]["metrics"][metric_name]["value"]
            for sample_id in method_available
        ],
        dtype=np.float64,
    )
    differences = (
        method_values - baseline_values
        if spec.direction == "higher"
        else baseline_values - method_values
    )
    result: dict[str, Any] = {
        "status": "available",
        "reason_code": None,
        "family": spec.family,
        "direction": spec.direction,
        "holm_family": spec.holm_family,
        "method_available_ids": method_available,
        "baseline_available_ids": baseline_available,
        "paired_ids": method_available,
        "n_paired": len(method_available),
        "method_mean": _comparison_value(
            "available", float(method_values.mean()), None
        ),
        "baseline_mean": _comparison_value(
            "available", float(baseline_values.mean()), None
        ),
        "mean_improvement": _comparison_value(
            "available", float(differences.mean()), None
        ),
    }

    if len(method_available) == 1:
        reason = "insufficient_paired_metric_pages"
        for name in ("bootstrap_mean_95_ci", "cohen_dz", "permutation_p"):
            result[name] = _comparison_value("unavailable", None, reason)
        result["holm_input_p"] = 1.0
        result["holm_adjusted_p"] = _comparison_value(
            "unavailable", None, reason, mechanical_value=1.0
        )
        return result

    result["bootstrap_mean_95_ci"] = _comparison_value(
        "available",
        bootstrap_ci(
            differences,
            bootstrap_samples,
            _metric_rng(seed, metric_name, "bootstrap"),
        ),
        None,
    )
    raw_p = paired_permutation_p(
        differences,
        permutation_samples,
        _metric_rng(seed, metric_name, "permutation"),
    )
    result["permutation_p"] = _comparison_value("available", raw_p, None)
    result["holm_input_p"] = raw_p
    difference_std = float(differences.std(ddof=1))
    if difference_std == 0.0:
        result["cohen_dz"] = _comparison_value(
            "unavailable", None, "zero_paired_difference_variance"
        )
    else:
        result["cohen_dz"] = _comparison_value(
            "available", float(differences.mean() / difference_std), None
        )
    result["holm_adjusted_p"] = _comparison_value(
        "available", None, None
    )
    return result


def compare_reference_reports(
    method: dict[str, Any],
    baseline: dict[str, Any],
    *,
    tier: str = "ai_silver",
    bootstrap_samples: int = 10_000,
    permutation_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """对两个 v2 报告执行来源与 cohort 受控的配对比较。"""
    if tier != "ai_silver":
        raise ValueError("v2 推断比较仅允许 tier=ai_silver。")
    if bootstrap_samples <= 0 or permutation_samples <= 0:
        raise ValueError("bootstrap/permutation samples 必须大于 0。")
    method_hashes = _validate_reference_report("method", method)
    baseline_hashes = _validate_reference_report("baseline", baseline)
    for hash_name in DATA_MANIFEST_HASHES:
        if method_hashes[hash_name] != baseline_hashes[hash_name]:
            raise ValueError(f"两份报告的 {hash_name} 哈希不一致。")
    if method["run"]["split"] != baseline["run"]["split"]:
        raise ValueError("两份报告的 run.split 不一致。")
    if method["dataset"]["expected_ids"] != baseline["dataset"]["expected_ids"]:
        raise ValueError("两份报告的预定 cohort 不一致。")

    method_expected = _tier_ids(method, tier)
    baseline_expected = _tier_ids(baseline, tier)
    if method_expected != baseline_expected:
        raise ValueError("两份报告的预定 AI Silver cohort 不一致。")
    method_valid = _valid_ids(method, tier)
    baseline_valid = _valid_ids(baseline, tier)
    if method_valid != baseline_valid:
        raise ValueError("两份报告的 prediction-valid cohort 不一致。")

    primary_metrics = [
        name
        for name, spec in METRIC_SPECS.items()
        if spec.primary and tier in spec.allowed_tiers
    ]
    metric_results = {
        metric_name: _compare_metric(
            method,
            baseline,
            method_valid,
            metric_name,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
            seed=seed,
        )
        for metric_name in primary_metrics
    }

    families = sorted(
        {METRIC_SPECS[name].holm_family for name in primary_metrics}
    )
    for family in families:
        family_names = [
            name
            for name in primary_metrics
            if METRIC_SPECS[name].holm_family == family
        ]
        adjusted = holm_adjust(
            {name: metric_results[name]["holm_input_p"] for name in family_names}
        )
        for name in family_names:
            raw_status = metric_results[name]["permutation_p"]["status"]
            if raw_status == "available":
                metric_results[name]["holm_adjusted_p"] = _comparison_value(
                    "available", adjusted[name], None
                )
            else:
                reason = metric_results[name]["permutation_p"]["reason_code"]
                metric_results[name]["holm_adjusted_p"] = _comparison_value(
                    "unavailable",
                    None,
                    reason,
                    mechanical_value=adjusted[name],
                )

    output = {
        "schema_version": "intent-comparison/v2",
        "tier": tier,
        "method": method.get("run", {}),
        "baseline": baseline.get("run", {}),
        "statistics": {
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "permutation_samples": permutation_samples,
        },
        "cohort": {
            "expected_ids": method_expected,
            "method_valid_ids": method_valid,
            "baseline_valid_ids": baseline_valid,
            "method_failed_ids": sorted(set(method_expected) - set(method_valid)),
            "baseline_failed_ids": sorted(
                set(baseline_expected) - set(baseline_valid)
            ),
        },
        "metrics": metric_results,
    }
    json.dumps(output, ensure_ascii=False, allow_nan=False)
    return output


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _validate_legacy_report(report_name: str, report: Any) -> None:
    if not isinstance(report, dict):
        raise ValueError(f"{report_name} legacy 报告必须是 object。")
    per_sample = report.get("per_sample")
    if not isinstance(per_sample, dict):
        raise ValueError(f"{report_name} legacy per_sample 必须是 object。")
    for sample_id, metrics in per_sample.items():
        if not isinstance(sample_id, str) or not isinstance(metrics, dict):
            raise ValueError(f"{report_name} legacy per_sample 结构非法。")
        for metric_name, value in metrics.items():
            if not isinstance(metric_name, str) or (
                value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                )
            ):
                raise ValueError(
                    f"{report_name} legacy per_sample 必须是扁平数值映射。"
                )


def _comparison_mode(method: Any, baseline: Any) -> str:
    method_has_version = isinstance(method, dict) and "schema_version" in method
    baseline_has_version = (
        isinstance(baseline, dict) and "schema_version" in baseline
    )
    if method_has_version or baseline_has_version:
        if (
            method_has_version
            and baseline_has_version
            and method.get("schema_version") == "intent-evaluation/v2"
            and baseline.get("schema_version") == "intent-evaluation/v2"
        ):
            return "v2"
        raise ValueError(
            "两份报告的 schema_version 必须同时为 intent-evaluation/v2。"
        )
    _validate_legacy_report("method", method)
    _validate_legacy_report("baseline", baseline)
    return "legacy"


def _compare_legacy_results(
    method: dict[str, Any],
    baseline: dict[str, Any],
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, Any]:
    method_pages = method["per_sample"]
    baseline_pages = baseline["per_sample"]
    common_ids = sorted(set(method_pages) & set(baseline_pages))
    if bootstrap_samples <= 0 or permutation_samples <= 0:
        raise ValueError("bootstrap/permutation samples 必须大于 0。")
    metrics = sorted(
        {
            metric
            for sample_id in common_ids
            for page in (method_pages[sample_id], baseline_pages[sample_id])
            for metric in page
        }
    )

    rng = np.random.default_rng(seed)
    results = {}
    raw_p_values = {}
    for metric in metrics:
        paired_ids = [
            sample_id
            for sample_id in common_ids
            if _is_finite_number(method_pages[sample_id].get(metric))
            and _is_finite_number(baseline_pages[sample_id].get(metric))
        ]
        if len(paired_ids) < 2:
            results[metric] = {
                "status": "unavailable",
                "reason_code": "insufficient_paired_metric_pages",
                "paired_ids": paired_ids,
                "n_paired": len(paired_ids),
                "method_mean": None,
                "baseline_mean": None,
                "paired_difference": None,
                "bootstrap_95_ci": None,
                "cohen_dz": None,
                "permutation_p": None,
                "holm_adjusted_p": None,
            }
            continue
        method_values = np.asarray(
            [method_pages[sample_id][metric] for sample_id in paired_ids],
            dtype=np.float64,
        )
        baseline_values = np.asarray(
            [baseline_pages[sample_id][metric] for sample_id in paired_ids],
            dtype=np.float64,
        )
        differences = method_values - baseline_values
        std = float(differences.std(ddof=1))
        raw_p = paired_permutation_p(differences, permutation_samples, rng)
        raw_p_values[metric] = raw_p
        results[metric] = {
            "status": "available",
            "reason_code": None,
            "paired_ids": paired_ids,
            "n_paired": len(paired_ids),
            "method_mean": float(method_values.mean()),
            "baseline_mean": float(baseline_values.mean()),
            "paired_difference": float(differences.mean()),
            "bootstrap_95_ci": bootstrap_ci(
                differences, bootstrap_samples, rng
            ),
            "cohen_dz": (
                float(differences.mean() / std) if std != 0.0 else None
            ),
            "permutation_p": raw_p,
        }

    adjusted = holm_adjust(raw_p_values)
    for metric, value in adjusted.items():
        results[metric]["holm_adjusted_p"] = value
    return {
        "paired_pages": len(common_ids),
        "seed": seed,
        "metrics": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--permutation_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tier", default="ai_silver")
    parser.add_argument("--output", default="outputs/intent-comparison.json")
    args = parser.parse_args()

    method = json.loads(Path(args.method).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    if _comparison_mode(method, baseline) == "v2":
        output = compare_reference_reports(
            method,
            baseline,
            tier=args.tier,
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
            seed=args.seed,
        )
    else:
        output = {
            "method": args.method,
            "baseline": args.baseline,
            **_compare_legacy_results(
                method,
                baseline,
                bootstrap_samples=args.bootstrap_samples,
                permutation_samples=args.permutation_samples,
                seed=args.seed,
            ),
        }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
