"""Reference v1 的纯评测结果构造与分层汇总。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
from typing import Any

import numpy as np

from .metrics import METRIC_SPECS


REFERENCE_TIERS = ("human_gold", "ai_silver")
STRATUM_SCOPES = {
    "human_gold": "exploratory_human_reference",
    "ai_silver": "ai_proxy_consistency_only",
    "overall_mixed_descriptive": "mixed_descriptive_only",
}
TOKEN_METRICS = frozenset(
    name for name, spec in METRIC_SPECS.items() if spec.family == "token_proxy"
)
CONDITIONAL_LAYOUT_METRICS = frozenset(
    {
        "layout_mode_accuracy",
        "layout_mode_macro_f1",
        "gap_normalized_mae",
        "padding_normalized_mae",
    }
)
REQUIRED_RUN_FIELDS = (
    "checkpoint",
    "split",
    "reference_package",
    "statistics_seed",
    "bootstrap_samples",
    "hashes",
)
REQUIRED_RUN_HASHES = (
    "assignment",
    "selection_manifest",
    "ai_reference_manifest",
    "source_split",
    "checkpoint",
)
SHA256_TAG_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def _validate_run_audit(
    run: Mapping[str, Any],
    *,
    statistics_seed: int,
    bootstrap_samples: int,
) -> None:
    if not isinstance(run, Mapping):
        raise ValueError("run 必须是 object。")
    for field in REQUIRED_RUN_FIELDS:
        if field not in run:
            raise ValueError(f"run 缺少必填字段 {field}。")

    for field in ("checkpoint", "split", "reference_package"):
        if not isinstance(run[field], str) or not run[field]:
            raise ValueError(f"run.{field} 必须是非空字符串。")

    run_seed = run["statistics_seed"]
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise ValueError("run.statistics_seed 必须是整数。")
    if run_seed != statistics_seed:
        raise ValueError("run.statistics_seed 与实际聚合参数不一致。")

    run_bootstrap_samples = run["bootstrap_samples"]
    if (
        isinstance(run_bootstrap_samples, bool)
        or not isinstance(run_bootstrap_samples, int)
        or run_bootstrap_samples <= 0
    ):
        raise ValueError("run.bootstrap_samples 必须是正整数。")
    if run_bootstrap_samples != bootstrap_samples:
        raise ValueError("run.bootstrap_samples 与实际聚合参数不一致。")

    hashes = run["hashes"]
    if not isinstance(hashes, Mapping):
        raise ValueError("run.hashes 必须是 object。")
    for name in REQUIRED_RUN_HASHES:
        if name not in hashes:
            raise ValueError(f"run.hashes 缺少必填字段 {name}。")
        value = hashes[name]
        if not isinstance(value, str) or SHA256_TAG_PATTERN.fullmatch(value) is None:
            raise ValueError(
                f"run.hashes.{name} 必须是 sha256:<64hex>。"
            )


def _metric_object(
    *,
    status: str,
    value: float | None,
    scope: str,
    reason_code: str | None,
    interpretation: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "value": value,
        "scope": scope,
        "reason_code": reason_code,
    }
    if interpretation is not None:
        result["interpretation"] = interpretation
    return result


def _unavailable_reason(
    metric_name: str,
    raw_metrics: Mapping[str, float | None],
) -> str:
    if metric_name == "layout_match_coverage":
        return "reference_layout_labels_absent"
    if metric_name in CONDITIONAL_LAYOUT_METRICS:
        coverage = raw_metrics.get("layout_match_coverage")
        if coverage is None:
            return "reference_layout_labels_absent"
        if coverage == 0:
            return "no_matched_layout"
    return "metric_not_computed"


def make_valid_sample_result(
    sample_id: str,
    reference_tier: str,
    raw_metrics: Mapping[str, float | None],
) -> dict[str, Any]:
    """把核心数值转成固定 catalog 的逐页机器可读结果。"""
    if reference_tier not in REFERENCE_TIERS:
        raise ValueError(f"未知 reference tier：{reference_tier}")
    unknown = sorted(set(raw_metrics) - set(METRIC_SPECS))
    if unknown:
        raise ValueError(f"未知 metric：{unknown}")

    scope = STRATUM_SCOPES[reference_tier]
    metrics: dict[str, dict[str, Any]] = {}
    for name, spec in METRIC_SPECS.items():
        if reference_tier == "human_gold" and name in TOKEN_METRICS:
            metrics[name] = _metric_object(
                status="unavailable",
                value=None,
                scope=scope,
                reason_code="human_gold_token_labels_not_collected",
            )
            continue

        value = raw_metrics.get(name)
        if value is None:
            metrics[name] = _metric_object(
                status="unavailable",
                value=None,
                scope=scope,
                reason_code=_unavailable_reason(name, raw_metrics),
                interpretation=(
                    spec.interpretation
                    if reference_tier == "ai_silver"
                    else None
                ),
            )
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{sample_id} 的 {name} 必须是数值或 None。")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{sample_id} 的 {name} 不是有限值。")
        metrics[name] = _metric_object(
            status="available",
            value=numeric,
            scope=scope,
            reason_code=None,
            interpretation=(
                spec.interpretation if reference_tier == "ai_silver" else None
            ),
        )

    return {
        "sample_id": str(sample_id),
        "reference_tier": reference_tier,
        "status": "valid",
        "failure": None,
        "metrics": metrics,
    }


def make_failed_sample_result(
    sample_id: str,
    reference_tier: str,
    *,
    stage: str,
    error_type: str,
    error_summary: str,
) -> dict[str, Any]:
    if reference_tier not in REFERENCE_TIERS:
        raise ValueError(f"未知 reference tier：{reference_tier}")
    if stage not in {"decode", "ir_validation"}:
        raise ValueError(f"不允许降级为逐页失败的 stage：{stage}")
    normalized_error_type = str(error_type).strip()
    if not normalized_error_type:
        raise ValueError("error_type 不能为空。")
    normalized_error_summary = str(error_summary).strip() or normalized_error_type
    scope = STRATUM_SCOPES[reference_tier]
    metrics = {
        name: _metric_object(
            status="unavailable",
            value=None,
            scope=scope,
            reason_code="prediction_failed",
            interpretation=(
                spec.interpretation if reference_tier == "ai_silver" else None
            ),
        )
        for name, spec in METRIC_SPECS.items()
    }
    return {
        "sample_id": str(sample_id),
        "reference_tier": reference_tier,
        "status": "failed",
        "failure": {
            "stage": stage,
            "error_type": normalized_error_type,
            "error_summary": normalized_error_summary,
        },
        "metrics": metrics,
    }


def _summary_value(
    status: str,
    value: float | list[float] | None,
    reason_code: str | None,
) -> dict[str, Any]:
    return {"status": status, "value": value, "reason_code": reason_code}


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    metric_name: str,
    samples: int,
    seed: int,
) -> list[float]:
    digest = hashlib.sha256(f"{seed}:{metric_name}".encode("utf-8")).digest()
    metric_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(metric_seed)
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _metric_counts(
    records: Sequence[Mapping[str, Any]],
    metric_name: str,
) -> tuple[dict[str, Any], list[float]]:
    valid = [record for record in records if record["status"] == "valid"]
    available = [
        record["metrics"][metric_name]
        for record in valid
        if record["metrics"][metric_name]["status"] == "available"
    ]
    unavailable = [
        record["metrics"][metric_name]
        for record in valid
        if record["metrics"][metric_name]["status"] != "available"
    ]
    reasons = Counter(item["reason_code"] for item in unavailable)
    counts = {
        "n_expected_pages": len(records),
        "n_valid_predictions": len(valid),
        "n_failed_predictions": len(records) - len(valid),
        "n_available": len(available),
        "n_unavailable": len(unavailable),
        "unavailable_reason_counts": dict(sorted(reasons.items())),
    }
    return counts, [float(item["value"]) for item in available]


def _aggregate_metric(
    records: Sequence[Mapping[str, Any]],
    metric_name: str,
    *,
    stratum: str,
    statistics_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    spec = METRIC_SPECS[metric_name]
    scope = STRATUM_SCOPES[stratum]
    counts, values = _metric_counts(records, metric_name)
    result: dict[str, Any] = {
        "status": "available",
        "scope": scope,
        "counts": counts,
        "mean": _summary_value("unavailable", None, "no_metric_values"),
        "sample_std": _summary_value("unavailable", None, "no_metric_values"),
        "bootstrap_mean_95_ci": _summary_value(
            "unavailable", None, "no_metric_values"
        ),
        "estimand": "mean_among_metric_available_valid_predictions",
        "reason_code": None,
    }
    if stratum == "ai_silver" and spec.interpretation is not None:
        result["interpretation"] = spec.interpretation

    if stratum == "human_gold":
        reason = "human_gold_aggregate_prohibited"
        result.update(status="suppressed_by_protocol", reason_code=reason)
        for name in ("mean", "sample_std", "bootstrap_mean_95_ci"):
            result[name] = _summary_value(
                "suppressed_by_protocol", None, reason
            )
        return result

    if stratum == "overall_mixed_descriptive" and metric_name in TOKEN_METRICS:
        reason = "cross_tier_token_truth_spaces_incomparable"
        result.update(status="unavailable", reason_code=reason)
        for name in ("mean", "sample_std", "bootstrap_mean_95_ci"):
            result[name] = _summary_value("unavailable", None, reason)
        return result

    if not values:
        reason = "no_metric_values"
        result.update(status="unavailable", reason_code=reason)
        return result

    result["mean"] = _summary_value(
        "available", float(sum(values) / len(values)), None
    )
    if stratum == "overall_mixed_descriptive":
        reason = "mixed_inference_prohibited"
        result["sample_std"] = _summary_value(
            "suppressed_by_protocol", None, reason
        )
        result["bootstrap_mean_95_ci"] = _summary_value(
            "suppressed_by_protocol", None, reason
        )
        return result

    if len(values) == 1:
        reason = "insufficient_metric_pages"
        result["sample_std"] = _summary_value("unavailable", None, reason)
        result["bootstrap_mean_95_ci"] = _summary_value(
            "unavailable", None, reason
        )
        return result

    result["sample_std"] = _summary_value(
        "available", float(np.std(values, ddof=1)), None
    )
    result["bootstrap_mean_95_ci"] = _summary_value(
        "available",
        _bootstrap_mean_ci(
            values,
            metric_name=metric_name,
            samples=bootstrap_samples,
            seed=statistics_seed,
        ),
        None,
    )
    return result


def _build_stratum(
    records: Sequence[Mapping[str, Any]],
    *,
    stratum: str,
    statistics_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    valid = sum(record["status"] == "valid" for record in records)
    inference = {
        "human_gold": {"status": "not_performed"},
        "ai_silver": {"status": "exploratory_low_power"},
        "overall_mixed_descriptive": {"status": "prohibited"},
    }[stratum]
    return {
        "scope": STRATUM_SCOPES[stratum],
        "counts": {
            "expected": len(records),
            "valid": valid,
            "failed": len(records) - valid,
        },
        "aggregate_metrics": {
            name: _aggregate_metric(
                records,
                name,
                stratum=stratum,
                statistics_seed=statistics_seed,
                bootstrap_samples=bootstrap_samples,
            )
            for name in METRIC_SPECS
        },
        "inference": inference,
    }


def build_evaluation_report(
    records: Sequence[Mapping[str, Any]],
    *,
    run: Mapping[str, Any],
    expected_ids: Sequence[str],
    statistics_seed: int = 42,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """从逐页结果构造不依赖模型和文件系统的 v2 报告。"""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples 必须大于 0。")
    _validate_run_audit(
        run,
        statistics_seed=statistics_seed,
        bootstrap_samples=bootstrap_samples,
    )
    expected = sorted(str(value) for value in expected_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("expected_ids 包含重复项。")
    by_id = {str(record["sample_id"]): dict(record) for record in records}
    if len(by_id) != len(records):
        raise ValueError("逐页结果包含重复 sample ID。")
    if sorted(by_id) != expected:
        missing = sorted(set(expected) - set(by_id))
        extra = sorted(set(by_id) - set(expected))
        raise ValueError(f"逐页结果与预定 cohort 不一致：missing={missing}, extra={extra}")

    ordered_records = [by_id[sample_id] for sample_id in expected]
    for record in ordered_records:
        if record["reference_tier"] not in REFERENCE_TIERS:
            raise ValueError(f"未知 reference tier：{record['reference_tier']}")
        if record["status"] not in {"valid", "failed"}:
            raise ValueError(f"未知逐页状态：{record['status']}")
        if set(record["metrics"]) != set(METRIC_SPECS):
            raise ValueError(f"{record['sample_id']} 的 metric catalog 不完整。")

    run_payload = {**run, "hashes": dict(run["hashes"])}
    tier_records = {
        tier: [
            record for record in ordered_records
            if record["reference_tier"] == tier
        ]
        for tier in REFERENCE_TIERS
    }
    report = {
        "schema_version": "intent-evaluation/v2",
        "run": run_payload,
        "dataset": {
            "expected_ids": expected,
            "counts_by_tier": {
                tier: len(tier_records[tier]) for tier in REFERENCE_TIERS
            },
        },
        "strata": {
            "human_gold": _build_stratum(
                tier_records["human_gold"],
                stratum="human_gold",
                statistics_seed=statistics_seed,
                bootstrap_samples=bootstrap_samples,
            ),
            "ai_silver": _build_stratum(
                tier_records["ai_silver"],
                stratum="ai_silver",
                statistics_seed=statistics_seed,
                bootstrap_samples=bootstrap_samples,
            ),
            "overall_mixed_descriptive": _build_stratum(
                ordered_records,
                stratum="overall_mixed_descriptive",
                statistics_seed=statistics_seed,
                bootstrap_samples=bootstrap_samples,
            ),
        },
        "per_sample": {sample_id: by_id[sample_id] for sample_id in expected},
        "failures": [
            by_id[sample_id]
            for sample_id in expected
            if by_id[sample_id]["status"] == "failed"
        ],
    }
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    return report
