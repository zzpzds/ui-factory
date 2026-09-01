from copy import deepcopy
import json

import pytest

from scripts.compare_intent_results import (
    _compare_legacy_results,
    _comparison_mode,
    compare_reference_reports,
)
from src.design_intent.evaluation import (
    build_evaluation_report,
    make_failed_sample_result,
    make_valid_sample_result,
)
from src.design_intent.metrics import METRIC_SPECS


def _values(default: float) -> dict[str, float]:
    return {name: default for name in METRIC_SPECS}


def _run(*, checkpoint_hash: str = "e" * 64) -> dict:
    return {
        "checkpoint": "outputs/example/best.pt",
        "split": "test",
        "reference_package": "data/annotations/intent_gold_v1",
        "statistics_seed": 42,
        "bootstrap_samples": 50,
        "hashes": {
            "assignment": f"sha256:{'a' * 64}",
            "selection_manifest": f"sha256:{'b' * 64}",
            "ai_reference_manifest": f"sha256:{'c' * 64}",
            "source_split": f"sha256:{'d' * 64}",
            "checkpoint": f"sha256:{checkpoint_hash}",
        },
    }


def _report(records, *, checkpoint_hash: str = "e" * 64):
    return build_evaluation_report(
        records,
        run=_run(checkpoint_hash=checkpoint_hash),
        expected_ids=sorted(record["sample_id"] for record in records),
        statistics_seed=42,
        bootstrap_samples=50,
    )


def test_v2_comparison_rejects_non_silver_inference():
    report = _report(
        [make_valid_sample_result("a1", "ai_silver", _values(0.5))]
    )

    with pytest.raises(ValueError, match="ai_silver"):
        compare_reference_reports(report, report, tier="human_gold")
    with pytest.raises(ValueError, match="ai_silver"):
        compare_reference_reports(
            report, report, tier="overall_mixed_descriptive"
        )


def test_v2_comparison_fails_when_prediction_valid_cohorts_differ():
    method = _report(
        [
            make_valid_sample_result("a1", "ai_silver", _values(0.5)),
            make_valid_sample_result("a2", "ai_silver", _values(0.5)),
        ]
    )
    baseline = _report(
        [
            make_valid_sample_result("a1", "ai_silver", _values(0.5)),
            make_failed_sample_result(
                "a2",
                "ai_silver",
                stage="decode",
                error_type="ValueError",
                error_summary="bad",
            ),
        ]
    )

    with pytest.raises(ValueError, match="prediction-valid cohort"):
        compare_reference_reports(method, baseline)


def test_metric_cohort_mismatch_blocks_only_that_metric():
    method_values = _values(0.6)
    method_values["layout_mode_macro_f1"] = None
    method_values["layout_match_coverage"] = 0.0
    method = _report(
        [make_valid_sample_result("a1", "ai_silver", method_values)]
    )
    baseline = _report(
        [make_valid_sample_result("a1", "ai_silver", _values(0.4))]
    )

    comparison = compare_reference_reports(method, baseline)

    layout = comparison["metrics"]["layout_mode_macro_f1"]
    assert layout["status"] == "unavailable"
    assert layout["reason_code"] == "metric_cohort_mismatch"
    assert layout["holm_input_p"] == 1.0
    assert comparison["metrics"]["leaf_f1"]["status"] == "available"


def test_single_paired_page_keeps_only_descriptive_improvement():
    method_values = _values(0.6)
    baseline_values = _values(0.4)
    method_values["tree_normalized_edit_distance"] = 0.2
    baseline_values["tree_normalized_edit_distance"] = 0.5
    method = _report(
        [make_valid_sample_result("a1", "ai_silver", method_values)]
    )
    baseline = _report(
        [make_valid_sample_result("a1", "ai_silver", baseline_values)]
    )

    comparison = compare_reference_reports(method, baseline)

    leaf = comparison["metrics"]["leaf_f1"]
    assert leaf["n_paired"] == 1
    assert leaf["mean_improvement"] == {
        "status": "available",
        "value": pytest.approx(0.2),
        "reason_code": None,
    }
    for key in ("bootstrap_mean_95_ci", "cohen_dz", "permutation_p"):
        assert leaf[key]["status"] == "unavailable"
        assert leaf[key]["reason_code"] == "insufficient_paired_metric_pages"
    assert leaf["holm_input_p"] == 1.0
    assert comparison["metrics"]["tree_normalized_edit_distance"][
        "mean_improvement"
    ]["value"] == pytest.approx(0.3)


def test_empty_metric_cohort_and_zero_variance_are_json_safe():
    method_one = _values(0.6)
    baseline_one = _values(0.4)
    method_two = _values(0.6)
    baseline_two = _values(0.4)
    for values in (method_one, baseline_one, method_two, baseline_two):
        values["layout_mode_macro_f1"] = None
        values["layout_match_coverage"] = 0.0
    method = _report(
        [
            make_valid_sample_result("a1", "ai_silver", method_one),
            make_valid_sample_result("a2", "ai_silver", method_two),
        ]
    )
    baseline = _report(
        [
            make_valid_sample_result("a1", "ai_silver", baseline_one),
            make_valid_sample_result("a2", "ai_silver", baseline_two),
        ]
    )

    comparison = compare_reference_reports(
        method,
        baseline,
        bootstrap_samples=100,
        permutation_samples=100,
        seed=9,
    )

    layout = comparison["metrics"]["layout_mode_macro_f1"]
    assert layout["status"] == "unavailable"
    assert layout["reason_code"] == "no_paired_metric_pages"
    leaf = comparison["metrics"]["leaf_f1"]
    assert leaf["bootstrap_mean_95_ci"]["status"] == "available"
    assert leaf["permutation_p"]["status"] == "available"
    assert leaf["cohen_dz"] == {
        "status": "unavailable",
        "value": None,
        "reason_code": "zero_paired_difference_variance",
    }
    json.dumps(comparison, ensure_ascii=False, allow_nan=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["dataset"].__setitem__(
                "expected_ids", ["a2", "a1"]
            ),
            "expected_ids",
        ),
        (
            lambda report: report["dataset"].__setitem__(
                "expected_ids", ["a1", "a1", "a2"]
            ),
            "expected_ids",
        ),
        (
            lambda report: report["per_sample"].pop("a2"),
            "per_sample",
        ),
        (
            lambda report: report["dataset"]["counts_by_tier"].__setitem__(
                "ai_silver", 1
            ),
            "counts_by_tier",
        ),
        (
            lambda report: report.__setitem__("failures", []),
            "failures",
        ),
        (
            lambda report: report["per_sample"]["a1"].__setitem__(
                "reference_tier", "unknown"
            ),
            "reference_tier",
        ),
        (
            lambda report: report["per_sample"]["a1"].__setitem__(
                "status", "unknown"
            ),
            "status",
        ),
        (
            lambda report: report["per_sample"]["a1"]["metrics"].pop(
                "leaf_f1"
            ),
            "metric catalog",
        ),
    ],
)
def test_v2_comparison_rejects_internally_inconsistent_reports(
    mutation, message
):
    records = [
        make_valid_sample_result("a1", "ai_silver", _values(0.5)),
        make_failed_sample_result(
            "a2",
            "ai_silver",
            stage="decode",
            error_type="ValueError",
            error_summary="bad",
        ),
    ]
    method = _report(records)
    baseline = deepcopy(method)
    mutation(method)

    with pytest.raises(ValueError, match=message):
        compare_reference_reports(method, baseline)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["strata"]["ai_silver"][
            "aggregate_metrics"
        ].__setitem__("leaf_f1", {}),
        lambda report: report["strata"]["ai_silver"]["aggregate_metrics"][
            "leaf_f1"
        ]["counts"].__setitem__("n_available", 0),
        lambda report: report["strata"]["ai_silver"]["aggregate_metrics"][
            "leaf_f1"
        ]["mean"].__setitem__("value", 0.123),
        lambda report: report["strata"]["ai_silver"]["aggregate_metrics"][
            "leaf_f1"
        ].__setitem__("status", "unavailable"),
    ],
)
def test_v2_comparison_rejects_corrupt_aggregate_payload(mutation):
    method = _report(
        [
            make_valid_sample_result("a1", "ai_silver", _values(0.4)),
            make_valid_sample_result("a2", "ai_silver", _values(0.6)),
        ]
    )
    baseline = deepcopy(method)
    mutation(method)

    with pytest.raises(ValueError, match="aggregate_metrics"):
        compare_reference_reports(method, baseline)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["run"].pop("checkpoint"),
            "run.checkpoint",
        ),
        (
            lambda report: report["run"]["hashes"].__setitem__(
                "assignment", "sha256:not-a-digest"
            ),
            "run.hashes.assignment",
        ),
        (
            lambda report: report["run"]["hashes"].pop("source_split"),
            "run.hashes.source_split",
        ),
    ],
)
def test_v2_comparison_rejects_incomplete_run_audit(mutation, message):
    method = _report(
        [make_valid_sample_result("a1", "ai_silver", _values(0.5))]
    )
    baseline = deepcopy(method)
    mutation(method)

    with pytest.raises(ValueError, match=message):
        compare_reference_reports(method, baseline)


def test_v2_comparison_requires_identical_data_manifest_hashes_only():
    method = _report(
        [make_valid_sample_result("a1", "ai_silver", _values(0.6))],
        checkpoint_hash="1" * 64,
    )
    baseline = _report(
        [make_valid_sample_result("a1", "ai_silver", _values(0.4))],
        checkpoint_hash="2" * 64,
    )

    comparison = compare_reference_reports(method, baseline)
    assert comparison["metrics"]["leaf_f1"]["status"] == "available"

    baseline["run"]["hashes"]["assignment"] = f"sha256:{'f' * 64}"
    with pytest.raises(ValueError, match="assignment"):
        compare_reference_reports(method, baseline)


def test_v2_comparison_cannot_jointly_drop_a_declared_failed_page():
    records = [
        make_valid_sample_result("a1", "ai_silver", _values(0.5)),
        make_failed_sample_result(
            "a2",
            "ai_silver",
            stage="decode",
            error_type="ValueError",
            error_summary="bad",
        ),
    ]
    method = _report(records)
    baseline = deepcopy(method)
    for report in (method, baseline):
        report["per_sample"].pop("a2")
        report["failures"] = []

    with pytest.raises(ValueError, match="per_sample"):
        compare_reference_reports(method, baseline)


def test_v2_dz_remains_available_for_nonzero_near_constant_differences():
    method_one = _values(0.5)
    method_two = _values(0.5)
    baseline_one = _values(0.3)
    baseline_two = _values(0.3)
    method_two["leaf_f1"] += 1e-13
    method = _report(
        [
            make_valid_sample_result("a1", "ai_silver", method_one),
            make_valid_sample_result("a2", "ai_silver", method_two),
        ]
    )
    baseline = _report(
        [
            make_valid_sample_result("a1", "ai_silver", baseline_one),
            make_valid_sample_result("a2", "ai_silver", baseline_two),
        ]
    )

    comparison = compare_reference_reports(
        method,
        baseline,
        bootstrap_samples=20,
        permutation_samples=20,
    )

    assert comparison["metrics"]["leaf_f1"]["cohen_dz"]["status"] == (
        "available"
    )


def test_legacy_comparison_uses_finite_metric_specific_paired_cohorts():
    method = {
        "per_sample": {
            "a1": {"layout": None, "leaf": 0.9},
            "a2": {"layout": 0.6, "leaf": None},
            "a3": {"layout": 0.8, "leaf": float("nan")},
        }
    }
    baseline = {
        "per_sample": {
            "a1": {"layout": 0.2, "leaf": 0.7},
            "a2": {"layout": 0.4, "leaf": 0.6},
            "a3": {"layout": 0.5, "leaf": 0.5},
        }
    }

    comparison = _compare_legacy_results(
        method,
        baseline,
        bootstrap_samples=20,
        permutation_samples=20,
        seed=7,
    )

    assert comparison["metrics"]["layout"]["status"] == "available"
    assert comparison["metrics"]["layout"]["paired_ids"] == ["a2", "a3"]
    assert comparison["metrics"]["layout"]["n_paired"] == 2
    assert comparison["metrics"]["leaf"] == {
        "status": "unavailable",
        "reason_code": "insufficient_paired_metric_pages",
        "paired_ids": ["a1"],
        "n_paired": 1,
        "method_mean": None,
        "baseline_mean": None,
        "paired_difference": None,
        "bootstrap_95_ci": None,
        "cohen_dz": None,
        "permutation_p": None,
        "holm_adjusted_p": None,
    }


@pytest.mark.parametrize(
    ("method", "baseline"),
    [
        (
            {"schema_version": "intent-evaluation/v3", "per_sample": {}},
            {"schema_version": "intent-evaluation/v3", "per_sample": {}},
        ),
        (
            {"per_sample": {"a1": {"metrics": {"leaf_f1": 0.5}}}},
            {"per_sample": {"a1": {"metrics": {"leaf_f1": 0.4}}}},
        ),
        (
            {"schema_version": "intent-evaluation/v2", "per_sample": {}},
            {"per_sample": {}},
        ),
    ],
)
def test_comparison_mode_rejects_unknown_or_v2_like_unversioned_reports(
    method,
    baseline,
):
    with pytest.raises(ValueError, match="schema_version|legacy"):
        _comparison_mode(method, baseline)


def test_comparison_mode_accepts_only_flat_unversioned_legacy_reports():
    method = {"per_sample": {"a1": {"leaf_f1": 0.5, "layout": None}}}
    baseline = {"per_sample": {"a1": {"leaf_f1": 0.4, "layout": 0.2}}}

    assert _comparison_mode(method, baseline) == "legacy"
