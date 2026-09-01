from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts import evaluate_intent as evaluation_cli
from scripts import evaluate_intent_baseline as baseline_cli
from src.design_intent import metrics as metric_module
from src.design_intent.evaluation import (
    build_evaluation_report,
    make_failed_sample_result,
    make_valid_sample_result,
)
from src.design_intent.metrics import compute_intent_metrics
from src.design_intent.weak_supervision import build_weak_intent
from tests.test_design_intent import make_page_graph


def _cli_function(name):
    assert hasattr(evaluation_cli, name)
    return getattr(evaluation_cli, name)


def test_metric_catalog_matches_full_core_metric_output():
    graph = make_page_graph()
    intent = build_weak_intent(graph)

    metrics = compute_intent_metrics(intent, deepcopy(intent), graph)

    assert hasattr(metric_module, "METRIC_SPECS")
    assert set(metrics) == set(metric_module.METRIC_SPECS)


def test_core_metrics_can_skip_unavailable_token_supervision():
    graph = make_page_graph()
    intent = build_weak_intent(graph)

    metrics = compute_intent_metrics(
        intent,
        deepcopy(intent),
        graph,
        include_token_metrics=False,
    )

    assert not any(name.startswith("token_") for name in metrics)


def test_layout_metrics_are_unavailable_without_a_matched_reference_group():
    graph = make_page_graph()
    gold = build_weak_intent(graph)
    predicted = deepcopy(gold)
    predicted.layouts = []

    metrics = compute_intent_metrics(predicted, gold, graph)

    assert metrics["layout_match_coverage"] == 0.0
    assert metrics["layout_mode_accuracy"] is None
    assert metrics["layout_mode_macro_f1"] is None
    assert metrics["gap_normalized_mae"] is None
    assert metrics["padding_normalized_mae"] is None


def test_layout_coverage_is_unavailable_without_reference_layout_labels():
    graph = make_page_graph()
    gold = build_weak_intent(graph)
    gold.layouts = []
    predicted = deepcopy(gold)

    metrics = compute_intent_metrics(predicted, gold, graph)

    assert metrics["layout_match_coverage"] is None
    assert metrics["layout_mode_accuracy"] is None
    assert metrics["layout_mode_macro_f1"] is None
    assert metrics["gap_normalized_mae"] is None
    assert metrics["padding_normalized_mae"] is None


def _metric_values(value: float) -> dict[str, float]:
    return {name: value for name in metric_module.METRIC_SPECS}


def _valid_run(**overrides):
    run = {
        "checkpoint": "outputs/example/best.pt",
        "split": "test",
        "reference_package": "data/annotations/intent_gold_v1",
        "statistics_seed": 42,
        "bootstrap_samples": 100,
        "hashes": {
            "assignment": f"sha256:{'a' * 64}",
            "selection_manifest": f"sha256:{'b' * 64}",
            "ai_reference_manifest": f"sha256:{'c' * 64}",
            "source_split": f"sha256:{'d' * 64}",
            "checkpoint": f"sha256:{'e' * 64}",
        },
    }
    run.update(overrides)
    return run


def test_human_sample_fills_catalog_and_marks_token_metrics_unavailable():
    values = _metric_values(0.75)
    values.pop("token_bcubed_f1")
    values.pop("token_bcubed_precision")
    values.pop("token_bcubed_recall")
    values.pop("token_coverage")

    sample = make_valid_sample_result("h1", "human_gold", values)

    assert set(sample["metrics"]) == set(metric_module.METRIC_SPECS)
    assert sample["metrics"]["leaf_f1"]["value"] == 0.75
    assert sample["metrics"]["token_bcubed_f1"] == {
        "status": "unavailable",
        "value": None,
        "scope": "exploratory_human_reference",
        "reason_code": "human_gold_token_labels_not_collected",
    }


def test_sample_result_explains_both_layout_unavailability_cases():
    no_reference = _metric_values(0.5)
    no_reference["layout_match_coverage"] = None
    no_reference["layout_mode_macro_f1"] = None
    no_match = dict(no_reference)
    no_match["layout_match_coverage"] = 0.0

    without_reference = make_valid_sample_result(
        "a1", "ai_silver", no_reference
    )
    without_match = make_valid_sample_result("a2", "ai_silver", no_match)

    assert without_reference["metrics"]["layout_mode_macro_f1"][
        "reason_code"
    ] == "reference_layout_labels_absent"
    assert without_match["metrics"]["layout_mode_macro_f1"][
        "reason_code"
    ] == "no_matched_layout"
    assert without_match["metrics"]["layout_match_coverage"]["value"] == 0.0


def test_tiered_report_preserves_failures_and_metric_specific_denominators():
    human_one = _metric_values(0.2)
    human_two = _metric_values(0.4)
    ai_one = _metric_values(0.6)
    ai_two = _metric_values(0.8)
    ai_two["layout_mode_macro_f1"] = None
    ai_two["layout_match_coverage"] = 0.0
    records = [
        make_valid_sample_result("h1", "human_gold", human_one),
        make_valid_sample_result("h2", "human_gold", human_two),
        make_valid_sample_result("a1", "ai_silver", ai_one),
        make_valid_sample_result("a2", "ai_silver", ai_two),
        make_failed_sample_result(
            "a3",
            "ai_silver",
            stage="decode",
            error_type="ValueError",
            error_summary="无法解码",
        ),
    ]

    report = build_evaluation_report(
        records,
        run=_valid_run(),
        expected_ids=["a1", "a2", "a3", "h1", "h2"],
        statistics_seed=42,
        bootstrap_samples=100,
    )

    assert report["schema_version"] == "intent-evaluation/v2"
    assert list(report["per_sample"]) == ["a1", "a2", "a3", "h1", "h2"]
    assert report["dataset"]["counts_by_tier"] == {
        "human_gold": 2,
        "ai_silver": 3,
    }
    assert report["strata"]["ai_silver"]["counts"] == {
        "expected": 3,
        "valid": 2,
        "failed": 1,
    }
    layout = report["strata"]["ai_silver"]["aggregate_metrics"][
        "layout_mode_macro_f1"
    ]
    assert layout["counts"] == {
        "n_expected_pages": 3,
        "n_valid_predictions": 2,
        "n_failed_predictions": 1,
        "n_available": 1,
        "n_unavailable": 1,
        "unavailable_reason_counts": {"no_matched_layout": 1},
    }
    assert layout["mean"]["value"] == 0.6
    assert layout["sample_std"]["status"] == "unavailable"
    assert layout["bootstrap_mean_95_ci"]["status"] == "unavailable"
    assert report["failures"] == [report["per_sample"]["a3"]]
    json.dumps(report, ensure_ascii=False, allow_nan=False)


def test_report_applies_human_suppression_and_mixed_descriptive_policy():
    records = [
        make_valid_sample_result("h1", "human_gold", _metric_values(0.2)),
        make_valid_sample_result("h2", "human_gold", _metric_values(0.4)),
        make_valid_sample_result("a1", "ai_silver", _metric_values(0.6)),
        make_valid_sample_result("a2", "ai_silver", _metric_values(0.8)),
    ]

    report = build_evaluation_report(
        records,
        run=_valid_run(statistics_seed=7),
        expected_ids=["a1", "a2", "h1", "h2"],
        statistics_seed=7,
        bootstrap_samples=100,
    )

    human_leaf = report["strata"]["human_gold"]["aggregate_metrics"]["leaf_f1"]
    assert human_leaf["status"] == "suppressed_by_protocol"
    assert human_leaf["mean"]["status"] == "suppressed_by_protocol"
    mixed_leaf = report["strata"]["overall_mixed_descriptive"][
        "aggregate_metrics"
    ]["leaf_f1"]
    assert mixed_leaf["mean"]["value"] == pytest.approx(0.5)
    assert mixed_leaf["sample_std"]["status"] == "suppressed_by_protocol"
    assert mixed_leaf["bootstrap_mean_95_ci"]["status"] == (
        "suppressed_by_protocol"
    )
    mixed_token = report["strata"]["overall_mixed_descriptive"][
        "aggregate_metrics"
    ]["token_bcubed_f1"]
    assert mixed_token["status"] == "unavailable"
    assert mixed_token["reason_code"] == (
        "cross_tier_token_truth_spaces_incomparable"
    )
    human_token = report["strata"]["human_gold"]["aggregate_metrics"][
        "token_bcubed_f1"
    ]
    silver_token = report["strata"]["ai_silver"]["aggregate_metrics"][
        "token_bcubed_f1"
    ]
    assert "interpretation" not in human_token
    assert "interpretation" not in mixed_token
    assert silver_token["interpretation"] == "ai_computed_style_proxy_only"


def test_ai_bootstrap_is_deterministic_for_a_fixed_seed():
    records = [
        make_valid_sample_result("a1", "ai_silver", _metric_values(0.1)),
        make_valid_sample_result("a2", "ai_silver", _metric_values(0.9)),
    ]
    kwargs = {
        "run": _valid_run(statistics_seed=123, bootstrap_samples=200),
        "expected_ids": ["a1", "a2"],
        "statistics_seed": 123,
        "bootstrap_samples": 200,
    }

    first = build_evaluation_report(records, **kwargs)
    second = build_evaluation_report(records, **kwargs)

    first_ci = first["strata"]["ai_silver"]["aggregate_metrics"]["leaf_f1"][
        "bootstrap_mean_95_ci"
    ]
    second_ci = second["strata"]["ai_silver"]["aggregate_metrics"]["leaf_f1"][
        "bootstrap_mean_95_ci"
    ]
    assert first_ci == second_ci


@pytest.mark.parametrize(
    "missing_field",
    [
        "checkpoint",
        "split",
        "reference_package",
        "statistics_seed",
        "bootstrap_samples",
        "hashes",
    ],
)
def test_report_rejects_missing_run_audit_fields(missing_field):
    run = _valid_run()
    run.pop(missing_field)

    with pytest.raises(ValueError, match=missing_field):
        build_evaluation_report(
            [],
            run=run,
            expected_ids=[],
            statistics_seed=42,
            bootstrap_samples=100,
        )


@pytest.mark.parametrize(
    "hash_name",
    [
        "assignment",
        "selection_manifest",
        "ai_reference_manifest",
        "source_split",
        "checkpoint",
    ],
)
def test_report_rejects_missing_required_run_hashes(hash_name):
    run = _valid_run()
    run["hashes"].pop(hash_name)

    with pytest.raises(ValueError, match=hash_name):
        build_evaluation_report(
            [],
            run=run,
            expected_ids=[],
            statistics_seed=42,
            bootstrap_samples=100,
        )


@pytest.mark.parametrize(
    "invalid_hash",
    ["abc", "sha256:abc", f"md5:{'a' * 64}", f"sha256:{'g' * 64}"],
)
def test_report_rejects_noncanonical_sha256_hashes(invalid_hash):
    run = _valid_run()
    run["hashes"]["checkpoint"] = invalid_hash

    with pytest.raises(ValueError, match="checkpoint"):
        build_evaluation_report(
            [],
            run=run,
            expected_ids=[],
            statistics_seed=42,
            bootstrap_samples=100,
        )


def test_report_rejects_run_statistics_that_disagree_with_aggregation():
    with pytest.raises(ValueError, match="statistics_seed"):
        build_evaluation_report(
            [],
            run=_valid_run(statistics_seed=7),
            expected_ids=[],
            statistics_seed=42,
            bootstrap_samples=100,
        )


def test_report_rejects_non_finite_metric_values():
    values = _metric_values(0.5)
    values["leaf_f1"] = float("nan")

    with pytest.raises(ValueError, match="leaf_f1"):
        make_valid_sample_result("a1", "ai_silver", values)


class _FakeModel:
    def __call__(self, batch):
        return {"sample_id": batch["sample_id"][0]}


def _reference_batch(sample_id="a1", tier="ai_silver"):
    graph = make_page_graph()
    graph.sample_id = sample_id
    intent = build_weak_intent(graph)
    return {
        "sample_id": [sample_id],
        "reference_tier": [tier],
        "graph": [graph],
        "intent_ir": [intent],
    }


def test_reference_batch_evaluation_disables_human_token_metrics():
    batch = _reference_batch("h1", "human_gold")
    seen = {}

    def metric_fn(predicted, gold, graph, *, include_token_metrics):
        seen["include_token_metrics"] = include_token_metrics
        return compute_intent_metrics(
            predicted,
            gold,
            graph,
            include_token_metrics=include_token_metrics,
        )

    records = _cli_function("evaluate_reference_batches")(
        _FakeModel(),
        [batch],
        decode_fn=lambda graph, outputs: deepcopy(batch["intent_ir"][0]),
        validate_fn=lambda predicted, graph: [],
        metric_fn=metric_fn,
    )

    assert seen["include_token_metrics"] is False
    assert records[0]["status"] == "valid"
    assert records[0]["metrics"]["token_bcubed_f1"]["status"] == "unavailable"


def test_reference_batch_evaluation_captures_only_decode_and_validation_failures():
    decode_batch = _reference_batch("a1")
    validation_batch = _reference_batch("a2")

    decode_records = _cli_function("evaluate_reference_batches")(
        _FakeModel(),
        [decode_batch],
        decode_fn=lambda graph, outputs: (_ for _ in ()).throw(ValueError("bad")),
        validate_fn=lambda predicted, graph: [],
        metric_fn=compute_intent_metrics,
    )
    validation_records = _cli_function("evaluate_reference_batches")(
        _FakeModel(),
        [validation_batch],
        decode_fn=lambda graph, outputs: deepcopy(validation_batch["intent_ir"][0]),
        validate_fn=lambda predicted, graph: ["tree invalid"],
        metric_fn=compute_intent_metrics,
    )

    assert decode_records[0]["failure"]["stage"] == "decode"
    assert validation_records[0]["failure"] == {
        "stage": "ir_validation",
        "error_type": "IRValidationError",
        "error_summary": "tree invalid",
    }


def test_failed_sample_uses_error_type_when_exception_message_is_empty():
    batch = _reference_batch("a1")

    records = _cli_function("evaluate_reference_batches")(
        _FakeModel(),
        [batch],
        decode_fn=lambda graph, outputs: (_ for _ in ()).throw(ValueError()),
        validate_fn=lambda predicted, graph: [],
        metric_fn=compute_intent_metrics,
    )

    assert records[0]["failure"] == {
        "stage": "decode",
        "error_type": "ValueError",
        "error_summary": "ValueError",
    }


def test_failed_sample_rejects_empty_error_type():
    with pytest.raises(ValueError, match="error_type"):
        make_failed_sample_result(
            "a1",
            "ai_silver",
            stage="decode",
            error_type="",
            error_summary="bad",
        )


def test_failed_token_proxy_interpretation_is_limited_to_ai_silver():
    human = make_failed_sample_result(
        "h1",
        "human_gold",
        stage="decode",
        error_type="ValueError",
        error_summary="bad",
    )
    silver = make_failed_sample_result(
        "a1",
        "ai_silver",
        stage="decode",
        error_type="ValueError",
        error_summary="bad",
    )

    assert "interpretation" not in human["metrics"]["token_bcubed_f1"]
    assert silver["metrics"]["token_bcubed_f1"]["interpretation"] == (
        "ai_computed_style_proxy_only"
    )


def test_reference_batch_evaluation_propagates_metric_errors():
    batch = _reference_batch("a1")

    with pytest.raises(RuntimeError, match="metric broke"):
        _cli_function("evaluate_reference_batches")(
            _FakeModel(),
            [batch],
            decode_fn=lambda graph, outputs: deepcopy(batch["intent_ir"][0]),
            validate_fn=lambda predicted, graph: [],
            metric_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("metric broke")
            ),
        )


def test_test_unseal_uses_exclusive_create_and_reuses_first_payload(tmp_path):
    create_or_read = _cli_function("create_or_read_test_unseal")
    first_payload = {
        "unsealed_at": "2026-09-01T10:00:00+08:00",
        "git_commit": "abc",
        "assignment_hash": f"sha256:{'a' * 64}",
        "checkpoint_hash": f"sha256:{'b' * 64}",
    }
    second_payload = {
        **first_payload,
        "unsealed_at": "2026-09-01T11:00:00+08:00",
    }

    first = create_or_read(tmp_path, first_payload)
    second = create_or_read(tmp_path, second_payload)

    assert first == first_payload
    assert second == first_payload
    assert json.loads((tmp_path / "test_unseal.json").read_text()) == first_payload


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {},
        {
            "unsealed_at": "not-a-time",
            "git_commit": "abc",
            "assignment_hash": f"sha256:{'a' * 64}",
            "checkpoint_hash": f"sha256:{'b' * 64}",
        },
    ],
)
def test_test_unseal_rejects_malformed_existing_record(
    tmp_path,
    invalid_payload,
):
    (tmp_path / "test_unseal.json").write_text(
        json.dumps(invalid_payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="test unseal"):
        _cli_function("create_or_read_test_unseal")(
            tmp_path,
            {
                "unsealed_at": "2026-09-01T10:00:00+08:00",
                "git_commit": "abc",
                "assignment_hash": f"sha256:{'a' * 64}",
                "checkpoint_hash": f"sha256:{'b' * 64}",
            },
        )


def test_formal_reference_loads_model_then_unseals_before_dataset(
    tmp_path, monkeypatch
):
    package_dir = tmp_path / "formal_reference"
    package_dir.mkdir()
    assignment_path = package_dir / "assignment.json"
    assignment_path.write_text('{"samples": []}\n', encoding="utf-8")
    (package_dir / "selection_manifest.json").write_text(
        '{"samples": []}\n', encoding="utf-8"
    )
    (package_dir / "ai_reference_manifest.json").write_text(
        '{"samples": []}\n', encoding="utf-8"
    )
    source_split_path = tmp_path / "split.json"
    source_split_path.write_text('{"splits": {}}\n', encoding="utf-8")
    (tmp_path / "processed").mkdir()
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    output_path = tmp_path / "arbitrary-results" / "evaluation.json"
    fixed_unseal_path = tmp_path / "fixed-audit" / "test_unseal.json"
    events = []

    class FakeReferenceDataset:
        def __init__(self, **kwargs):
            if kwargs.get("_metadata_only"):
                events.append("metadata")
                assert not fixed_unseal_path.exists()
            else:
                events.append("dataset")
                assert fixed_unseal_path.exists()
            self.samples = []
            self.manifest_hashes = {
                "assignment": evaluation_cli._sha256(assignment_path),
                "selection_manifest": "b" * 64,
                "ai_reference_manifest": "c" * 64,
                "source_split": "d" * 64,
            }

    def fake_load_model(*args):
        events.append("model")
        assert not fixed_unseal_path.exists()
        return _FakeModel()

    checkpoint = {
        "config": {
            "data": {
                "data_dir": str(tmp_path / "processed"),
                "max_nodes": 16,
                "source_split_manifest": str(source_split_path),
                "verify_hashes": True,
            },
            "model": {},
        },
        "model": {},
    }
    monkeypatch.setattr(
        evaluation_cli, "FORMAL_REFERENCE_PACKAGE", package_dir.resolve()
    )
    monkeypatch.setattr(
        evaluation_cli, "FORMAL_TEST_UNSEAL_PATH", fixed_unseal_path
    )
    monkeypatch.setattr(evaluation_cli, "ReferenceIntentDataset", FakeReferenceDataset)
    monkeypatch.setattr(evaluation_cli, "DataLoader", lambda dataset, **kwargs: [])
    monkeypatch.setattr(evaluation_cli, "_load_model", fake_load_model)

    result = evaluation_cli._evaluate_reference(
        checkpoint,
        checkpoint_path=checkpoint_path,
        package_dir=package_dir,
        split="test",
        statistics_seed=42,
        bootstrap_samples=20,
    )

    assert events == ["model", "metadata", "dataset"]
    assert fixed_unseal_path.exists()
    assert not (output_path.parent / "test_unseal.json").exists()
    audit = json.loads(fixed_unseal_path.read_text(encoding="utf-8"))
    assert audit["assignment_hash"] == evaluation_cli._sha256_tag(assignment_path)
    assert audit["checkpoint_hash"] == evaluation_cli._sha256_tag(checkpoint_path)
    assert result["run"]["test_unseal"]["path"] == str(fixed_unseal_path)


def test_formal_reference_model_preflight_failure_does_not_unseal(
    tmp_path, monkeypatch
):
    package_dir = tmp_path / "formal_reference"
    package_dir.mkdir()
    for name in (
        "assignment.json",
        "selection_manifest.json",
        "ai_reference_manifest.json",
    ):
        (package_dir / name).write_text("{}\n", encoding="utf-8")
    source_split_path = tmp_path / "split.json"
    source_split_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "processed").mkdir()
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    fixed_unseal_path = tmp_path / "fixed-audit" / "test_unseal.json"
    checkpoint = {
        "config": {
            "data": {
                "data_dir": str(tmp_path / "processed"),
                "max_nodes": 16,
                "source_split_manifest": str(source_split_path),
            },
            "model": {},
        },
        "model": {},
    }
    monkeypatch.setattr(
        evaluation_cli, "FORMAL_REFERENCE_PACKAGE", package_dir.resolve()
    )
    monkeypatch.setattr(
        evaluation_cli, "FORMAL_TEST_UNSEAL_PATH", fixed_unseal_path
    )
    monkeypatch.setattr(
        evaluation_cli,
        "_load_model",
        lambda *args: (_ for _ in ()).throw(RuntimeError("bad model")),
    )

    with pytest.raises(RuntimeError, match="bad model"):
        evaluation_cli._evaluate_reference(
            checkpoint,
            checkpoint_path=checkpoint_path,
            package_dir=package_dir,
            split="test",
            statistics_seed=42,
            bootstrap_samples=20,
        )

    assert not fixed_unseal_path.exists()


def test_formal_reference_manifest_preflight_failure_does_not_unseal(
    tmp_path, monkeypatch
):
    package_dir = tmp_path / "formal_reference"
    package_dir.mkdir()
    (package_dir / "assignment.json").write_text("{}\n", encoding="utf-8")
    (package_dir / "selection_manifest.json").write_text(
        "not-json\n", encoding="utf-8"
    )
    (package_dir / "ai_reference_manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )
    source_split_path = tmp_path / "split.json"
    source_split_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "processed").mkdir()
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    fixed_unseal_path = tmp_path / "fixed-audit" / "test_unseal.json"
    checkpoint = {
        "config": {
            "data": {
                "data_dir": str(tmp_path / "processed"),
                "max_nodes": 16,
                "source_split_manifest": str(source_split_path),
            },
            "model": {},
        },
        "model": {},
    }
    monkeypatch.setattr(
        evaluation_cli, "FORMAL_REFERENCE_PACKAGE", package_dir.resolve()
    )
    monkeypatch.setattr(
        evaluation_cli, "FORMAL_TEST_UNSEAL_PATH", fixed_unseal_path
    )
    monkeypatch.setattr(evaluation_cli, "_load_model", lambda *args: _FakeModel())
    monkeypatch.setattr(
        evaluation_cli,
        "ReferenceIntentDataset",
        lambda **kwargs: pytest.fail("manifest 预检失败后不应构造 Dataset"),
    )

    with pytest.raises(json.JSONDecodeError):
        evaluation_cli._evaluate_reference(
            checkpoint,
            checkpoint_path=checkpoint_path,
            package_dir=package_dir,
            split="test",
            statistics_seed=42,
            bootstrap_samples=20,
        )

    assert not fixed_unseal_path.exists()


def test_formal_reference_semantic_preflight_failure_does_not_unseal(
    tmp_path, monkeypatch
):
    package_dir = tmp_path / "formal_reference"
    package_dir.mkdir()
    for name in (
        "assignment.json",
        "selection_manifest.json",
        "ai_reference_manifest.json",
    ):
        (package_dir / name).write_text("{}\n", encoding="utf-8")
    source_split_path = tmp_path / "split.json"
    source_split_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "processed").mkdir()
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    fixed_unseal_path = tmp_path / "fixed-audit" / "test_unseal.json"
    events = []

    def fake_dataset(**kwargs):
        if kwargs.get("_metadata_only"):
            events.append("metadata")
            raise ValueError("manifest 语义冲突")
        pytest.fail("语义预检失败后不应构造完整 Dataset")

    checkpoint = {
        "config": {
            "data": {
                "data_dir": str(tmp_path / "processed"),
                "max_nodes": 16,
                "source_split_manifest": str(source_split_path),
            },
            "model": {},
        },
        "model": {},
    }
    monkeypatch.setattr(
        evaluation_cli, "FORMAL_REFERENCE_PACKAGE", package_dir.resolve()
    )
    monkeypatch.setattr(
        evaluation_cli, "FORMAL_TEST_UNSEAL_PATH", fixed_unseal_path
    )
    monkeypatch.setattr(evaluation_cli, "ReferenceIntentDataset", fake_dataset)
    monkeypatch.setattr(
        evaluation_cli,
        "_load_model",
        lambda *args: events.append("model") or _FakeModel(),
    )

    with pytest.raises(ValueError, match="语义冲突"):
        evaluation_cli._evaluate_reference(
            checkpoint,
            checkpoint_path=checkpoint_path,
            package_dir=package_dir,
            split="test",
            statistics_seed=42,
            bootstrap_samples=20,
        )

    assert events == ["model", "metadata"]
    assert not fixed_unseal_path.exists()


def test_reference_rejects_manifest_change_after_metadata_preflight(
    tmp_path, monkeypatch
):
    package_dir = tmp_path / "reference"
    package_dir.mkdir()
    for name in (
        "assignment.json",
        "selection_manifest.json",
        "ai_reference_manifest.json",
    ):
        (package_dir / name).write_text("{}\n", encoding="utf-8")
    source_split_path = tmp_path / "split.json"
    source_split_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "processed").mkdir()
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    calls = []

    class FakeReferenceDataset:
        def __init__(self, **kwargs):
            metadata_only = bool(kwargs.get("_metadata_only"))
            calls.append(metadata_only)
            suffix = "a" if metadata_only else "b"
            self.samples = []
            self.manifest_hashes = {
                "assignment": suffix * 64,
                "selection_manifest": "c" * 64,
                "ai_reference_manifest": "d" * 64,
                "source_split": "e" * 64,
            }

    checkpoint = {
        "config": {
            "data": {
                "data_dir": str(tmp_path / "processed"),
                "max_nodes": 16,
                "source_split_manifest": str(source_split_path),
            },
            "model": {},
        },
        "model": {},
    }
    monkeypatch.setattr(evaluation_cli, "ReferenceIntentDataset", FakeReferenceDataset)
    monkeypatch.setattr(evaluation_cli, "_load_model", lambda *args: _FakeModel())

    with pytest.raises(RuntimeError, match="预检后发生变化"):
        evaluation_cli._evaluate_reference(
            checkpoint,
            checkpoint_path=checkpoint_path,
            package_dir=package_dir,
            split="validation",
            statistics_seed=42,
            bootstrap_samples=20,
        )

    assert calls == [True, False]


def test_formal_reference_invalid_bootstrap_count_fails_before_unseal(
    tmp_path, monkeypatch
):
    package_dir = tmp_path / "formal_reference"
    package_dir.mkdir()
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    fixed_unseal_path = tmp_path / "fixed-audit" / "test_unseal.json"
    checkpoint = {"config": {"data": {}, "model": {}}, "model": {}}
    monkeypatch.setattr(
        evaluation_cli, "FORMAL_REFERENCE_PACKAGE", package_dir.resolve()
    )
    monkeypatch.setattr(
        evaluation_cli, "FORMAL_TEST_UNSEAL_PATH", fixed_unseal_path
    )

    with pytest.raises(ValueError, match="bootstrap_samples"):
        evaluation_cli._evaluate_reference(
            checkpoint,
            checkpoint_path=checkpoint_path,
            package_dir=package_dir,
            split="test",
            statistics_seed=42,
            bootstrap_samples=0,
        )

    assert not fixed_unseal_path.exists()


def test_formal_reference_test_requires_hash_verification_before_unseal(
    tmp_path, monkeypatch
):
    package_dir = tmp_path / "formal_reference"
    package_dir.mkdir()
    for name in (
        "assignment.json",
        "selection_manifest.json",
        "ai_reference_manifest.json",
    ):
        (package_dir / name).write_text("{}\n", encoding="utf-8")
    source_split_path = tmp_path / "split.json"
    source_split_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "processed").mkdir()
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    fixed_unseal_path = tmp_path / "fixed-audit" / "test_unseal.json"
    checkpoint = {
        "config": {
            "data": {
                "data_dir": str(tmp_path / "processed"),
                "max_nodes": 16,
                "source_split_manifest": str(source_split_path),
                "verify_hashes": False,
            },
            "model": {},
        },
        "model": {},
    }
    monkeypatch.setattr(
        evaluation_cli, "FORMAL_REFERENCE_PACKAGE", package_dir.resolve()
    )
    monkeypatch.setattr(
        evaluation_cli, "FORMAL_TEST_UNSEAL_PATH", fixed_unseal_path
    )
    monkeypatch.setattr(
        evaluation_cli,
        "_load_model",
        lambda *args: pytest.fail(
            "verify_hashes=false 应在模型加载和解封前失败"
        ),
    )

    with pytest.raises(ValueError, match="verify_hashes"):
        evaluation_cli._evaluate_reference(
            checkpoint,
            checkpoint_path=checkpoint_path,
            package_dir=package_dir,
            split="test",
            statistics_seed=42,
            bootstrap_samples=20,
        )

    assert not fixed_unseal_path.exists()


@pytest.mark.parametrize(
    "split_manifest_key",
    ["source_split_manifest", "split_manifest"],
)
def test_reference_cli_writes_v2_report_for_a_synthetic_fixture(
    tmp_path,
    monkeypatch,
    split_manifest_key,
):
    checkpoint_path = tmp_path / "mock.pt"
    checkpoint_path.write_bytes(b"mock checkpoint")
    package_dir = tmp_path / "reference_fixture"
    package_dir.mkdir()
    for name in (
        "assignment.json",
        "selection_manifest.json",
        "ai_reference_manifest.json",
    ):
        (package_dir / name).write_text("{}\n", encoding="utf-8")
    source_split_path = tmp_path / "split.json"
    source_split_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "processed").mkdir()
    output_path = tmp_path / "results" / "evaluation.json"
    batches = [
        _reference_batch("a1", "ai_silver"),
        _reference_batch("h1", "human_gold"),
    ]

    class FakeReferenceDataset:
        def __init__(self, **kwargs):
            self.samples = [
                SimpleNamespace(sample_id="a1"),
                SimpleNamespace(sample_id="h1"),
            ]
            self.manifest_hashes = {
                "assignment": "a" * 64,
                "selection_manifest": "b" * 64,
                "ai_reference_manifest": "c" * 64,
                "source_split": "d" * 64,
            }

    checkpoint = {
        "config": {
            "data": {
                "dataset_kind": "reference",
                "data_dir": str(tmp_path / "processed"),
                "max_nodes": 16,
                split_manifest_key: str(source_split_path),
                "verify_hashes": True,
            },
            "model": {},
        },
        "model": {},
    }
    monkeypatch.setattr(evaluation_cli, "ReferenceIntentDataset", FakeReferenceDataset)
    monkeypatch.setattr(
        evaluation_cli,
        "DataLoader",
        lambda dataset, **kwargs: batches,
    )
    monkeypatch.setattr(evaluation_cli.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(evaluation_cli, "_load_model", lambda *args: _FakeModel())
    monkeypatch.setattr(
        evaluation_cli,
        "decode_intent_ir",
        lambda graph, outputs: deepcopy(
            next(
                batch["intent_ir"][0]
                for batch in batches
                if batch["sample_id"][0] == outputs["sample_id"]
            )
        ),
    )
    monkeypatch.setattr(evaluation_cli, "validate_intent_ir", lambda *args: [])

    evaluation_cli.main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--reference-package",
            str(package_dir),
            "--split",
            "test",
            "--bootstrap-samples",
            "20",
            "--output",
            str(output_path),
        ]
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["schema_version"] == "intent-evaluation/v2"
    assert result["dataset"]["counts_by_tier"] == {
        "human_gold": 1,
        "ai_silver": 1,
    }
    assert not (output_path.parent / "test_unseal.json").exists()


def test_reference_cli_requires_explicit_split(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "mock.pt"
    checkpoint_path.write_bytes(b"mock checkpoint")
    package_dir = tmp_path / "reference_fixture"
    checkpoint = {
        "config": {
            "data": {
                "dataset_kind": "reference",
                "package_dir": str(package_dir),
            },
            "model": {},
        },
        "model": {},
    }
    monkeypatch.setattr(
        evaluation_cli.torch,
        "load",
        lambda *args, **kwargs: checkpoint,
    )

    with pytest.raises(ValueError, match="显式提供 --split"):
        evaluation_cli.main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )


def test_legacy_cli_requires_explicit_split(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "mock.pt"
    checkpoint_path.write_bytes(b"mock checkpoint")
    checkpoint = {
        "config": {
            "data": {"dataset_kind": "page_directory"},
            "model": {},
        },
        "model": {},
    }
    monkeypatch.setattr(
        evaluation_cli.torch,
        "load",
        lambda *args, **kwargs: checkpoint,
    )
    monkeypatch.setattr(
        evaluation_cli,
        "_evaluate_legacy",
        lambda *args, **kwargs: pytest.fail(
            "缺少显式 split 时不应进入 legacy 评测"
        ),
    )

    with pytest.raises(ValueError, match="显式提供 --split"):
        evaluation_cli.main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )


def test_weak_baseline_cli_requires_explicit_split():
    with pytest.raises(SystemExit):
        baseline_cli.main([])


def test_weak_baseline_cli_requires_split_manifest(tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    output_path = tmp_path / "result.json"

    with pytest.raises(SystemExit):
        baseline_cli.main(
            [
                "--data_dir",
                str(data_dir),
                "--split",
                "validation",
                "--output",
                str(output_path),
            ]
        )

    assert not output_path.exists()


def test_weak_baseline_skips_unavailable_metrics_in_aggregate(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "processed"
    sample_dir = data_dir / "dev1"
    sample_dir.mkdir(parents=True)
    (sample_dir / "page_graph.json").write_text("{}\n", encoding="utf-8")
    (sample_dir / "gold_intent.json").write_text("{}\n", encoding="utf-8")
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(
        json.dumps({"splits": {"validation": ["dev1"]}}),
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"
    marker = object()

    monkeypatch.setattr(baseline_cli.PageGraph, "load", lambda path: marker)
    monkeypatch.setattr(baseline_cli.DesignIntentIR, "load", lambda path: marker)
    monkeypatch.setattr(baseline_cli, "build_weak_intent", lambda graph: marker)
    monkeypatch.setattr(baseline_cli, "validate_intent_ir", lambda *args: [])
    monkeypatch.setattr(
        baseline_cli,
        "compute_intent_metrics",
        lambda *args: {"leaf_f1": 0.5, "layout_mode_accuracy": None},
    )

    baseline_cli.main(
        [
            "--data_dir",
            str(data_dir),
            "--split_manifest",
            str(split_manifest),
            "--split",
            "validation",
            "--output",
            str(output_path),
        ]
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["metrics"] == {"leaf_f1": 0.5}
    assert result["per_sample"]["dev1"]["layout_mode_accuracy"] is None
