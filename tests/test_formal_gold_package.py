import json
import hashlib
from pathlib import Path
import shutil

from scripts.replace_formal_reference_samples import replace
from src.annotation import AnnotationStore
from src.design_intent.schema import DesignIntentIR


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "data/annotations/intent_gold_v1"


def _load(name: str):
    return json.loads((PACKAGE_DIR / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_gold_manifest_has_frozen_split_and_unique_samples():
    manifest = _load("selection_manifest.json")
    samples = manifest["samples"]

    assert manifest["purpose"] == "design_intent_reference_v1_sampling"
    assert len(samples) == 60
    assert {split: len(ids) for split, ids in manifest["splits"].items()} == {
        "train": 30,
        "validation": 10,
        "test": 20,
    }
    assert len({item["sample_id"] for item in samples}) == 60
    assert len({item["duplicate_cluster"] for item in samples}) == 60
    assert len({item["structure_fingerprint"] for item in samples}) == 60
    excluded = {item["sample_id"] for item in manifest["visual_exclusions"]}
    excluded.update(
        item["sample_id"] for item in manifest["prior_pilot_rejections"]
    )
    assert excluded.isdisjoint(item["sample_id"] for item in samples)
    assert manifest["annotation_required_ids"] == []
    assert len(manifest["reference_complete_ids"]) == 60
    assert len(manifest["human_gold_sample_ids"]) == 10
    assert len(manifest["ai_silver_sample_ids"]) == 50
    assert sum(
        item["selection_status"] == "annotation_required" for item in samples
    ) == 50
    for item in samples:
        if item["selection_status"] == "annotation_required":
            assert item["content_safety_flags"] == []
            assert item["broken_image_proxy"]["mass_failure_suspected"] is False
            assert item["status"] == "ai_reference_complete"
        else:
            assert item["selection_status"] == "existing_gold"
            assert item["status"] == "existing_human_adjudicated_gold"
        sample_dir = REPO_ROOT / "data/processed" / item["sample_id"]
        assert item["asset_sha256"] == {
            "screenshot": _sha256(sample_dir / "screenshot.png"),
            "page_graph": _sha256(sample_dir / "page_graph.json"),
        }


def test_failed_full_resolution_samples_are_replaced_from_reserve_pool():
    manifest = _load("selection_manifest.json")
    assignment = _load("assignment.json")
    replacements = {
        item["old_sample_id"]: item["new_sample_id"]
        for item in manifest["quality_replacements"]
    }

    assert replacements["0144"] == "0433"
    assert replacements["0553"] == "0584"
    assert replacements["1365"] == "1530"
    selected_ids = {item["sample_id"] for item in manifest["samples"]}
    assigned_ids = {item["sample_id"] for item in assignment["samples"]}
    assert {"0144", "0553", "1365"}.isdisjoint(selected_ids)
    assert {"0433", "0584", "1530"} <= selected_ids
    assert assigned_ids == selected_ids
    assert len(manifest["splits"]["validation"]) == 10
    validation_sizes = {
        size: sum(
            item["size_bin"] == size
            for item in manifest["samples"]
            if item["split"] == "validation"
        )
        for size in ("small", "medium", "large")
    }
    assert validation_sizes == {"small": 3, "medium": 4, "large": 3}


def test_replacement_script_recovers_assignment_after_partial_previous_write(
    tmp_path,
):
    package_dir = tmp_path / "intent_gold_v1"
    package_dir.mkdir()
    for name in (
        "selection_manifest.json",
        "assignment.json",
        "visual_exclusions.json",
    ):
        shutil.copy2(PACKAGE_DIR / name, package_dir / name)

    assignment_path = package_dir / "assignment.json"
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    interrupted_item = next(
        item for item in assignment["samples"] if item["sample_id"] == "0433"
    )
    interrupted_item["sample_id"] = "0144"
    assignment_path.write_text(
        json.dumps(assignment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    replace(package_dir)

    repaired_assignment = json.loads(
        assignment_path.read_text(encoding="utf-8")
    )
    assigned_ids = {
        item["sample_id"] for item in repaired_assignment["samples"]
    }
    selected_ids = {
        item["sample_id"]
        for item in json.loads(
            (package_dir / "selection_manifest.json").read_text(
                encoding="utf-8"
            )
        )["samples"]
    }
    assert assigned_ids == selected_ids
    repaired_item = next(
        item for item in repaired_assignment["samples"]
        if item["sample_id"] == "0433"
    )
    assert repaired_item["screenshot"].endswith("/0433/screenshot.png")
    assert repaired_item["page_graph"].endswith("/0433/page_graph.json")


def test_formal_gold_package_has_two_tier_read_only_reference_set():
    assignment = _load("assignment.json")
    store = AnnotationStore(REPO_ROOT, PACKAGE_DIR)
    human_gold = set(assignment["human_gold_sample_ids"])
    ai_reference = set(assignment["ai_silver_sample_ids"])
    read_only = set(assignment["read_only_sample_ids"])

    assert assignment["purpose"] == "design_intent_reference_v1"
    assert assignment["annotators"] == ["reference"]
    assert len(assignment["samples"]) == 60
    assert len(human_gold) == 10
    assert len(ai_reference) == 50
    assert human_gold.isdisjoint(ai_reference)
    assert human_gold | ai_reference == read_only
    progress = store.progress()["reference"]
    assert progress == {"complete": 60, "draft": 0, "total": 60}
    workflow = assignment["annotation_workflow"]
    assert workflow["mode"] == "ai_multiview_reference_generation"
    assert workflow["ai_silver_sample_count"] == 50
    assert workflow["human_gold_sample_count"] == 10
    assert workflow["require_human_review_confirmation"] is False
    diagnostic_replay_human_gold = set(
        workflow["diagnostic_replay_human_gold_sample_ids"]
    )
    non_replayed_human_gold = set(
        workflow["non_replayed_human_gold_sample_ids"]
    )
    assert len(diagnostic_replay_human_gold) == 8
    assert len(non_replayed_human_gold) == 2
    assert diagnostic_replay_human_gold | non_replayed_human_gold == human_gold
    assert diagnostic_replay_human_gold.isdisjoint(non_replayed_human_gold)
    split_by_id = {
        item["sample_id"]: item["split"] for item in assignment["samples"]
    }
    assert {split_by_id[sample_id] for sample_id in non_replayed_human_gold} == {
        "test"
    }
    assert workflow["generator_threshold_selection_used_human_gold"] is False
    assert workflow["developer_human_gold_blinding_status"] == "not_guaranteed"
    assert workflow["rule_teacher_candidate_used"] is True
    assert "weak_labels_visible" not in assignment
    assert assignment["human_gold_original_assignment_context"] == {
        "weak_labels_visible": False,
        "scope": "original_human_annotation_only",
    }
    assert workflow["human_gold_token_labels_available"] is False
    assert workflow["human_gold_token_metrics_reportable"] is False
    assert assignment["token_annotation_mode"] == "automatic_reference"
    conditions = {
        item["annotation_condition"] for item in assignment["samples"]
    }
    assert conditions == {
        "ai_multiview_silver", "existing_human_adjudicated_gold",
    }
    for item in assignment["samples"]:
        sample_id = item["sample_id"]
        annotation = store.annotation("reference", sample_id)
        assert store.screenshot_path(sample_id).is_file()
        assert store.graph(sample_id).sample_id == sample_id
        assert annotation.provenance["status"] == "complete"
        if sample_id in human_gold:
            assert item["status"] == "existing_human_adjudicated_gold"
            source = REPO_ROOT / "data/processed" / sample_id / "gold_intent.json"
            source_payload = DesignIntentIR.load(source).to_dict()
            reference_payload = annotation.to_dict()
            reference_provenance = reference_payload.pop("provenance")
            source_provenance = source_payload.pop("provenance")
            source_provenance.pop("reviewed_by", None)
            assert reference_payload == source_payload
            assert {
                key: value
                for key, value in reference_provenance.items()
                if key not in {
                    "reference_tier",
                    "token_labels_available",
                    "token_review",
                    "reference_packaged_at",
                    "human_reviewer_count",
                    "review_process",
                    "reviewer_id",
                    "reviewer_independent_of_base_annotator",
                    "reviewer_qualification_documented",
                    "reviewer_relationship_to_author",
                    "reviewer_conflict_of_interest_status",
                    "reviewer_public_name_removed",
                    "public_name_consent_status",
                    "review_notes_detail",
                }
            } == source_provenance
            assert "reviewed_by" not in reference_provenance
            assert reference_provenance["reference_tier"] == "human_gold"
            assert reference_provenance["human_reviewer_count"] == 1
            assert reference_provenance["review_process"] == (
                "single_human_ai_assisted_finalization"
            )
            assert reference_provenance["reviewer_id"] == "human_reviewer_01"
            assert (
                reference_provenance[
                    "reviewer_independent_of_base_annotator"
                ]
                is False
            )
            assert reference_provenance["reviewer_qualification_documented"] is False
            assert reference_provenance["reviewer_relationship_to_author"] == (
                "not_documented"
            )
            assert reference_provenance["public_name_consent_status"] == (
                "not_documented"
            )
            assert reference_provenance["token_labels_available"] is False
            assert reference_provenance["token_review"] == {
                "status": "not_collected",
                "positive_labels_available": False,
                "explicit_negative_labels_available": False,
            }
            assert annotation.provenance["status"] == "complete"
        else:
            assert item["status"] == "ai_reference_complete"
            provenance = annotation.provenance
            assert provenance["reference_tier"] == "ai_silver"
            assert provenance["label_source"] == "ai_multiview_silver"
            assert provenance["annotator_kind"] == "ai_agent_pipeline"
            assert provenance["human_review_performed"] is False
            assert provenance["human_labels_viewed_for_target"] is False
            assert provenance["human_gold_used_for_generator_calibration"] is False
            assert provenance["human_gold_diagnostic_replay_performed"] is True
            assert set(provenance["generator_diagnostic_replay_sample_ids"]) == (
                diagnostic_replay_human_gold
            )
            replay_artifact = provenance[
                "generator_diagnostic_replay_artifact"
            ]
            assert _sha256(REPO_ROOT / replay_artifact["path"]) == (
                replay_artifact["sha256"]
            )
            assert provenance["ai_treated_as_human_annotator"] is False
            assert provenance["visual_input_used"] is True
            assert provenance["page_graph_used"] is True
            assert "human_review_confirmed" not in provenance
            assert "human_reviewed_at" not in provenance
            assert "human_active_seconds" not in provenance
            inputs = provenance["pipeline_inputs"]
            assert {entry["kind"] for entry in inputs} == {
                "pagegraph_structure_candidate",
                "expanded_pagegraph_structure_candidate",
                "screenshot_visual_semantic_review",
                "computed_style_token_candidate",
            }
            for entry in inputs:
                path = REPO_ROOT / entry["path"]
                assert path.is_file()
                assert entry["sha256"] == _sha256(path)


def test_ai_reference_tokens_are_bounded_and_referentially_valid():
    assignment = _load("assignment.json")
    ai_reference = set(assignment["ai_silver_sample_ids"])
    token_pages = 0

    for sample_id in sorted(ai_reference):
        annotation = DesignIntentIR.load(
            PACKAGE_DIR / "reference" / f"{sample_id}.json"
        )
        token_payload = json.loads(
            (PACKAGE_DIR / "ai_token_candidates" / f"{sample_id}.json")
            .read_text(encoding="utf-8")
        )
        assert token_payload["correction_operations_applied"] is True
        assert token_payload["style_tokens"] == (
            annotation.to_dict()["style_tokens"]
        )
        entity_ids = {
            entity.id for entity in [*annotation.elements, *annotation.groups]
        }
        token_ids = {token.id for token in annotation.style_tokens}
        assert len(annotation.style_tokens) <= 12
        if annotation.style_tokens:
            token_pages += 1
        for token in annotation.style_tokens:
            assert len(set(token.member_ids)) >= 2
            assert set(token.member_ids) <= entity_ids
        for element in annotation.elements:
            assert set(element.style_token_refs) <= token_ids

    assert token_pages >= 29


def test_active_structure_candidate_directory_has_no_replaced_samples():
    assignment = _load("assignment.json")
    expected = set(assignment["ai_silver_sample_ids"])
    actual = {
        path.stem
        for path in (PACKAGE_DIR / "annotator_ai_prelabel").glob("*.json")
    }

    assert actual == expected


def test_ai_reference_manifest_is_complete_and_does_not_claim_human_labels():
    manifest = _load("ai_reference_manifest.json")

    assert manifest["purpose"] == "ai_multiview_silver_annotation"
    assert manifest["ai_silver_samples"] == 50
    assert manifest["human_gold_samples"] == 10
    assert manifest["human_review_performed"] is False
    assert manifest["ai_labels_treated_as_human"] is False
    assert manifest["generator_threshold_selection_used_human_gold"] is False
    assert manifest["developer_human_gold_blinding_status"] == "not_guaranteed"
    assert manifest["human_gold_token_labels_available"] is False
    assert manifest["human_gold_token_metrics_reportable"] is False
    assert manifest["token_policy"]["evaluation_scope"] == (
        "ai_silver_proxy_consistency_only"
    )
    reproducibility = manifest["visual_review_reproducibility"]
    assert reproducibility["supported_scope"] == "frozen_artifact_replay"
    assert reproducibility["end_to_end_regeneration_supported"] is False
    assert reproducibility["model_version_recorded"] is False
    assert reproducibility["prompt_hash_recorded"] is False
    replay_artifact = manifest["diagnostic_replay_artifact"]
    assert _sha256(REPO_ROOT / replay_artifact["path"]) == (
        replay_artifact["sha256"]
    )
    assert replay_artifact["reported_structure_metrics"]["parent_f1"] < 0.5
    assert replay_artifact["token_metrics_interpretable"] is False

    historical = _load("ai_assistance_manifest.json")
    assert historical["status"] == "superseded"
    assert historical["human_assisted_samples_planned"] == 45
    assert historical["human_reviews_completed"] == 0
    assert historical["blind_control_annotations_completed"] == 0
    assert historical["human_study_results_available"] is False
    assert "human_assisted_samples" not in historical
    assert "blind_control_samples" not in historical
    assert len(manifest["samples"]) == 50
    assert all(item["quality_status"] == "accepted" for item in manifest["samples"])
    assert len(manifest["human_gold_files"]) == 10
    human_review = manifest["human_gold_review_provenance"]
    assert human_review["human_reviewer_count"] == 1
    assert human_review["reviewer_independent_of_base_annotator"] is False
    assert human_review["reviewer_qualification_documented"] is False
    assert human_review["reviewer_relationship_to_author"] == "not_documented"
    assert human_review["brief_review_notes_samples"] == 7
    for item in manifest["human_gold_files"]:
        assert _sha256(REPO_ROOT / item["source_path"]) == item["source_sha256"]
        assert _sha256(REPO_ROOT / item["reference_path"]) == (
            item["reference_sha256"]
        )
        assert item["token_labels_available"] is False


def test_formal_gold_new_samples_meet_token_enrichment_minimums():
    manifest = _load("selection_manifest.json")
    minimums = manifest["sampling_strategy"][
        "token_rich_minimums_for_new_samples"
    ]
    new_samples = [
        item for item in manifest["samples"]
        if item["selection_status"] == "annotation_required"
    ]

    assert len(new_samples) == 50
    for split, minimum in minimums.items():
        actual = sum(
            item["token_rich_candidate"]
            for item in new_samples
            if item["split"] == split
        )
        assert actual >= minimum
