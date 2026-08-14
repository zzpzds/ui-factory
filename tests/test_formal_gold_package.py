import json
import hashlib
from pathlib import Path

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
    for item in samples:
        if item["status"] == "annotation_required":
            assert item["content_safety_flags"] == []
            assert item["broken_image_proxy"]["mass_failure_suspected"] is False
        sample_dir = REPO_ROOT / "data/processed" / item["sample_id"]
        assert item["asset_sha256"] == {
            "screenshot": _sha256(sample_dir / "screenshot.png"),
            "page_graph": _sha256(sample_dir / "page_graph.json"),
        }


def test_formal_gold_package_has_locked_gold_and_auditable_ai_assistance():
    assignment = _load("assignment.json")
    store = AnnotationStore(REPO_ROOT, PACKAGE_DIR)
    locked = set(assignment["locked_sample_ids"])

    assert assignment["annotators"] == ["annotator_a"]
    assert len(assignment["samples"]) == 60
    assert len(locked) == 10
    progress = store.progress()["annotator_a"]
    assert progress["total"] == 60
    assert progress["complete"] >= 10
    workflow = assignment["annotation_workflow"]
    assert workflow["assisted_sample_count"] == 45
    assert workflow["blind_control_sample_count"] == 5
    assert workflow["require_human_review_confirmation"] is True
    conditions = {
        item["annotation_condition"] for item in assignment["samples"]
    }
    assert conditions == {
        "ai_assisted", "blind_control", "existing_independent_gold",
    }
    for item in assignment["samples"]:
        sample_id = item["sample_id"]
        annotation = store.annotation("annotator_a", sample_id)
        assert store.screenshot_path(sample_id).is_file()
        assert store.graph(sample_id).sample_id == sample_id
        if sample_id in locked:
            assert item["status"] == "existing_gold"
            assert annotation.provenance["status"] == "complete"
        else:
            assert item["status"] == "annotation_required"
            ai_path = PACKAGE_DIR / "annotator_ai_prelabel" / f"{sample_id}.json"
            ai = DesignIntentIR.load(ai_path)
            assert ai.provenance["label_source"] == "ai_preannotation"
            assert ai.provenance["human_labels_viewed"] is False
            assert ai.provenance["visual_input_used"] is False
            assert ai.provenance["status"] == "complete"
            assert not ai.style_tokens
            assert annotation.provenance["ai_preannotation_sha256"] == _sha256(
                ai_path
            )
            if item["annotation_condition"] == "ai_assisted":
                assert annotation.provenance["ai_assistance_mode"] == (
                    "preannotation"
                )
                assert annotation.provenance["ai_preannotation_visible"] is True
            else:
                assert annotation.provenance["ai_assistance_mode"] == (
                    "blind_control"
                )
                assert annotation.provenance["ai_preannotation_visible"] is False


def test_formal_gold_new_samples_meet_token_enrichment_minimums():
    manifest = _load("selection_manifest.json")
    minimums = manifest["sampling_strategy"][
        "token_rich_minimums_for_new_samples"
    ]
    new_samples = [
        item for item in manifest["samples"]
        if item["status"] == "annotation_required"
    ]

    assert len(new_samples) == 50
    for split, minimum in minimums.items():
        actual = sum(
            item["token_rich_candidate"]
            for item in new_samples
            if item["split"] == split
        )
        assert actual >= minimum
