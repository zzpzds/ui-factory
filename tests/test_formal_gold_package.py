import json
import hashlib
from pathlib import Path

from src.annotation import AnnotationStore


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


def test_formal_gold_package_has_10_locked_gold_and_50_blank_drafts():
    assignment = _load("assignment.json")
    store = AnnotationStore(REPO_ROOT, PACKAGE_DIR)
    locked = set(assignment["locked_sample_ids"])

    assert assignment["annotators"] == ["annotator_a"]
    assert len(assignment["samples"]) == 60
    assert len(locked) == 10
    assert store.progress()["annotator_a"] == {
        "complete": 10,
        "draft": 50,
        "total": 60,
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
            assert annotation.provenance["status"] == "draft"
            assert not annotation.elements
            assert not annotation.groups
            assert not annotation.style_tokens


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
