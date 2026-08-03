import hashlib
import json
from pathlib import Path

from scripts.generate_ai_proxy_annotations import SPECS, build_annotation
from src.annotation.store import AnnotationStore
from src.design_intent.schema import DesignIntentIR, PageGraph


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "data/annotations/intent_pilot_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ai_proxy_specs_cover_assignment_and_validate():
    assignment = json.loads(
        (PACKAGE_DIR / "assignment.json").read_text(encoding="utf-8")
    )
    sample_ids = [item["sample_id"] for item in assignment["samples"]]
    assert set(SPECS) == set(sample_ids)

    for item in assignment["samples"]:
        sample_id = item["sample_id"]
        graph_path = REPO_ROOT / item["page_graph"]
        screenshot_path = REPO_ROOT / item["screenshot"]
        ir = build_annotation(sample_id, graph_path, screenshot_path)
        graph = PageGraph.load(graph_path)

        assert AnnotationStore._submission_errors(ir, graph) == []
        assert ir.provenance["label_source"] == "ai_proxy_annotation"
        assert ir.provenance["annotator_kind"] == "ai_proxy"
        assert ir.provenance["human_labels_viewed"] is False
        assert ir.provenance["weak_labels_viewed"] is False


def test_frozen_ai_proxy_manifest_matches_files():
    manifest = json.loads(
        (PACKAGE_DIR / "annotator_ai/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["human_labels_viewed_before_freeze"] is False
    assert manifest["weak_labels_viewed_before_freeze"] is False
    assert len(manifest["samples"]) == 10

    for item in manifest["samples"]:
        path = PACKAGE_DIR / "annotator_ai" / f"{item['sample_id']}.json"
        ir = DesignIntentIR.load(path)
        assert item["annotation_sha256"] == _sha256(path)
        assert item["elements"] == len(ir.elements)
        assert item["groups"] == len(ir.groups)
        assert item["tokens"] == len(ir.style_tokens)
        assert ir.provenance["status"] == "complete"
