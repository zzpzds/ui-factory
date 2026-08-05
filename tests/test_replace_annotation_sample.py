import json
from pathlib import Path

from scripts.create_annotation_pilot import empty_annotation, structure_fingerprint
from scripts.replace_annotation_sample import replace_sample
from src.design_intent.schema import BBox, Canvas, PageGraph, PageNode


def _graph(sample_id: str, node_count: int) -> PageGraph:
    return PageGraph(
        schema_version="1.0",
        sample_id=sample_id,
        canvas=Canvas(1280, 800),
        nodes=[
            PageNode(
                id=index,
                parent_id=None,
                tag="div",
                text=f"节点 {index}",
                attributes={},
                bbox=BBox(0, index % 700, 100, 20),
                computed_style={},
            )
            for index in range(node_count)
        ],
    )


def test_replace_archives_ai_track_and_marks_rebuild_pending(tmp_path):
    data_dir = tmp_path / "data"
    package_dir = tmp_path / "annotations"
    old_graph = _graph("old", 130)
    new_graph = _graph("new", 135)
    for graph in (old_graph, new_graph):
        sample_dir = data_dir / graph.sample_id
        sample_dir.mkdir(parents=True)
        graph.dump(sample_dir / "page_graph.json")
        (sample_dir / "screenshot.png").write_bytes(b"png")

    package_dir.mkdir()
    assignment = {
        "samples": [
            {
                "sample_id": "old",
                "node_count": 130,
                "size_bin": "large",
                "structure_fingerprint": structure_fingerprint(old_graph),
                "screenshot": str(data_dir / "old/screenshot.png"),
                "page_graph": str(data_dir / "old/page_graph.json"),
            }
        ],
        "annotation_tracks": {
            "ai_proxy": {"status": "complete"},
        },
    }
    (package_dir / "assignment.json").write_text(
        json.dumps(assignment), encoding="utf-8"
    )
    for annotator in ("annotator_a", "annotator_b"):
        directory = package_dir / annotator
        directory.mkdir()
        empty_annotation(old_graph, annotator).dump(directory / "old.json")
    ai_dir = package_dir / "annotator_ai"
    ai_dir.mkdir()
    (ai_dir / "old.json").write_text("{}", encoding="utf-8")
    (ai_dir / "manifest.json").write_text("{}", encoding="utf-8")

    result = replace_sample(
        package_dir=package_dir,
        data_dir=data_dir,
        old_sample_id="old",
        new_sample_id="new",
        reason="页面质量不合格",
    )

    archive = package_dir / "replaced/old_to_new"
    assert result["ai_proxy_rebuild_required"] is True
    assert (archive / "annotator_ai/old.json").exists()
    assert (archive / "annotator_ai/manifest.json").exists()
    assert not (ai_dir / "old.json").exists()
    assert not (ai_dir / "manifest.json").exists()
    assert (package_dir / "annotator_a/new.json").exists()
    assert (package_dir / "annotator_b/new.json").exists()

    updated = json.loads(
        (package_dir / "assignment.json").read_text(encoding="utf-8")
    )
    assert updated["samples"][0]["sample_id"] == "new"
    ai_track = updated["annotation_tracks"]["ai_proxy"]
    assert ai_track["status"] == "replacement_pending"
    assert ai_track["pending_sample_id"] == "new"
    assert ai_track["replaced_sample_id"] == "old"
