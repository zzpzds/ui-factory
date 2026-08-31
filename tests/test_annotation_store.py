import hashlib
import json

import pytest

from src.annotation import AnnotationStore
from src.design_intent.schema import BBox, Canvas, PageGraph, PageNode


def _node(node_id, text, y):
    return PageNode(
        id=node_id,
        parent_id=None,
        tag="p",
        text=text,
        attributes={},
        bbox=BBox(8, y, 84, 20),
        computed_style={"fontSize": 16, "fontWeight": 400},
    )


def _empty_annotation(sample_id, annotator):
    return {
        "schema_version": "1.0",
        "canvas": {"width": 100, "height": 100},
        "elements": [],
        "groups": [],
        "layouts": [],
        "tree": [],
        "style_tokens": [],
        "provenance": {
            "source_sample_id": sample_id,
            "label_source": "human_annotation",
            "annotator": annotator,
            "status": "draft",
            "weak_labels_viewed": False,
        },
    }


def _valid_annotation():
    return {
        "schema_version": "1.0",
        "canvas": {"width": 100, "height": 100},
        "elements": [
            {
                "id": "e_1",
                "source_node_ids": [0],
                "type": "TEXT",
                "bbox": {"x": 8, "y": 8, "width": 84, "height": 20},
                "name": "标题",
                "text": "标题",
                "style": {},
                "style_token_refs": [],
                "confidence": 1,
            },
            {
                "id": "e_2",
                "source_node_ids": [1],
                "type": "TEXT",
                "bbox": {"x": 8, "y": 40, "width": 84, "height": 20},
                "name": "正文",
                "text": "正文",
                "style": {},
                "style_token_refs": [],
                "confidence": 1,
            },
        ],
        "groups": [
            {
                "id": "g_1",
                "source_element_ids": ["e_1", "e_2"],
                "role": "SECTION",
                "bbox": {"x": 8, "y": 8, "width": 84, "height": 52},
                "name": "内容区",
                "style": {},
                "source_node_id": None,
                "confidence": 1,
            },
        ],
        "layouts": [
            {
                "target_id": "g_1",
                "mode": "VERTICAL",
                "gap": 12,
                "padding": [0, 0, 0, 0],
                "primary_align": "START",
                "cross_align": "START",
                "horizontal_resize": "FIXED",
                "vertical_resize": "HUG",
                "confidence": 1,
            },
        ],
        "tree": [
            {
                "parent_id": "page_root",
                "child_id": "g_1",
                "order": 0,
                "confidence": 1,
            },
            {
                "parent_id": "g_1",
                "child_id": "e_1",
                "order": 0,
                "confidence": 1,
            },
            {
                "parent_id": "g_1",
                "child_id": "e_2",
                "order": 1,
                "confidence": 1,
            },
        ],
        "style_tokens": [
            {
                "id": "token_text_1",
                "kind": "TEXT",
                "value": {"font_size": 16, "font_weight": 400},
                "member_ids": ["e_1", "e_2"],
                "name": "正文样式",
                "confidence": 1,
            },
        ],
        "provenance": {"status": "draft"},
    }


@pytest.fixture()
def annotation_store(tmp_path):
    graph_path = tmp_path / "data" / "page_graph.json"
    graph_path.parent.mkdir(parents=True)
    PageGraph(
        schema_version="1.0",
        sample_id="sample-1",
        canvas=Canvas(100, 100),
        nodes=[
            _node(0, "标题", 8),
            _node(1, "正文", 30),
            _node(2, "第二行标题", 52),
            _node(3, "第二行正文", 74),
        ],
    ).dump(graph_path)
    screenshot_path = tmp_path / "data" / "screenshot.png"
    screenshot_path.write_bytes(b"png")

    package = tmp_path / "annotations"
    assignment = {
        "schema_version": "1.0",
        "samples": [
            {
                "sample_id": "sample-1",
                "size_bin": "small",
                "page_graph": "data/page_graph.json",
                "screenshot": "data/screenshot.png",
            },
        ],
    }
    package.mkdir()
    (package / "assignment.json").write_text(
        json.dumps(assignment), encoding="utf-8"
    )
    for annotator in ("annotator_a", "annotator_b"):
        target = package / annotator
        target.mkdir()
        (target / "sample-1.json").write_text(
            json.dumps(_empty_annotation("sample-1", annotator)),
            encoding="utf-8",
        )
    return AnnotationStore(tmp_path, package)


def test_assignment_progress_is_isolated_by_annotator(annotation_store):
    payload = annotation_store.assignment_payload()

    assert payload["progress"]["annotator_a"] == {
        "complete": 0,
        "draft": 1,
        "total": 1,
    }
    assert payload["sample_status"]["annotator_b"]["sample-1"] == "draft"


def test_assignment_can_configure_one_annotator_and_lock_sample(tmp_path):
    graph_path = tmp_path / "data/page_graph.json"
    graph_path.parent.mkdir(parents=True)
    PageGraph(
        schema_version="1.0",
        sample_id="sample-1",
        canvas=Canvas(100, 100),
        nodes=[_node(0, "标题", 8)],
    ).dump(graph_path)
    screenshot_path = tmp_path / "data/screenshot.png"
    screenshot_path.write_bytes(b"png")

    package = tmp_path / "annotations"
    annotation_dir = package / "annotator_a"
    annotation_dir.mkdir(parents=True)
    assignment = {
        "schema_version": "1.0",
        "annotators": ["annotator_a"],
        "locked_sample_ids": ["sample-1"],
        "samples": [{
            "sample_id": "sample-1",
            "size_bin": "small",
            "page_graph": "data/page_graph.json",
            "screenshot": "data/screenshot.png",
        }],
    }
    (package / "assignment.json").write_text(
        json.dumps(assignment), encoding="utf-8"
    )
    (annotation_dir / "sample-1.json").write_text(
        json.dumps(_empty_annotation("sample-1", "annotator_a")),
        encoding="utf-8",
    )

    store = AnnotationStore(tmp_path, package)

    assert store.assignment_payload()["annotators"] == ["annotator_a"]
    with pytest.raises(ValueError, match="已冻结为金标准"):
        store.save(
            "annotator_a",
            "sample-1",
            _empty_annotation("sample-1", "annotator_a"),
        )


def test_read_only_reference_cannot_be_saved_or_rewritten(annotation_store):
    annotation_store.read_only_sample_ids = {"sample-1"}
    path = annotation_store.annotation_path("annotator_a", "sample-1")
    before = path.read_bytes()
    payload = _valid_annotation()
    payload["provenance"].update({
        "reference_tier": "ai_silver",
        "label_source": "ai_multiview_silver",
        "annotator_kind": "ai_agent_pipeline",
        "human_review_performed": False,
    })

    with pytest.raises(ValueError, match="已冻结为参考标注"):
        annotation_store.save(
            "annotator_a", "sample-1", payload, submit=True
        )

    assert path.read_bytes() == before


def test_draft_saves_even_when_incomplete(annotation_store):
    result = annotation_store.save(
        "annotator_a",
        "sample-1",
        _empty_annotation("sample-1", "annotator_a"),
    )

    assert result["submitted"] is False
    assert "完整标注至少需要一个原子元素" in result["errors"]
    assert result["annotation"]["provenance"]["status"] == "draft"


def test_valid_submission_sets_provenance_and_token_references(annotation_store):
    result = annotation_store.save(
        "annotator_a", "sample-1", _valid_annotation(), submit=True
    )

    assert result["errors"] == []
    assert result["submitted"] is True
    provenance = result["annotation"]["provenance"]
    assert provenance["status"] == "complete"
    assert provenance["source_sample_id"] == "sample-1"
    assert provenance["annotator"] == "annotator_a"
    assert provenance["weak_labels_viewed"] is False
    assert result["annotation"]["elements"][0]["style_token_refs"] == [
        "token_text_1"
    ]
    assert annotation_store.progress()["annotator_a"]["complete"] == 1
    assert annotation_store.progress()["annotator_b"]["complete"] == 0


def test_invalid_token_submission_remains_draft(annotation_store):
    payload = _valid_annotation()
    payload["style_tokens"][0]["member_ids"] = ["e_1"]

    result = annotation_store.save(
        "annotator_a", "sample-1", payload, submit=True
    )

    assert result["submitted"] is False
    assert result["annotation"]["provenance"]["status"] == "draft"
    assert any("至少需要两个成员" in error for error in result["errors"])


def test_candidate_review_mode_requires_explicit_token_review(annotation_store):
    annotation_store.assignment["token_annotation_mode"] = "candidate_review"
    payload = _valid_annotation()

    pending = annotation_store.save(
        "annotator_a", "sample-1", payload, submit=True
    )

    assert pending["submitted"] is False
    assert any("Token 样式候选检查" in error for error in pending["errors"])

    payload["provenance"]["token_review"] = {
        "status": "reviewed",
        "reviewed_candidate_keys": [],
        "accepted_candidate_keys": [],
        "ignored_candidate_keys": [],
    }
    reviewed = annotation_store.save(
        "annotator_a", "sample-1", payload, submit=True
    )

    assert reviewed["submitted"] is True


def test_ai_assisted_submission_requires_human_confirmation(annotation_store):
    annotation_store.assignment["annotation_workflow"] = {
        "require_human_review_confirmation": True,
    }
    source_path = annotation_store.repo_root / "data/page_graph.json"
    payload = _valid_annotation()
    payload["provenance"].update({
        "ai_assistance_mode": "preannotation",
        "ai_preannotation_visible": True,
        "ai_preannotation_path": "data/page_graph.json",
        "ai_preannotation_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "human_review_confirmed": False,
    })

    pending = annotation_store.save(
        "annotator_a", "sample-1", payload, submit=True
    )
    assert pending["submitted"] is False
    assert any("设计师已完成逐项复核" in error for error in pending["errors"])

    payload["provenance"]["human_review_confirmed"] = True
    reviewed = annotation_store.save(
        "annotator_a", "sample-1", payload, submit=True
    )
    assert reviewed["submitted"] is True
    provenance = reviewed["annotation"]["provenance"]
    assert provenance["label_source"] == "human_corrected_ai_preannotation"
    assert provenance["weak_labels_viewed"] is True
    assert provenance["ai_assistance_disclosed"] is True
    assert provenance["human_reviewed_at"]


def test_group_can_directly_contain_only_child_groups(annotation_store):
    payload = _valid_annotation()
    payload["elements"].extend(
        [
            {
                "id": "e_3",
                "source_node_ids": [2],
                "type": "TEXT",
                "bbox": {"x": 8, "y": 52, "width": 84, "height": 20},
                "name": "第二行标题",
                "text": "第二行标题",
                "style": {},
                "style_token_refs": [],
                "confidence": 1,
            },
            {
                "id": "e_4",
                "source_node_ids": [3],
                "type": "TEXT",
                "bbox": {"x": 8, "y": 74, "width": 84, "height": 20},
                "name": "第二行正文",
                "text": "第二行正文",
                "style": {},
                "style_token_refs": [],
                "confidence": 1,
            },
        ]
    )
    payload["groups"] = [
        {
            "id": "g_row_1",
            "source_element_ids": [],
            "role": "TABLE_ROW",
            "bbox": {"x": 8, "y": 8, "width": 84, "height": 42},
            "name": "第一行",
            "style": {},
            "source_node_id": None,
            "confidence": 1,
        },
        {
            "id": "g_row_2",
            "source_element_ids": [],
            "role": "TABLE_ROW",
            "bbox": {"x": 8, "y": 52, "width": 84, "height": 42},
            "name": "第二行",
            "style": {},
            "source_node_id": None,
            "confidence": 1,
        },
        {
            "id": "g_table",
            "source_element_ids": [],
            "role": "TABLE",
            "bbox": {"x": 8, "y": 8, "width": 84, "height": 86},
            "name": "数据表格",
            "style": {},
            "source_node_id": None,
            "confidence": 1,
        },
    ]
    payload["layouts"] = [
        {
            "target_id": group_id,
            "mode": "HORIZONTAL" if "row" in group_id else "VERTICAL",
            "gap": 2,
            "padding": [0, 0, 0, 0],
            "primary_align": "START",
            "cross_align": "START",
            "horizontal_resize": "FIXED",
            "vertical_resize": "HUG",
            "confidence": 1,
        }
        for group_id in ("g_row_1", "g_row_2", "g_table")
    ]
    payload["tree"] = [
        {"parent_id": "page_root", "child_id": "g_table", "order": 0},
        {"parent_id": "g_table", "child_id": "g_row_1", "order": 0},
        {"parent_id": "g_table", "child_id": "g_row_2", "order": 1},
        {"parent_id": "g_row_1", "child_id": "e_1", "order": 0},
        {"parent_id": "g_row_1", "child_id": "e_2", "order": 1},
        {"parent_id": "g_row_2", "child_id": "e_3", "order": 0},
        {"parent_id": "g_row_2", "child_id": "e_4", "order": 1},
    ]

    result = annotation_store.save(
        "annotator_a", "sample-1", payload, submit=True
    )

    assert result["submitted"] is True
    groups = {
        group["id"]: group for group in result["annotation"]["groups"]
    }
    assert groups["g_table"]["source_element_ids"] == [
        "e_1", "e_2", "e_3", "e_4"
    ]
    table_children = [
        edge["child_id"]
        for edge in result["annotation"]["tree"]
        if edge["parent_id"] == "g_table"
    ]
    assert table_children == ["g_row_1", "g_row_2"]


def test_group_with_only_one_atomic_child_is_rejected(annotation_store):
    payload = _valid_annotation()
    for edge in payload["tree"]:
        if edge["child_id"] == "e_2":
            edge["parent_id"] = "page_root"

    result = annotation_store.save(
        "annotator_a", "sample-1", payload, submit=True
    )

    assert result["submitted"] is False
    assert any(
        "不能只包含一个直接原子元素" in error
        for error in result["errors"]
    )


def test_rejects_source_path_outside_repo(annotation_store):
    annotation_store.samples["sample-1"]["page_graph"] = "../outside.json"

    with pytest.raises(ValueError, match="越出仓库"):
        annotation_store.graph("sample-1")
