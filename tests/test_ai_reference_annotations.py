from pathlib import Path
import copy

from scripts.prepare_ai_reference_gold import (
    _remove_stale_json_files,
    _source_candidate_is_current,
    apply_visual_correction_operations,
    attach_reference_tokens,
    build_corrected_token_preview,
    build_expanded_structure_candidate,
)
from src.annotation import AnnotationStore
from src.design_intent.grouping import direct_children
from src.design_intent.schema import DesignIntentIR, PageGraph


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_automatic_reference_tokens_are_small_and_consistent():
    annotation = DesignIntentIR.load(
        REPO_ROOT
        / "data/annotations/intent_gold_v1/annotator_ai_prelabel/0044.json"
    )

    attach_reference_tokens(annotation)

    assert 1 <= len(annotation.style_tokens) <= 12
    token_ids = {token.id for token in annotation.style_tokens}
    entity_ids = {
        entity.id for entity in [*annotation.elements, *annotation.groups]
    }
    assert all(len(set(token.member_ids)) >= 2 for token in annotation.style_tokens)
    assert all(set(token.member_ids) <= entity_ids for token in annotation.style_tokens)
    for element in annotation.elements:
        expected = sorted(
            token.id
            for token in annotation.style_tokens
            if element.id in token.member_ids
        )
        assert element.style_token_refs == expected

    first_pass = annotation.to_dict()["style_tokens"]
    attach_reference_tokens(annotation)
    assert annotation.to_dict()["style_tokens"] == first_pass
    assert all(set(element.style_token_refs) <= token_ids for element in annotation.elements)


def test_expanded_structure_candidate_recovers_visible_container_hierarchy():
    graph = PageGraph.load(
        REPO_ROOT / "data/processed/0544/page_graph.json"
    )

    annotation = build_expanded_structure_candidate(graph)

    assert len(annotation.groups) >= 2
    assert annotation.provenance["label_source"] == (
        "expanded_pagegraph_structure_candidate"
    )
    assert AnnotationStore._submission_errors(annotation, graph) == []


def test_visual_group_corrections_reparent_real_candidate_entities():
    sample_id = "0044"
    graph = PageGraph.load(
        REPO_ROOT / f"data/processed/{sample_id}/page_graph.json"
    )
    annotation = DesignIntentIR.load(
        REPO_ROOT
        / f"data/annotations/intent_gold_v1/annotator_ai_prelabel/{sample_id}.json"
    )
    operations = [
        {
            "action": "create_group",
            "id": "g_visual_top_nav",
            "name": "顶部导航",
            "role": "NAV",
            "child_ids": ["e_5", "e_7", "e_8", "e_9", "e_10", "e_11"],
            "parent_id": "page_root",
        },
        {
            "action": "create_group",
            "id": "g_visual_recent_searches",
            "name": "近期搜索",
            "role": "LIST",
            "child_ids": ["e_42", "e_45", "e_46", "e_49"],
            "parent_id": "page_root",
        },
    ]

    apply_visual_correction_operations(annotation, operations)

    assert direct_children(annotation, "g_visual_top_nav") == operations[0]["child_ids"]
    recent_group = next(
        group for group in annotation.groups
        if group.id == "g_visual_recent_searches"
    )
    assert recent_group.source_element_ids == operations[1]["child_ids"]
    assert any(
        layout.target_id == "g_visual_top_nav"
        for layout in annotation.layouts
    )
    assert AnnotationStore._submission_errors(annotation, graph) == []


def test_source_candidate_reuse_requires_matching_sample_and_asset_hashes():
    sample_id = "0044"
    annotation = DesignIntentIR.load(
        REPO_ROOT
        / f"data/annotations/intent_gold_v1/annotator_ai_prelabel/{sample_id}.json"
    )
    graph_path = REPO_ROOT / f"data/processed/{sample_id}/page_graph.json"
    screenshot_path = REPO_ROOT / f"data/processed/{sample_id}/screenshot.png"

    assert _source_candidate_is_current(
        annotation, sample_id, graph_path, screenshot_path
    )
    annotation.provenance["page_graph_sha256"] = "stale"
    assert not _source_candidate_is_current(
        annotation, sample_id, graph_path, screenshot_path
    )

    expected = DesignIntentIR.load(
        REPO_ROOT
        / f"data/annotations/intent_gold_v1/annotator_ai_prelabel/{sample_id}.json"
    )
    changed = copy.deepcopy(expected)
    changed.elements[0].name = "被误改但 provenance 未变化"
    assert not _source_candidate_is_current(
        changed,
        sample_id,
        graph_path,
        screenshot_path,
        expected=expected,
    )


def test_stale_generated_json_files_are_removed(tmp_path):
    (tmp_path / "keep.json").write_text("{}", encoding="utf-8")
    (tmp_path / "stale.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")

    _remove_stale_json_files(tmp_path, {"keep"})

    assert (tmp_path / "keep.json").is_file()
    assert not (tmp_path / "stale.json").exists()
    assert (tmp_path / "notes.txt").is_file()


def test_token_preview_applies_visual_corrections_before_tokenization():
    package_dir = REPO_ROOT / "data/annotations/intent_gold_v1"
    structure = DesignIntentIR.load(
        package_dir / "annotator_ai_prelabel/0044.json"
    )
    reviews = __import__("json").loads(
        (package_dir / "ai_visual_reviews/batch_01.json").read_text(
            encoding="utf-8"
        )
    )["reviews"]
    review = next(item for item in reviews if item["sample_id"] == "0044")

    preview = build_corrected_token_preview(structure, review)
    reference = DesignIntentIR.load(package_dir / "reference/0044.json")

    assert {group.id for group in preview.groups} >= {
        "g_visual_top_nav",
        "g_visual_recent_searches",
    }
    assert preview.to_dict()["style_tokens"] == (
        reference.to_dict()["style_tokens"]
    )
