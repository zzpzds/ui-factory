import json
from pathlib import Path
import shutil

import pytest

from src.annotation import AdjudicationStore


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = REPO_ROOT / "data/annotations/intent_pilot_v1"


def _store(tmp_path: Path) -> AdjudicationStore:
    package = tmp_path / "annotations"
    package.mkdir()
    assignment = json.loads(
        (SOURCE_PACKAGE / "assignment.json").read_text(encoding="utf-8")
    )
    assignment["samples"] = [
        item for item in assignment["samples"] if item["sample_id"] == "0001"
    ]
    (package / "assignment.json").write_text(
        json.dumps(assignment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for directory in (
        "annotator_a",
        "annotator_ai",
        "adjudication",
        "gold_drafts",
    ):
        target = package / directory
        target.mkdir()
        shutil.copy2(SOURCE_PACKAGE / directory / "0001.json", target / "0001.json")
    return AdjudicationStore(REPO_ROOT, package)


def test_adjudication_save_review_and_lock(tmp_path):
    store = _store(tmp_path)
    payload = store.payload("0001")
    assert payload["record"]["status"] == "pending"
    assert payload["draft"]["provenance"]["label_source"] == (
        "human_ai_adjudication_draft"
    )
    assert store.progress() == {"reviewed": 0, "pending": 1, "total": 1}

    saved = store.save("0001", payload["draft"])
    assert saved["reviewed"] is False
    assert saved["errors"] == []

    blocked = store.save(
        "0001",
        saved["draft"],
        review_payload={"reviewed_by": "", "notes": ""},
        submit=True,
    )
    assert blocked["reviewed"] is False
    assert "请填写真实审核人标识" in blocked["errors"]
    assert "请填写审核说明" in blocked["errors"]

    reviewed = store.save(
        "0001",
        blocked["draft"],
        review_payload={
            "reviewed_by": "designer_a",
            "notes": "逐项核对元素、层级、布局和 Token。",
        },
        submit=True,
    )
    assert reviewed["reviewed"] is True
    assert reviewed["errors"] == []
    assert reviewed["record"]["status"] == "reviewed"
    assert reviewed["draft"]["provenance"]["adjudication_status"] == "reviewed"
    assert reviewed["progress"] == {"reviewed": 1, "pending": 0, "total": 1}

    with pytest.raises(ValueError, match="已完成审核"):
        store.save("0001", reviewed["draft"])


def test_adjudication_rejects_ai_as_reviewer(tmp_path):
    store = _store(tmp_path)
    payload = store.payload("0001")

    result = store.save(
        "0001",
        payload["draft"],
        review_payload={
            "reviewed_by": "annotator_ai",
            "notes": "由 AI 直接确认。",
        },
        submit=True,
    )

    assert result["reviewed"] is False
    assert "AI 代理不能作为金标准审核人" in result["errors"]
    assert store.progress() == {"reviewed": 0, "pending": 1, "total": 1}


def test_adjudication_rejects_changed_source_annotation(tmp_path):
    store = _store(tmp_path)
    payload = store.payload("0001")
    human_path = store.human_path("0001")
    human_path.write_text(
        human_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    result = store.save(
        "0001",
        payload["draft"],
        review_payload={
            "reviewed_by": "designer_a",
            "notes": "核对完成。",
        },
        submit=True,
    )

    assert result["reviewed"] is False
    assert "人工原始标注哈希已变化，请重新生成仲裁材料" in result["errors"]
