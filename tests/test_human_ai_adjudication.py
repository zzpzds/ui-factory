import json
from pathlib import Path

from scripts.finalize_human_ai_adjudication import finalize_adjudication
from scripts.prepare_human_ai_adjudication import prepare_adjudication
from src.design_intent.schema import DesignIntentIR


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "data/annotations/intent_pilot_v1"


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_prepare_adjudication_is_incremental_and_draft_only(tmp_path):
    adjudication_dir = tmp_path / "adjudication"
    gold_drafts_dir = tmp_path / "gold_drafts"

    first = prepare_adjudication(
        REPO_ROOT,
        PACKAGE_DIR,
        adjudication_dir,
        gold_drafts_dir,
    )
    assert len(first["prepared"]) == 7
    assert first["incomplete_human"] == ["1094", "1429", "1474"]
    assert first["invalid"] == []
    assert first["training_exported"] is False

    record = json.loads(
        (adjudication_dir / "0001.json").read_text(encoding="utf-8")
    )
    draft = DesignIntentIR.load(gold_drafts_dir / "0001.json")
    assert record["comparison_kind"] == "human-ai"
    assert record["status"] == "pending"
    assert record["inputs"]["ai"]["must_not_be_reported_as_human"] is True
    assert draft.provenance["label_source"] == "human_ai_adjudication_draft"
    assert draft.provenance["gold_finalized"] is False
    assert draft.provenance["ai_treated_as_human_annotator"] is False

    second = prepare_adjudication(
        REPO_ROOT,
        PACKAGE_DIR,
        adjudication_dir,
        gold_drafts_dir,
    )
    assert second["prepared"] == []
    assert len(second["existing"]) == 7


def test_finalize_requires_review_and_exports_disclosed_gold(tmp_path):
    adjudication_dir = tmp_path / "adjudication"
    gold_drafts_dir = tmp_path / "gold_drafts"
    output_dir = tmp_path / "output"
    prepare_adjudication(
        REPO_ROOT,
        PACKAGE_DIR,
        adjudication_dir,
        gold_drafts_dir,
    )

    blocked = finalize_adjudication(
        REPO_ROOT,
        PACKAGE_DIR,
        adjudication_dir,
        gold_drafts_dir,
        output_dir=output_dir,
        sample_ids={"0001"},
    )
    assert blocked["exported"] == []
    assert blocked["errors"]
    assert not (output_dir / "0001/gold_intent.json").exists()

    reviewed_at = "2026-08-03T15:00:00+08:00"
    record_path = adjudication_dir / "0001.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["status"] = "reviewed"
    record["review"].update(
        {
            "reviewed_by": "designer_a",
            "reviewed_at": reviewed_at,
            "notes": "已逐项核对差异并修订金标准草稿。",
        }
    )
    _write(record_path, record)

    draft_path = gold_drafts_dir / "0001.json"
    draft = DesignIntentIR.load(draft_path).to_dict()
    draft["provenance"].update(
        {
            "status": "complete",
            "adjudication_status": "reviewed",
            "reviewed_by": "designer_a",
            "reviewed_at": reviewed_at,
            "review_notes": "已逐项核对差异并修订金标准草稿。",
        }
    )
    _write(draft_path, draft)

    result = finalize_adjudication(
        REPO_ROOT,
        PACKAGE_DIR,
        adjudication_dir,
        gold_drafts_dir,
        output_dir=output_dir,
        sample_ids={"0001"},
    )
    assert result["errors"] == []
    assert result["exported"] == ["0001"]
    gold = DesignIntentIR.load(output_dir / "0001/gold_intent.json")
    assert gold.provenance["label_source"] == "human_ai_adjudicated_gold"
    assert gold.provenance["gold_finalized"] is True
    assert gold.provenance["ai_assistance_disclosed"] is True
