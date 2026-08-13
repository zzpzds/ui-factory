from pathlib import Path

from scripts.summarize_adjudicated_gold import summarize


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_summarize_adjudicated_gold_is_complete():
    result = summarize(
        REPO_ROOT,
        REPO_ROOT / "data/annotations/intent_pilot_v1",
    )

    assert result["complete"] is True
    assert result["validated_gold"] == 10
    assert result["errors"] == []
    assert set(result["per_sample"]) == {
        "0001", "0020", "0319", "0419", "0601",
        "0860", "0052", "1094", "1429", "0937",
    }
    assert result["interpretation"] == "跨来源描述性比较，不是双人标注者间一致性"
