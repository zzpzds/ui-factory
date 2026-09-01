from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs/intent"
WEAK_CONFIGS = {
    "full.yaml",
    "code_only.yaml",
    "visual_only.yaml",
    "no_alignment.yaml",
    "no_dom_graph.yaml",
    "no_structured.yaml",
}
REFERENCE_CONFIGS = {
    "gold_finetune.yaml",
    "reference_finetune.yaml",
    "reference_ai_only.yaml",
    "reference_finetune_w025.yaml",
    "reference_finetune_w100.yaml",
    "code_only_reference_finetune.yaml",
    "visual_only_reference_finetune.yaml",
    "no_alignment_reference_finetune.yaml",
    "no_dom_graph_reference_finetune.yaml",
    "no_structured_reference_finetune.yaml",
}
REFERENCE_ABLATION_PAIRS = {
    "code_only_reference_finetune.yaml": "code_only.yaml",
    "visual_only_reference_finetune.yaml": "visual_only.yaml",
    "no_alignment_reference_finetune.yaml": "no_alignment.yaml",
    "no_dom_graph_reference_finetune.yaml": "no_dom_graph.yaml",
    "no_structured_reference_finetune.yaml": "no_structured.yaml",
}


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_all_v2_intent_configs_exist_and_use_distinct_outputs():
    expected = WEAK_CONFIGS | REFERENCE_CONFIGS
    assert expected <= {path.name for path in CONFIG_DIR.glob("*.yaml")}

    outputs = []
    for name in sorted(expected):
        config = _load(name)
        assert config["experiment"]["name"].endswith("-v2")
        assert config["experiment"]["output_dir"].endswith("-v2")
        outputs.append(config["experiment"]["output_dir"])
    assert len(outputs) == len(set(outputs))


def test_weak_v2_configs_use_frozen_pilot_split_and_existing_data():
    for name in sorted(WEAK_CONFIGS):
        data = _load(name)["data"]
        assert data["dataset_kind"] == "page_directory"
        assert data["data_dir"] == "data/processed"
        assert data["annotation_name"] == "weak_intent.json"
        assert data["split_manifest"] == "data/intent_pilot_split.json"
        assert (REPO_ROOT / data["data_dir"]).is_dir()
        assert (REPO_ROOT / data["split_manifest"]).is_file()


def test_reference_v2_configs_have_static_inputs_and_selection_tier():
    expected_train_tiers = {
        "gold_finetune.yaml": ["human_gold"],
        "reference_finetune.yaml": ["human_gold", "ai_silver"],
        "reference_ai_only.yaml": ["ai_silver"],
        "reference_finetune_w025.yaml": ["human_gold", "ai_silver"],
        "reference_finetune_w100.yaml": ["human_gold", "ai_silver"],
        **{
            name: ["human_gold", "ai_silver"]
            for name in REFERENCE_ABLATION_PAIRS
        },
    }
    expected_silver_weights = {
        "reference_finetune.yaml": 0.5,
        "reference_finetune_w025.yaml": 0.25,
        "reference_finetune_w100.yaml": 1.0,
        **{name: 0.5 for name in REFERENCE_ABLATION_PAIRS},
    }

    for name in sorted(REFERENCE_CONFIGS):
        config = _load(name)
        data = config["data"]
        training = config["training"]
        assert data["dataset_kind"] == "reference"
        assert data["train_tiers"] == expected_train_tiers[name]
        assert data["validation_tiers"] == ["human_gold", "ai_silver"]
        assert training["model_selection_tier"] == "ai_silver"
        assert (REPO_ROOT / data["data_dir"]).is_dir()
        assert (REPO_ROOT / data["package_dir"]).is_dir()
        assert (REPO_ROOT / data["source_split_manifest"]).is_file()
        if name in expected_silver_weights:
            assert training["reference_tier_weights"]["ai_silver"] == (
                expected_silver_weights[name]
            )


def test_reference_ablation_configs_match_weak_model_and_checkpoint():
    for reference_name, weak_name in REFERENCE_ABLATION_PAIRS.items():
        reference = _load(reference_name)
        weak = _load(weak_name)
        assert reference["model"] == weak["model"]
        assert reference["training"]["init_checkpoint"] == (
            f"{weak['experiment']['output_dir']}/best.pt"
        )
