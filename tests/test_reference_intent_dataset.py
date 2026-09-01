from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pytest
import torch

from src.data import intent_dataset as intent_data
from src.design_intent.schema import (
    BBox,
    Canvas,
    DesignElement,
    DesignGroup,
    DesignIntentIR,
    LayoutConstraint,
    PageGraph,
    PageNode,
    TreeEdge,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/processed"
PACKAGE_DIR = REPO_ROOT / "data/annotations/intent_gold_v1"
SOURCE_SPLIT = REPO_ROOT / "data/intent_pilot_split.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _minimal_package(tmp_path: Path) -> Path:
    # 只复用 train/validation 开发样本，测试夹具不得读取正式 test IR。
    sample_ids = {"0001", "0051", "0079"}
    package_dir = tmp_path / "intent_reference"
    reference_dir = package_dir / "reference"
    reference_dir.mkdir(parents=True)
    for sample_id in sample_ids:
        shutil.copy2(
            PACKAGE_DIR / "reference" / f"{sample_id}.json",
            reference_dir / f"{sample_id}.json",
        )

    assignment = _json(PACKAGE_DIR / "assignment.json")
    assignment["samples"] = [
        item for item in assignment["samples"] if item["sample_id"] in sample_ids
    ]
    assignment["human_gold_sample_ids"] = ["0001"]
    assignment["ai_silver_sample_ids"] = ["0051", "0079"]
    assignment["locked_sample_ids"] = ["0001"]
    assignment["read_only_sample_ids"] = sorted(sample_ids)
    assignment["annotation_workflow"]["canonical_reference_dir"] = str(
        reference_dir.resolve()
    )
    _write_json(package_dir / "assignment.json", assignment)

    selection = _json(PACKAGE_DIR / "selection_manifest.json")
    selection["samples"] = [
        item for item in selection["samples"] if item["sample_id"] in sample_ids
    ]
    selection["splits"] = {
        split: [sample_id for sample_id in values if sample_id in sample_ids]
        for split, values in selection["splits"].items()
    }
    selection["existing_gold_ids"] = ["0001"]
    selection["annotation_required_ids"] = ["0051", "0079"]
    _write_json(package_dir / "selection_manifest.json", selection)

    reference_manifest = _json(PACKAGE_DIR / "ai_reference_manifest.json")
    reference_manifest["samples"] = [
        item
        for item in reference_manifest["samples"]
        if item["sample_id"] in {"0051", "0079"}
    ]
    reference_manifest["ai_silver_samples"] = 2
    reference_manifest["human_gold_samples"] = 1
    reference_manifest["human_gold_files"] = [
        item
        for item in reference_manifest["human_gold_files"]
        if item["sample_id"] == "0001"
    ]
    for item in [
        *reference_manifest["samples"],
        *reference_manifest["human_gold_files"],
    ]:
        sample_id = item["sample_id"]
        item["reference_path"] = str(
            (reference_dir / f"{sample_id}.json").resolve()
        )
        item["reference_sha256"] = _sha256(
            reference_dir / f"{sample_id}.json"
        )
    reference_manifest["canonical_reference_dir"] = str(reference_dir.resolve())
    _write_json(package_dir / "ai_reference_manifest.json", reference_manifest)
    return package_dir


def _style() -> dict:
    return {
        "display": "block",
        "backgroundColor": [0.0, 0.0, 0.0, 0.0],
        "color": [0.0, 0.0, 0.0, 1.0],
        "borderColor": [0.0, 0.0, 0.0, 0.0],
        "borderRadius": 0,
        "borderWidth": 0,
        "fontSize": 16,
        "fontWeight": 400,
    }


def _graph(node_count: int = 4) -> PageGraph:
    return PageGraph(
        schema_version="1.0",
        sample_id="synthetic",
        canvas=Canvas(400, 100),
        nodes=[
            PageNode(
                id=index,
                parent_id=None,
                tag="div",
                text=f"node-{index}",
                attributes={},
                bbox=BBox(index * 100, 0, 80, 80),
                computed_style=_style(),
            )
            for index in range(node_count)
        ],
    )


def _element(element_id: str, *source_ids: int) -> DesignElement:
    first = source_ids[0]
    return DesignElement(
        id=element_id,
        source_node_ids=list(source_ids),
        type="TEXT",
        bbox=BBox(first * 100, 0, 80, 80),
        name=element_id,
        text=element_id,
    )


def _ir(
    elements: list[DesignElement],
    groups: list[DesignGroup],
    tree: list[TreeEdge],
) -> DesignIntentIR:
    return DesignIntentIR(
        schema_version="1.0",
        canvas=Canvas(400, 100),
        elements=elements,
        groups=groups,
        layouts=[
            LayoutConstraint(
                target_id=group.id,
                mode="HORIZONTAL",
                gap=10,
                padding=[0, 0, 0, 0],
            )
            for group in groups
        ],
        tree=tree,
        style_tokens=[],
        provenance={},
    )


def test_reference_dataset_type_is_available():
    assert hasattr(intent_data, "ReferenceIntentDataset")


@pytest.mark.parametrize(
    ("split", "expected_total", "expected_counts"),
    [
        ("train", 30, {"human_gold": 7, "ai_silver": 23}),
        ("validation", 10, {"human_gold": 1, "ai_silver": 9}),
    ],
)
def test_reference_dataset_loads_frozen_split_counts_and_tiers(
    split: str,
    expected_total: int,
    expected_counts: dict[str, int],
):
    dataset = intent_data.ReferenceIntentDataset(
        data_dir=str(DATA_DIR),
        package_dir=str(PACKAGE_DIR),
        split=split,
        max_nodes=192,
        source_split_manifest=str(SOURCE_SPLIT),
        verify_hashes=True,
    )

    assert len(dataset) == expected_total
    assert dataset.tier_counts == expected_counts
    assert len(dataset.reference_tiers) == len(dataset)
    for tier, count in expected_counts.items():
        assert dataset.reference_tiers.count(tier) == count


def test_reference_collate_preserves_optional_metadata():
    dataset = intent_data.ReferenceIntentDataset(
        data_dir=str(DATA_DIR),
        package_dir=str(PACKAGE_DIR),
        split="train",
        max_nodes=192,
        tiers=["human_gold", "ai_silver"],
        source_split_manifest=str(SOURCE_SPLIT),
        verify_hashes=False,
    )
    human_index = dataset.reference_tiers.index("human_gold")
    silver_index = dataset.reference_tiers.index("ai_silver")

    batch = intent_data.intent_collate_fn(
        [dataset[human_index], dataset[silver_index]]
    )

    assert batch["reference_tier"] == ["human_gold", "ai_silver"]
    assert batch["label_source"] == [
        "human_ai_adjudicated_gold",
        "ai_multiview_silver",
    ]
    assert torch.equal(
        batch["token_supervision_available"],
        torch.tensor([False, True]),
    )
    assert batch["token_supervision_scope"] == [
        "not_collected",
        "ai_computed_style_proxy",
    ]
    assert torch.equal(
        batch["negative_action_supervision_available"],
        torch.tensor([False, True]),
    )


def test_reference_dataset_filters_by_split_without_test_leakage(tmp_path: Path):
    package_dir = _minimal_package(tmp_path)
    dataset = intent_data.ReferenceIntentDataset(
        data_dir=str(DATA_DIR),
        package_dir=str(package_dir),
        split="train",
        max_nodes=192,
        source_split_manifest=str(SOURCE_SPLIT),
        verify_hashes=True,
    )

    assert len(dataset) == 2
    assert [sample.sample_id for sample in dataset.samples] == ["0001", "0051"]


def test_train_and_validation_do_not_open_or_hash_test_sample_files(monkeypatch):
    selection = _json(PACKAGE_DIR / "selection_manifest.json")
    test_ids = set(selection["splits"]["test"])
    active_ids = set(selection["splits"]["train"]) | set(
        selection["splits"]["validation"]
    )
    opened_reference_ids: set[str] = set()
    hashed_sample_ids: set[str] = set()
    real_load_json = intent_data._load_json_object
    real_sha256 = intent_data._sha256

    def tracking_load_json(path: Path):
        path = Path(path)
        if path.parent.name == "reference":
            opened_reference_ids.add(path.stem)
        return real_load_json(path)

    def tracking_sha256(path: Path):
        path = Path(path)
        if path.parent.name == "reference":
            hashed_sample_ids.add(path.stem)
        elif path.name in {"screenshot.png", "page_graph.json"}:
            hashed_sample_ids.add(path.parent.name)
        return real_sha256(path)

    monkeypatch.setattr(intent_data, "_load_json_object", tracking_load_json)
    monkeypatch.setattr(intent_data, "_sha256", tracking_sha256)
    datasets = []
    for split in ("train", "validation"):
        datasets.append(intent_data.ReferenceIntentDataset(
            data_dir=str(DATA_DIR),
            package_dir=str(PACKAGE_DIR),
            split=split,
            max_nodes=192,
            source_split_manifest=str(SOURCE_SPLIT),
            verify_hashes=True,
        ))

    assert opened_reference_ids == active_ids
    assert hashed_sample_ids == active_ids
    assert opened_reference_ids.isdisjoint(test_ids)
    assert hashed_sample_ids.isdisjoint(test_ids)
    expected_counts = {
        "train": {"human_gold": 7, "ai_silver": 23},
        "validation": {"human_gold": 1, "ai_silver": 9},
        "test": {"human_gold": 2, "ai_silver": 18},
    }
    assert datasets[0].all_split_tier_counts == expected_counts
    assert datasets[1].all_split_tier_counts == expected_counts


def test_formal_test_requires_unseal_before_opening_active_files(
    tmp_path: Path,
    monkeypatch,
):
    package_dir = _minimal_package(tmp_path)
    assignment_path = package_dir / "assignment.json"
    assignment = _json(assignment_path)
    next(
        item for item in assignment["samples"] if item["sample_id"] == "0051"
    )["split"] = "test"
    _write_json(assignment_path, assignment)

    selection_path = package_dir / "selection_manifest.json"
    selection = _json(selection_path)
    next(
        item for item in selection["samples"] if item["sample_id"] == "0051"
    )["split"] = "test"
    selection["splits"]["train"].remove("0051")
    selection["splits"]["test"].append("0051")
    _write_json(selection_path, selection)

    source_split = _json(SOURCE_SPLIT)
    source_split["splits"]["train"].remove("0051")
    source_split["splits"]["test"].append("0051")
    source_split_path = tmp_path / "source_split.json"
    _write_json(source_split_path, source_split)

    unseal_path = tmp_path / "audit/test_unseal.json"
    opened_active_paths: list[Path] = []
    real_load_json = intent_data._load_json_object
    real_sha256 = intent_data._sha256

    def tracking_load_json(path: Path):
        path = Path(path)
        if path.parent.name == "reference":
            opened_active_paths.append(path)
        return real_load_json(path)

    def tracking_sha256(path: Path):
        path = Path(path)
        if path.parent.name == "reference" or path.name in {
            "screenshot.png",
            "page_graph.json",
        }:
            opened_active_paths.append(path)
        return real_sha256(path)

    monkeypatch.setattr(
        intent_data, "_FORMAL_REFERENCE_PACKAGE", package_dir.resolve()
    )
    monkeypatch.setattr(intent_data, "_FORMAL_TEST_UNSEAL_PATH", unseal_path)
    monkeypatch.setattr(intent_data, "_load_json_object", tracking_load_json)
    monkeypatch.setattr(intent_data, "_sha256", tracking_sha256)

    metadata = intent_data.ReferenceIntentDataset(
        data_dir=str(DATA_DIR),
        package_dir=str(package_dir),
        split="test",
        max_nodes=192,
        source_split_manifest=str(source_split_path),
        verify_hashes=True,
        _metadata_only=True,
    )
    assert [sample.sample_id for sample in metadata.samples] == ["0051"]
    with pytest.raises(RuntimeError, match="仅元数据"):
        metadata[0]
    assert opened_active_paths == []
    assert not unseal_path.exists()

    with pytest.raises(PermissionError, match="尚未解封"):
        intent_data.ReferenceIntentDataset(
            data_dir=str(DATA_DIR),
            package_dir=str(package_dir),
            split="test",
            max_nodes=192,
            source_split_manifest=str(source_split_path),
            verify_hashes=True,
        )

    assert opened_active_paths == []
    assert not unseal_path.exists()

    _write_json(unseal_path, {})
    with pytest.raises(ValueError, match="test unseal"):
        intent_data.ReferenceIntentDataset(
            data_dir=str(DATA_DIR),
            package_dir=str(package_dir),
            split="test",
            max_nodes=192,
            source_split_manifest=str(source_split_path),
            verify_hashes=True,
        )
    assert opened_active_paths == []


@pytest.mark.parametrize(
    ("conflict", "message"),
    [
        ("duplicate_assignment", "assignment 包含重复 sample ID：0001"),
        ("selection_split", "样本 0001 的 selection_manifest.samples"),
        ("source_split", "样本 0001 的 source_split_manifest split"),
        ("provenance", "样本 0001 的 provenance.reference_tier"),
    ],
)
def test_reference_dataset_fails_closed_on_manifest_conflicts(
    tmp_path: Path,
    conflict: str,
    message: str,
):
    package_dir = _minimal_package(tmp_path)
    source_split = SOURCE_SPLIT

    if conflict == "duplicate_assignment":
        path = package_dir / "assignment.json"
        payload = _json(path)
        payload["samples"].append(deepcopy(payload["samples"][0]))
        _write_json(path, payload)
    elif conflict == "selection_split":
        path = package_dir / "selection_manifest.json"
        payload = _json(path)
        next(item for item in payload["samples"] if item["sample_id"] == "0001")[
            "split"
        ] = "test"
        _write_json(path, payload)
    elif conflict == "source_split":
        payload = _json(SOURCE_SPLIT)
        payload["splits"]["train"].remove("0001")
        payload["splits"]["test"].append("0001")
        source_split = tmp_path / "source_split.json"
        _write_json(source_split, payload)
    else:
        path = package_dir / "reference/0001.json"
        payload = _json(path)
        payload["provenance"]["reference_tier"] = "ai_silver"
        _write_json(path, payload)

    with pytest.raises(ValueError, match=message):
        intent_data.ReferenceIntentDataset(
            data_dir=str(DATA_DIR),
            package_dir=str(package_dir),
            split="train",
            max_nodes=192,
            source_split_manifest=str(source_split),
            verify_hashes=False,
        )


@pytest.mark.parametrize(
    ("conflict", "message"),
    [
        (
            "human_top_level",
            "human_gold_token_labels_available",
        ),
        (
            "human_file",
            "human_gold_files.*token_labels_available",
        ),
        (
            "silver_policy",
            "token_policy.evaluation_scope",
        ),
    ],
)
def test_reference_dataset_rejects_token_package_contract_conflicts(
    tmp_path: Path,
    conflict: str,
    message: str,
):
    package_dir = _minimal_package(tmp_path)
    manifest_path = package_dir / "ai_reference_manifest.json"
    manifest = _json(manifest_path)
    if conflict == "human_top_level":
        manifest["human_gold_token_labels_available"] = True
    elif conflict == "human_file":
        manifest["human_gold_files"][0]["token_labels_available"] = True
    else:
        manifest["token_policy"]["evaluation_scope"] = "human_truth"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match=message):
        intent_data.ReferenceIntentDataset(
            data_dir=str(DATA_DIR),
            package_dir=str(package_dir),
            split="train",
            max_nodes=192,
            source_split_manifest=str(SOURCE_SPLIT),
            verify_hashes=False,
        )


@pytest.mark.parametrize(
    ("manifest_name", "mutate", "message"),
    [
        (
            "selection_manifest.json",
            lambda payload: payload["samples"][0]["asset_sha256"].__setitem__(
                "screenshot", "not-a-sha256"
            ),
            "asset_sha256.screenshot",
        ),
        (
            "ai_reference_manifest.json",
            lambda payload: payload["human_gold_files"][0].__setitem__(
                "reference_sha256", None
            ),
            "reference_sha256",
        ),
    ],
)
def test_metadata_preflight_rejects_malformed_recorded_hashes(
    tmp_path: Path,
    manifest_name: str,
    mutate,
    message: str,
):
    package_dir = _minimal_package(tmp_path)
    manifest_path = package_dir / manifest_name
    payload = _json(manifest_path)
    mutate(payload)
    _write_json(manifest_path, payload)

    with pytest.raises(ValueError, match=message):
        intent_data.ReferenceIntentDataset(
            data_dir=str(DATA_DIR),
            package_dir=str(package_dir),
            split="train",
            max_nodes=192,
            source_split_manifest=str(SOURCE_SPLIT),
            verify_hashes=True,
            _metadata_only=True,
        )


def test_manifest_hashes_use_the_same_byte_snapshot_as_parsing(
    tmp_path: Path,
    monkeypatch,
):
    package_dir = _minimal_package(tmp_path)
    manifest_paths = {
        "assignment": package_dir / "assignment.json",
        "selection_manifest": package_dir / "selection_manifest.json",
        "ai_reference_manifest": package_dir / "ai_reference_manifest.json",
        "source_split": SOURCE_SPLIT,
    }
    expected = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in manifest_paths.items()
    }
    real_sha256 = intent_data._sha256

    def reject_manifest_reopen(path: Path):
        path = Path(path).resolve()
        if path in {item.resolve() for item in manifest_paths.values()}:
            raise AssertionError("manifest 不得在解析后重新打开计算哈希")
        return real_sha256(path)

    monkeypatch.setattr(intent_data, "_sha256", reject_manifest_reopen)

    dataset = intent_data.ReferenceIntentDataset(
        data_dir=str(DATA_DIR),
        package_dir=str(package_dir),
        split="train",
        max_nodes=192,
        source_split_manifest=str(SOURCE_SPLIT),
        verify_hashes=True,
        _metadata_only=True,
    )

    assert dataset.manifest_hashes == expected


@pytest.mark.parametrize(
    "noncanonical_path",
    ["absolute_asset", "dotdot_asset", "absolute_reference"],
)
def test_repo_package_rejects_noncanonical_manifest_paths(
    noncanonical_path: str,
):
    with tempfile.TemporaryDirectory(
        prefix=".reference-dataset-test-",
        dir=REPO_ROOT,
    ) as raw_tmp_dir:
        package_dir = _minimal_package(Path(raw_tmp_dir))
        reference_dir = package_dir / "reference"
        relative_reference_dir = reference_dir.relative_to(REPO_ROOT).as_posix()

        assignment_path = package_dir / "assignment.json"
        assignment = _json(assignment_path)
        assignment["annotation_workflow"][
            "canonical_reference_dir"
        ] = relative_reference_dir
        if noncanonical_path == "absolute_asset":
            next(
                item for item in assignment["samples"]
                if item["sample_id"] == "0001"
            )["screenshot"] = str(
                (DATA_DIR / "0001/screenshot.png").resolve()
            )
        elif noncanonical_path == "dotdot_asset":
            next(
                item for item in assignment["samples"]
                if item["sample_id"] == "0001"
            )["page_graph"] = "data/processed/0001/../0001/page_graph.json"
        _write_json(assignment_path, assignment)

        reference_manifest_path = package_dir / "ai_reference_manifest.json"
        reference_manifest = _json(reference_manifest_path)
        reference_manifest["canonical_reference_dir"] = relative_reference_dir
        for item in [
            *reference_manifest["samples"],
            *reference_manifest["human_gold_files"],
        ]:
            item["reference_path"] = (
                reference_dir / f"{item['sample_id']}.json"
            ).relative_to(REPO_ROOT).as_posix()
        if noncanonical_path == "absolute_reference":
            reference_manifest["human_gold_files"][0]["reference_path"] = str(
                (reference_dir / "0001.json").resolve()
            )
        _write_json(reference_manifest_path, reference_manifest)

        with pytest.raises(ValueError, match="规范仓库相对 POSIX 路径"):
            intent_data.ReferenceIntentDataset(
                data_dir=str(DATA_DIR),
                package_dir=str(package_dir),
                split="train",
                max_nodes=192,
                source_split_manifest=str(SOURCE_SPLIT),
                verify_hashes=False,
            )


def test_reference_dataset_rejects_asset_path_or_hash_mismatch(
    tmp_path: Path,
):
    package_dir = _minimal_package(tmp_path)
    assignment_path = package_dir / "assignment.json"
    assignment = _json(assignment_path)
    next(item for item in assignment["samples"] if item["sample_id"] == "0001")[
        "screenshot"
    ] = "data/processed/0051/screenshot.png"
    _write_json(assignment_path, assignment)

    with pytest.raises(ValueError, match="assignment.screenshot"):
        intent_data.ReferenceIntentDataset(
            data_dir=str(DATA_DIR),
            package_dir=str(package_dir),
            split="train",
            max_nodes=192,
            source_split_manifest=str(SOURCE_SPLIT),
            verify_hashes=True,
        )

    assignment = _json(PACKAGE_DIR / "assignment.json")
    assignment["samples"] = [
        item
        for item in assignment["samples"]
        if item["sample_id"] in {"0001", "0051", "0079"}
    ]
    assignment["human_gold_sample_ids"] = ["0001"]
    assignment["ai_silver_sample_ids"] = ["0051", "0079"]
    assignment["locked_sample_ids"] = ["0001"]
    assignment["read_only_sample_ids"] = ["0001", "0051", "0079"]
    assignment["annotation_workflow"]["canonical_reference_dir"] = str(
        (package_dir / "reference").resolve()
    )
    _write_json(assignment_path, assignment)
    selection_path = package_dir / "selection_manifest.json"
    selection = _json(selection_path)
    next(item for item in selection["samples"] if item["sample_id"] == "0001")[
        "asset_sha256"
    ]["screenshot"] = "0" * 64
    _write_json(selection_path, selection)

    with pytest.raises(ValueError, match="screenshot SHA-256"):
        intent_data.ReferenceIntentDataset(
            data_dir=str(DATA_DIR),
            package_dir=str(package_dir),
            split="train",
            max_nodes=192,
            source_split_manifest=str(SOURCE_SPLIT),
            verify_hashes=True,
        )


def test_human_supervision_masks_unknown_actions_and_uncollected_tokens():
    elements = [_element("e0", 0, 1), _element("e2", 2)]
    group = DesignGroup(
        id="g0",
        source_element_ids=["e0", "e2"],
        role="CARD",
        bbox=BBox(0, 0, 380, 80),
        name="group",
        source_node_id=3,
    )
    ir = _ir(
        elements,
        [group],
        [
            TreeEdge("g0", "e0", 0),
            TreeEdge("g0", "e2", 1),
            TreeEdge("page_root", "g0", 0),
        ],
    )

    human = intent_data.build_intent_targets(
        _graph(), ir, max_nodes=4, reference_tier="human_gold"
    )
    silver = intent_data.build_intent_targets(
        _graph(), ir, max_nodes=4, reference_tier="ai_silver"
    )

    assert human["action_mask"].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert human["action"].tolist() == [1, 0, 1, 2]
    assert human["token_pair_mask"].count_nonzero() == 0
    assert silver["action_mask"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert silver["token_pair_mask"].count_nonzero() > 0


def test_root_elements_are_negative_group_pairs():
    elements = [_element("e0", 0), _element("e1", 1)]
    ir = _ir(
        elements,
        [],
        [
            TreeEdge("page_root", "e0", 0),
            TreeEdge("page_root", "e1", 1),
        ],
    )

    targets = intent_data.build_intent_targets(_graph(2), ir, max_nodes=2)

    assert targets["group_pair_mask"][0, 1] == 1
    assert targets["group_affinity"][0, 1] == 0


def test_tree_mask_excludes_unrepresentable_parent_without_rewriting_to_root():
    elements = [_element("e0", 0), _element("e1", 1), _element("e2", 2)]
    group = DesignGroup(
        id="g_unanchored",
        source_element_ids=["e0", "e1"],
        role="CARD",
        bbox=BBox(0, 0, 180, 80),
        name="unanchored",
        source_node_id=None,
    )
    ir = _ir(
        elements,
        [group],
        [
            TreeEdge("g_unanchored", "e0", 0),
            TreeEdge("g_unanchored", "e1", 1),
            TreeEdge("page_root", "g_unanchored", 0),
            TreeEdge("page_root", "e2", 1),
        ],
    )

    targets = intent_data.build_intent_targets(_graph(3), ir, max_nodes=3)

    assert targets["tree_mask"].tolist() == [0.0, 0.0, 1.0]
    assert targets["parent"].tolist() == [-100, -100, 3]


def test_page_directory_dataset_keeps_old_api_without_reference_metadata():
    dataset = intent_data.IntentDataset(
        data_dir=str(DATA_DIR),
        max_nodes=192,
        annotation_name="weak_intent.json",
        sample_ids=["0001"],
    )

    item = dataset[0]
    batch = intent_data.intent_collate_fn([item])

    assert item["sample_id"] == "0001"
    assert "reference_tier" not in item
    assert "reference_tier" not in batch
    assert batch["node_mask"].shape == (1, 192)
