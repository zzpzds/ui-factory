from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import train_intent
from src.training.intent_losses import compute_intent_losses


ANNOTATION_TASKS = {
    "action",
    "element",
    "group",
    "role",
    "layout_mode",
    "layout_reg",
    "layout_align",
    "token",
    "tree",
}


def _loss_inputs(batch_size: int = 2, node_count: int = 2):
    def logits(*shape: int) -> torch.Tensor:
        return torch.zeros(*shape, requires_grad=True)

    outputs = {
        "action_logits": logits(batch_size, node_count, 4),
        "element_type_logits": logits(batch_size, node_count, 6),
        "group_affinity_logits": logits(batch_size, node_count, node_count),
        "role_logits": logits(batch_size, node_count, 10),
        "layout_mode_logits": logits(batch_size, node_count, 4),
        "primary_align_logits": logits(batch_size, node_count, 4),
        "cross_align_logits": logits(batch_size, node_count, 4),
        "gap": logits(batch_size, node_count),
        "padding": logits(batch_size, node_count, 4),
        "token_affinity_logits": logits(batch_size, 6, node_count, node_count),
        "parent_logits": logits(batch_size, node_count, node_count + 1),
        "alignment_logits": logits(batch_size, node_count, 3),
    }
    targets = {
        "action": torch.zeros(batch_size, node_count, dtype=torch.long),
        "action_confidence": torch.ones(batch_size, node_count),
        "action_mask": torch.zeros(batch_size, node_count),
        "element_type": torch.full(
            (batch_size, node_count), -100, dtype=torch.long
        ),
        "role": torch.full((batch_size, node_count), -100, dtype=torch.long),
        "layout_mode": torch.full(
            (batch_size, node_count), -100, dtype=torch.long
        ),
        "primary_align": torch.full(
            (batch_size, node_count), -100, dtype=torch.long
        ),
        "cross_align": torch.full(
            (batch_size, node_count), -100, dtype=torch.long
        ),
        "gap": torch.zeros(batch_size, node_count),
        "padding": torch.zeros(batch_size, node_count, 4),
        "layout_mask": torch.zeros(batch_size, node_count),
        "selected_mask": torch.zeros(batch_size, node_count),
        "tree_mask": torch.zeros(batch_size, node_count),
        "parent": torch.full(
            (batch_size, node_count), -100, dtype=torch.long
        ),
        "group_affinity": torch.zeros(batch_size, node_count, node_count),
        "group_pair_mask": torch.zeros(batch_size, node_count, node_count),
        "token_affinity": torch.zeros(
            batch_size, 6, node_count, node_count
        ),
        "token_pair_mask": torch.zeros(
            batch_size, 6, node_count, node_count
        ),
    }
    node_mask = torch.ones(batch_size, node_count)
    alignment_targets = torch.zeros(batch_size, node_count, 3)
    return outputs, targets, node_mask, alignment_targets


def _only_loss(name: str) -> dict[str, float]:
    weights = {task: 0.0 for task in ANNOTATION_TASKS | {"align"}}
    weights[name] = 1.0
    return weights


def test_annotation_page_weights_control_page_gradients():
    outputs, targets, node_mask, alignment_targets = _loss_inputs()
    targets["action_mask"][:, 0] = 1

    losses = compute_intent_losses(
        outputs,
        targets,
        node_mask,
        alignment_targets=alignment_targets,
        weights=_only_loss("action"),
        annotation_page_weights=torch.tensor([4 / 3, 2 / 3]),
    )
    losses["total"].backward()

    gradients = outputs["action_logits"].grad[:, 0].norm(dim=-1)
    assert gradients[0] / gradients[1] == pytest.approx(2.0)


def test_alignment_loss_does_not_use_annotation_page_weights():
    outputs, targets, node_mask, alignment_targets = _loss_inputs()

    losses = compute_intent_losses(
        outputs,
        targets,
        node_mask,
        alignment_targets=alignment_targets,
        weights=_only_loss("align"),
        annotation_page_weights=torch.tensor([4 / 3, 2 / 3]),
    )
    losses["total"].backward()

    gradients = outputs["alignment_logits"].grad.norm(dim=(1, 2))
    assert gradients[0] == pytest.approx(float(gradients[1]))


def test_task_without_labels_returns_connected_zero():
    outputs, targets, node_mask, alignment_targets = _loss_inputs(batch_size=1)

    losses = compute_intent_losses(
        outputs,
        targets,
        node_mask,
        alignment_targets=alignment_targets,
        weights=_only_loss("token"),
        annotation_page_weights=torch.ones(1),
    )
    losses["total"].backward()

    assert losses["token"].item() == 0.0
    assert outputs["token_affinity_logits"].grad is not None
    assert torch.count_nonzero(outputs["token_affinity_logits"].grad) == 0


def test_human_token_logits_do_not_change_loss_or_receive_gradients():
    outputs, targets, node_mask, _ = _loss_inputs()
    targets["token_pair_mask"][1, 0, 0, 1] = 1
    targets["token_affinity"][1, 0, 0, 1] = 1

    baseline = compute_intent_losses(
        outputs,
        targets,
        node_mask,
        weights=_only_loss("token"),
    )["total"].detach()
    with torch.no_grad():
        outputs["token_affinity_logits"][0].fill_(100.0)
    changed = compute_intent_losses(
        outputs,
        targets,
        node_mask,
        weights=_only_loss("token"),
    )["total"]
    changed.backward()

    assert float(changed.detach()) == pytest.approx(float(baseline))
    assert torch.count_nonzero(outputs["token_affinity_logits"].grad[0]) == 0
    assert torch.count_nonzero(outputs["token_affinity_logits"].grad[1]) > 0


def test_all_annotation_tasks_without_labels_are_finite_connected_zeros():
    outputs, targets, node_mask, _ = _loss_inputs()

    losses = compute_intent_losses(
        outputs,
        targets,
        node_mask,
        alignment_targets=None,
    )
    losses["total"].backward()

    for name in ANNOTATION_TASKS | {"total"}:
        assert torch.isfinite(losses[name])
        assert losses[name].item() == 0.0


def test_unlabeled_page_remains_in_annotation_batch_denominator():
    outputs, targets, node_mask, _ = _loss_inputs()
    targets["action_mask"][0, 0] = 1
    targets["action"][0, 0] = 1

    batch_loss = compute_intent_losses(
        outputs,
        targets,
        node_mask,
        weights=_only_loss("action"),
    )["action"]
    one_page_loss = compute_intent_losses(
        {key: value[:1] for key, value in outputs.items()},
        {key: value[:1] for key, value in targets.items()},
        node_mask[:1],
        weights=_only_loss("action"),
    )["action"]

    assert float(batch_loss.detach()) == pytest.approx(
        float(one_page_loss.detach() / 2)
    )


def test_action_and_tree_masks_override_legacy_masks():
    outputs, targets, node_mask, alignment_targets = _loss_inputs(
        batch_size=1,
        node_count=3,
    )
    targets["action_mask"][0, :2] = 1
    targets["action"][0, 0] = 1
    targets["action"][0, 1] = 2
    targets["selected_mask"][0, :] = 1
    targets["tree_mask"][0, 0] = 1
    targets["parent"][0, :] = 2

    losses = compute_intent_losses(
        outputs,
        targets,
        node_mask,
        alignment_targets=alignment_targets,
        weights={
            **_only_loss("action"),
            "tree": 1.0,
        },
    )
    losses["total"].backward()

    assert torch.count_nonzero(outputs["action_logits"].grad[0, 0]) > 0
    assert torch.count_nonzero(outputs["action_logits"].grad[0, 1]) > 0
    assert torch.count_nonzero(outputs["action_logits"].grad[0, 2]) == 0
    assert torch.count_nonzero(outputs["parent_logits"].grad[0, 0]) > 0
    assert torch.count_nonzero(outputs["parent_logits"].grad[0, 1]) == 0
    assert torch.count_nonzero(outputs["parent_logits"].grad[0, 2]) == 0


def test_normalize_reference_tier_weights_once_over_train_pages():
    tiers = ["human_gold"] * 7 + ["ai_silver"] * 23

    normalized = train_intent.normalize_reference_tier_weights(
        tiers,
        {"human_gold": 1.0, "ai_silver": 0.5},
    )

    page_weights = [normalized[tier] for tier in tiers]
    assert sum(page_weights) / len(page_weights) == pytest.approx(1.0)
    assert normalized["human_gold"] / normalized["ai_silver"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "weights",
    [
        {"human_gold": 1.0},
        {"human_gold": 1.0, "ai_silver": 0.0},
        {"human_gold": 1.0, "ai_silver": math.inf},
    ],
)
def test_normalize_reference_tier_weights_rejects_invalid_active_weights(weights):
    with pytest.raises(ValueError):
        train_intent.normalize_reference_tier_weights(
            ["human_gold", "ai_silver"], weights
        )


def test_microbatch_page_accumulation_matches_direct_page_mean():
    micro_parameter = torch.tensor(2.0, requires_grad=True)
    pages = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0])]
    for values in pages:
        page_mean_loss = (micro_parameter * values).mean()
        train_intent.backward_page_mean(page_mean_loss, len(values))
    train_intent.normalize_accumulated_gradients([micro_parameter], 5)

    direct_parameter = torch.tensor(2.0, requires_grad=True)
    (direct_parameter * torch.arange(1.0, 6.0)).mean().backward()

    assert micro_parameter.grad == pytest.approx(float(direct_parameter.grad))


def test_missing_init_checkpoint_fails_before_training(tmp_path: Path):
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="init checkpoint"):
        train_intent.require_init_checkpoint(str(missing))


def test_checkpoint_selection_uses_ai_silver_unweighted_total():
    validation = {
        "overall": {"total": 100.0},
        "by_tier": {
            "human_gold": {"total": 50.0},
            "ai_silver": {"total": 0.75},
        },
    }

    selected = train_intent.validation_selection_loss(
        validation, model_selection_tier="ai_silver"
    )

    assert selected == 0.75


@pytest.mark.parametrize(
    ("validation_tiers", "selection_tier"),
    [
        (["human_gold", "ai_silver"], "human_gold"),
        (["human_gold"], "ai_silver"),
        (["ai_silver"], "ai_silver"),
    ],
)
def test_reference_selection_contract_requires_ai_silver_validation(
    validation_tiers,
    selection_tier,
):
    with pytest.raises(ValueError, match="ai_silver"):
        train_intent.validate_reference_selection_tier(
            validation_tiers,
            selection_tier,
        )


def test_reference_selection_contract_accepts_ai_silver_validation():
    train_intent.validate_reference_selection_tier(
        ["human_gold", "ai_silver"],
        "ai_silver",
    )


def test_reference_selection_contract_accepts_actual_validation_distribution():
    train_intent.validate_reference_selection_tier(
        ["human_gold"] + ["ai_silver"] * 9,
        "ai_silver",
    )


def test_reference_dataset_routing_uses_separate_train_and_validation_tiers(
    monkeypatch,
):
    calls = []

    class FakeReferenceDataset:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.reference_tiers = list(kwargs["tiers"])
            self.tier_counts = {tier: 1 for tier in kwargs["tiers"]}

        def __len__(self):
            return 1

    monkeypatch.setattr(
        train_intent, "ReferenceIntentDataset", FakeReferenceDataset, raising=False
    )
    data_config = {
        "dataset_kind": "reference",
        "data_dir": "data/processed",
        "package_dir": "data/annotations/intent_gold_v1",
        "source_split_manifest": "data/intent_pilot_split.json",
        "train_tiers": ["human_gold"],
        "validation_tiers": ["human_gold", "ai_silver"],
        "verify_hashes": True,
        "max_nodes": 192,
    }

    train_set, validation_set = train_intent.build_datasets(data_config)

    assert train_set.reference_tiers == ["human_gold"]
    assert validation_set.reference_tiers == ["human_gold", "ai_silver"]
    assert calls[0]["split"] == "train"
    assert calls[0]["tiers"] == ["human_gold"]
    assert calls[1]["split"] == "validation"
    assert calls[1]["tiers"] == ["human_gold", "ai_silver"]


def test_reference_validation_calls_loss_without_annotation_weights(monkeypatch):
    seen_page_weights = []

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, batch):
            return {"marker": batch["marker"]}

    def fake_losses(
        outputs,
        targets,
        node_mask,
        alignment_targets=None,
        weights=None,
        annotation_page_weights=None,
    ):
        seen_page_weights.append(annotation_page_weights)
        return {"total": outputs["marker"].float().mean()}

    monkeypatch.setattr(train_intent, "compute_intent_losses", fake_losses)
    loader = [
        {
            "sample_id": ["h", "s"],
            "reference_tier": ["human_gold", "ai_silver"],
            "marker": torch.tensor([[2.0], [6.0]]),
            "targets": {"unused": torch.zeros(2, 1)},
            "node_mask": torch.ones(2, 1),
            "alignment_targets": torch.zeros(2, 1, 1),
        }
    ]

    result = train_intent.evaluate(
        FakeModel(),
        loader,
        torch.device("cpu"),
        {},
        stratify_by_tier=True,
    )

    assert seen_page_weights and all(value is None for value in seen_page_weights)
    assert result["overall"]["total"] == 4.0
    assert result["by_tier"]["human_gold"]["total"] == 2.0
    assert result["by_tier"]["ai_silver"]["total"] == 6.0


def test_index_batch_tensors_moves_indices_to_each_tensor_device():
    class DeviceCheckingTensor:
        def __init__(self, device: str):
            self.device = torch.device(device)
            self.index_device = None

        def index_select(self, dimension, indices):
            assert dimension == 0
            self.index_device = indices.device
            return self

    cpu_value = DeviceCheckingTensor("cpu")
    meta_value = DeviceCheckingTensor("meta")

    train_intent._index_batch_tensors(
        {"cpu": cpu_value, "meta": meta_value},
        torch.tensor([0]),
    )

    assert cpu_value.index_device == torch.device("cpu")
    assert meta_value.index_device == torch.device("meta")


def test_run_manifest_hashes_serialized_checkpoints(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"serialized checkpoint")

    train_intent.write_run_manifest(
        tmp_path,
        {"best": checkpoint},
    )

    manifest = json.loads(
        (tmp_path / "run_manifest.json").read_text(encoding="utf-8")
    )
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert manifest["checkpoints"]["best"]["sha256"] == expected
    assert manifest["checkpoints"]["best"]["path"] == str(checkpoint)


def test_training_audit_uses_dataset_frozen_manifest_hashes():
    frozen_hashes = {
        "assignment": "a" * 64,
        "selection_manifest": "b" * 64,
        "ai_reference_manifest": "c" * 64,
        "source_split": "d" * 64,
    }

    class FakeDataset:
        def __init__(self, counts):
            self.tier_counts = counts
            self.manifest_hashes = frozen_hashes
            self.all_split_tier_counts = {
                "train": {"human_gold": 7, "ai_silver": 23},
                "validation": {"human_gold": 1, "ai_silver": 9},
                "test": {"human_gold": 2, "ai_silver": 18},
            }

        def __len__(self):
            return sum(self.tier_counts.values())

    config = {
        "experiment": {"seed": 42},
        "data": {
            "dataset_kind": "reference",
            "package_dir": "must-not-be-read",
            "source_split_manifest": "must-not-be-read",
        },
    }

    audit = train_intent.build_training_audit(
        config,
        FakeDataset({"human_gold": 7, "ai_silver": 23}),
        FakeDataset({"human_gold": 1, "ai_silver": 9}),
        torch.device("cpu"),
        {"human_gold": 1.0, "ai_silver": 0.5},
        {"human_gold": 30 / 18.5, "ai_silver": 15 / 18.5},
        "2026-09-01T00:00:00+08:00",
    )

    assert audit["data_manifest_sha256"] == frozen_hashes
    assert audit["split_tier_counts"]["train"] == {
        "human_gold": 7,
        "ai_silver": 23,
    }
    assert audit["split_tier_counts"]["validation"] == {
        "human_gold": 1,
        "ai_silver": 9,
    }
    assert audit["split_tier_counts"]["test"] == {
        "human_gold": 2,
        "ai_silver": 18,
    }
    assert audit["active_split_tier_counts"] == {
        "train": {"human_gold": 7, "ai_silver": 23},
        "validation": {"human_gold": 1, "ai_silver": 9},
    }
    assert audit["training_finished_at"] is None


def test_training_audit_distinguishes_active_tiers_from_package_counts():
    package_counts = {
        "train": {"human_gold": 7, "ai_silver": 23},
        "validation": {"human_gold": 1, "ai_silver": 9},
        "test": {"human_gold": 2, "ai_silver": 18},
    }

    class FakeDataset:
        manifest_hashes = {"assignment": "a" * 64}
        all_split_tier_counts = package_counts

        def __init__(self, tier_counts):
            self.tier_counts = tier_counts

    audit = train_intent.build_training_audit(
        {
            "experiment": {"seed": 42},
            "data": {"dataset_kind": "reference"},
        },
        FakeDataset({"human_gold": 7}),
        FakeDataset({"human_gold": 1, "ai_silver": 9}),
        torch.device("cpu"),
        {"human_gold": 1.0},
        {"human_gold": 1.0},
        "2026-09-01T00:00:00+08:00",
    )

    assert audit["split_tier_counts"] == package_counts
    assert audit["active_split_tier_counts"] == {
        "train": {"human_gold": 7},
        "validation": {"human_gold": 1, "ai_silver": 9},
    }


def test_training_audit_rejects_mismatched_reference_count_snapshots():
    class FakeDataset:
        manifest_hashes = {"assignment": "a" * 64}
        tier_counts = {"human_gold": 1}

        def __init__(self, test_human_count: int):
            self.all_split_tier_counts = {
                "train": {"human_gold": 7, "ai_silver": 23},
                "validation": {"human_gold": 1, "ai_silver": 9},
                "test": {
                    "human_gold": test_human_count,
                    "ai_silver": 18,
                },
            }

    config = {
        "experiment": {"seed": 42},
        "data": {"dataset_kind": "reference"},
    }

    with pytest.raises(ValueError, match="tier count"):
        train_intent.build_training_audit(
            config,
            FakeDataset(test_human_count=2),
            FakeDataset(test_human_count=3),
            torch.device("cpu"),
            {},
            {},
            "2026-09-01T00:00:00+08:00",
        )


def test_reference_main_runs_one_epoch_with_overridden_mock_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    def make_batch(tiers: list[str]) -> dict:
        _, targets, node_mask, alignment_targets = _loss_inputs(
            batch_size=len(tiers),
            node_count=2,
        )
        targets["action_mask"][:, 0] = 1
        targets["action"][:, 0] = 1
        return {
            "sample_id": [f"{tier}-{index}" for index, tier in enumerate(tiers)],
            "reference_tier": tiers,
            "targets": targets,
            "node_mask": node_mask,
            "alignment_targets": alignment_targets,
        }

    hashes = {
        "assignment": "a" * 64,
        "selection_manifest": "b" * 64,
        "ai_reference_manifest": "c" * 64,
        "source_split": "d" * 64,
    }
    split_tier_counts = {
        "train": {"human_gold": 1, "ai_silver": 2},
        "validation": {"human_gold": 1, "ai_silver": 1},
        "test": {"human_gold": 0, "ai_silver": 0},
    }

    class FakeDataset:
        def __init__(self, batches: list[dict]):
            self.batches = batches
            self.reference_tiers = [
                tier
                for batch in batches
                for tier in batch["reference_tier"]
            ]
            self.tier_counts = {
                tier: self.reference_tiers.count(tier)
                for tier in sorted(set(self.reference_tiers))
            }
            self.manifest_hashes = hashes
            self.all_split_tier_counts = split_tier_counts

        def __len__(self):
            return len(self.reference_tiers)

    train_set = FakeDataset(
        [make_batch(["human_gold", "ai_silver"]), make_batch(["ai_silver"])]
    )
    validation_set = FakeDataset(
        [make_batch(["human_gold", "ai_silver"])]
    )

    class FakeLoader:
        def __init__(self, dataset, **_):
            self.batches = dataset.batches

        def __iter__(self):
            return iter(self.batches)

        def __len__(self):
            return len(self.batches)

    class FakeModel(torch.nn.Module):
        def __init__(self, **_):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, batch):
            batch_size, node_count = batch["node_mask"].shape

            def logits(*shape: int) -> torch.Tensor:
                return self.scale * torch.ones(shape, device=self.scale.device)

            return {
                "action_logits": logits(batch_size, node_count, 4),
                "element_type_logits": logits(batch_size, node_count, 6),
                "group_affinity_logits": logits(
                    batch_size, node_count, node_count
                ),
                "role_logits": logits(batch_size, node_count, 10),
                "layout_mode_logits": logits(batch_size, node_count, 4),
                "primary_align_logits": logits(batch_size, node_count, 4),
                "cross_align_logits": logits(batch_size, node_count, 4),
                "gap": logits(batch_size, node_count),
                "padding": logits(batch_size, node_count, 4),
                "token_affinity_logits": logits(
                    batch_size, 6, node_count, node_count
                ),
                "parent_logits": logits(
                    batch_size, node_count, node_count + 1
                ),
                "alignment_logits": logits(batch_size, node_count, 3),
            }

    init_checkpoint = tmp_path / "init.pt"
    torch.save({"model": FakeModel().state_dict()}, init_checkpoint)
    output_dir = tmp_path / "run"
    config_path = tmp_path / "reference.yaml"
    config = {
        "experiment": {
            "name": "reference-smoke-v2",
            "output_dir": str(output_dir),
            "seed": 42,
        },
        "data": {"dataset_kind": "reference"},
        "model": {},
        "training": {
            "init_checkpoint": "must-be-overridden.pt",
            "epochs": 1,
            "batch_size": 2,
            "grad_accum": 2,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "reference_tier_weights": {
                "human_gold": 1.0,
                "ai_silver": 0.5,
            },
            "model_selection_tier": "ai_silver",
            "loss_weights": _only_loss("action"),
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr(
        train_intent,
        "build_datasets",
        lambda _: (train_set, validation_set),
    )
    monkeypatch.setattr(train_intent, "DataLoader", FakeLoader)
    monkeypatch.setattr(train_intent, "DesignIntentRecoveryModel", FakeModel)
    monkeypatch.setattr(train_intent, "_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        train_intent.sys,
        "argv",
        [
            "train_intent.py",
            "--config",
            str(config_path),
            "--override",
            f"training.init_checkpoint={init_checkpoint}",
        ],
    )

    train_intent.main()

    best_path = output_dir / "best.pt"
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    record = json.loads(
        (output_dir / "train.jsonl").read_text(encoding="utf-8").strip()
    )
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert checkpoint["epoch"] == 1
    assert checkpoint["audit"]["training_finished_at"] is not None
    assert checkpoint["audit"]["effective_config"]["training"][
        "init_checkpoint"
    ] == str(init_checkpoint)
    assert checkpoint["audit"]["split_tier_counts"] == split_tier_counts
    assert record["selection"]["tier"] == "ai_silver"
    assert record["selection"]["tier_weighted"] is False
    assert manifest["checkpoints"]["best"]["sha256"] == hashlib.sha256(
        best_path.read_bytes()
    ).hexdigest()
