"""双流设计意图恢复模型训练入口。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Iterable

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import yaml

from src.data.intent_dataset import (
    IntentDataset,
    ReferenceIntentDataset,
    intent_collate_fn,
)
from src.models.intent import DesignIntentRecoveryModel
from src.training.intent_losses import compute_intent_losses


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed(value: int) -> None:
    random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _apply_overrides(config: dict, overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"无效 override：{override}")
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value


def _load_split_manifest(path: str) -> dict[str, list[str]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"切分清单不存在：{manifest_path}。"
            "请先运行 scripts/split_intent_dataset.py。"
        )
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["splits"]


def normalize_reference_tier_weights(
    reference_tiers: Iterable[str],
    raw_weights: dict[str, float],
) -> dict[str, float]:
    """在完整 Reference train 子集上归一化一次来源权重。"""
    tiers = list(reference_tiers)
    if not tiers:
        raise ValueError("Reference train 不能为空。")
    active_tiers = sorted(set(tiers))
    missing = [tier for tier in active_tiers if tier not in raw_weights]
    if missing:
        raise ValueError(f"缺少激活 tier 权重：{missing}")
    checked: dict[str, float] = {}
    for tier in active_tiers:
        value = float(raw_weights[tier])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"tier 权重必须有限且严格大于零：{tier}={value}")
        checked[tier] = value
    mean_weight = sum(checked[tier] for tier in tiers) / len(tiers)
    return {tier: checked[tier] / mean_weight for tier in active_tiers}


def backward_page_mean(loss: torch.Tensor, page_count: int) -> None:
    if page_count <= 0:
        raise ValueError("micro-batch 页面数必须大于零。")
    (loss * page_count).backward()


def normalize_accumulated_gradients(
    parameters: Iterable[torch.Tensor],
    page_count: int,
) -> None:
    if page_count <= 0:
        raise ValueError("梯度窗口页面数必须大于零。")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(page_count)


def require_init_checkpoint(path: str | None) -> Path | None:
    if not path:
        return None
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"init checkpoint 不存在：{checkpoint_path}")
    return checkpoint_path


def validation_selection_loss(
    validation: dict[str, Any],
    model_selection_tier: str | None = None,
) -> float:
    if model_selection_tier is None:
        value = validation.get("total")
    else:
        value = (
            validation.get("by_tier", {})
            .get(model_selection_tier, {})
            .get("total")
        )
    if value is None or not math.isfinite(float(value)):
        raise ValueError(
            f"validation 缺少可用 selection loss：{model_selection_tier or 'overall'}"
        )
    return float(value)


def validate_reference_selection_tier(
    validation_tiers: Iterable[str],
    model_selection_tier: str | None,
) -> None:
    if model_selection_tier != "ai_silver":
        raise ValueError(
            "Reference checkpoint 选择层必须是 ai_silver。"
        )
    tiers = set(validation_tiers)
    if tiers != {"human_gold", "ai_silver"}:
        raise ValueError(
            "Reference validation 必须固定包含 human_gold 和 ai_silver。"
        )


def build_datasets(data_config: dict[str, Any]):
    dataset_kind = data_config.get("dataset_kind", "page_directory")
    if dataset_kind == "page_directory":
        split_path = data_config.get("split_manifest")
        if not split_path:
            raise ValueError("正式训练必须配置 data.split_manifest，禁止随机页面切分。")
        splits = _load_split_manifest(split_path)
        dataset_args = {
            "data_dir": data_config["data_dir"],
            "max_nodes": int(data_config["max_nodes"]),
            "annotation_name": data_config["annotation_name"],
        }
        return (
            IntentDataset(**dataset_args, sample_ids=splits["train"]),
            IntentDataset(**dataset_args, sample_ids=splits["validation"]),
        )
    if dataset_kind != "reference":
        raise ValueError(f"未知 data.dataset_kind：{dataset_kind}")
    common = {
        "data_dir": data_config["data_dir"],
        "package_dir": data_config["package_dir"],
        "max_nodes": int(data_config["max_nodes"]),
        "source_split_manifest": data_config["source_split_manifest"],
        "verify_hashes": bool(data_config.get("verify_hashes", True)),
    }
    train_set = ReferenceIntentDataset(
        **common,
        split="train",
        tiers=list(data_config["train_tiers"]),
    )
    validation_set = ReferenceIntentDataset(
        **common,
        split="validation",
        tiers=list(data_config["validation_tiers"]),
    )
    return train_set, validation_set


def _index_batch_tensors(
    values: dict[str, torch.Tensor],
    indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        key: value.index_select(0, indices.to(value.device))
        for key, value in values.items()
    }


def _index_tensor_rows(
    value: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return value.index_select(0, indices.to(value.device))


def _add_loss_totals(
    totals: dict[str, float],
    losses: dict[str, torch.Tensor],
    page_count: int,
) -> None:
    for name, value in losses.items():
        totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * page_count


def _mean_loss_totals(
    totals: dict[str, float],
    page_count: int,
) -> dict[str, float]:
    if page_count <= 0:
        raise ValueError("损失汇总页面数必须大于零。")
    return {name: value / page_count for name, value in totals.items()}


def _manifest_hashes(data_config: dict[str, Any]) -> dict[str, str]:
    if data_config.get("dataset_kind", "page_directory") == "reference":
        package_dir = Path(data_config["package_dir"])
        paths = {
            "assignment": package_dir / "assignment.json",
            "selection_manifest": package_dir / "selection_manifest.json",
            "ai_reference_manifest": package_dir / "ai_reference_manifest.json",
            "source_split": Path(data_config["source_split_manifest"]),
        }
    else:
        paths = {"split_manifest": Path(data_config["split_manifest"])}
    return {name: _sha256(path) for name, path in paths.items()}


def build_training_audit(
    config: dict[str, Any],
    train_set,
    validation_set,
    device: torch.device,
    raw_tier_weights: dict[str, float],
    normalized_tier_weights: dict[str, float],
    started_at: str,
) -> dict[str, Any]:
    data_config = config["data"]
    is_reference = data_config.get("dataset_kind") == "reference"
    active_split_tier_counts = {
        "train": dict(getattr(train_set, "tier_counts", {})),
        "validation": dict(getattr(validation_set, "tier_counts", {})),
    }
    split_tier_counts = deepcopy(active_split_tier_counts)
    if not is_reference:
        split_tier_counts = {
            "train": {"page_directory": len(train_set)},
            "validation": {"page_directory": len(validation_set)},
        }
        manifest_hashes = _manifest_hashes(data_config)
    else:
        manifest_hashes = dict(train_set.manifest_hashes)
        if manifest_hashes != dict(validation_set.manifest_hashes):
            raise ValueError("Reference train/validation 的 manifest 快照不一致。")
        split_tier_counts = {
            split: dict(counts)
            for split, counts in train_set.all_split_tier_counts.items()
        }
        validation_split_tier_counts = {
            split: dict(counts)
            for split, counts in validation_set.all_split_tier_counts.items()
        }
        if split_tier_counts != validation_split_tier_counts:
            raise ValueError(
                "Reference train/validation 的 tier count 快照不一致。"
            )
    return {
        "git_commit": _git_commit(),
        "effective_config": deepcopy(config),
        "data_manifest_sha256": manifest_hashes,
        "split_tier_counts": split_tier_counts,
        "active_split_tier_counts": active_split_tier_counts,
        "raw_reference_tier_weights": dict(raw_tier_weights),
        "normalized_reference_tier_weights": dict(normalized_tier_weights),
        "seed": int(config["experiment"]["seed"]),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "device": str(device),
        },
        "training_started_at": started_at,
        "training_finished_at": None,
    }


def write_run_manifest(
    output_dir: str | Path,
    checkpoint_paths: dict[str, str | Path],
    metadata: dict[str, Any] | None = None,
) -> None:
    manifest = dict(metadata or {})
    manifest["checkpoints"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in sorted(checkpoint_paths.items())
        if Path(path).is_file()
    }
    path = Path(output_dir) / "run_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@torch.no_grad()
def evaluate(
    model: DesignIntentRecoveryModel,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    stratify_by_tier: bool = False,
) -> dict[str, Any]:
    model.eval()
    overall_totals: dict[str, float] = {}
    overall_pages = 0
    tier_totals: dict[str, dict[str, float]] = {}
    tier_pages: dict[str, int] = {}
    for batch in loader:
        outputs = model(batch)
        losses = compute_intent_losses(
            outputs,
            batch["targets"],
            batch["node_mask"],
            alignment_targets=batch["alignment_targets"],
            weights=weights,
            annotation_page_weights=None,
        )
        batch_pages = len(batch["sample_id"])
        _add_loss_totals(overall_totals, losses, batch_pages)
        overall_pages += batch_pages
        if not stratify_by_tier:
            continue
        tiers = list(batch["reference_tier"])
        for tier in sorted(set(tiers)):
            indices = torch.tensor(
                [index for index, value in enumerate(tiers) if value == tier],
                dtype=torch.long,
            )
            tier_losses = compute_intent_losses(
                _index_batch_tensors(outputs, indices),
                _index_batch_tensors(batch["targets"], indices),
                _index_tensor_rows(batch["node_mask"], indices),
                alignment_targets=_index_tensor_rows(
                    batch["alignment_targets"], indices
                ),
                weights=weights,
                annotation_page_weights=None,
            )
            count = len(indices)
            totals = tier_totals.setdefault(tier, {})
            _add_loss_totals(totals, tier_losses, count)
            tier_pages[tier] = tier_pages.get(tier, 0) + count
    overall = _mean_loss_totals(overall_totals, overall_pages)
    if not stratify_by_tier:
        return overall
    return {
        "overall": overall,
        "by_tier": {
            tier: _mean_loss_totals(tier_totals[tier], tier_pages[tier])
            for tier in sorted(tier_totals)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/intent/full.yaml")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    _apply_overrides(config, args.override)

    experiment = config["experiment"]
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["training"]
    seed = int(experiment["seed"])
    _seed(seed)
    started_at = _now()
    init_checkpoint = require_init_checkpoint(train_config.get("init_checkpoint"))

    train_set, val_set = build_datasets(data_config)
    if not train_set or not val_set:
        raise RuntimeError("训练集和验证集都必须非空。")
    is_reference = data_config.get("dataset_kind") == "reference"
    model_selection_tier = train_config.get("model_selection_tier")
    if is_reference:
        validate_reference_selection_tier(
            val_set.reference_tiers,
            model_selection_tier,
        )
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        collate_fn=intent_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        collate_fn=intent_collate_fn,
    )

    device = _device()
    model = DesignIntentRecoveryModel(**model_config).to(device)
    if init_checkpoint:
        initial = torch.load(
            init_checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(initial["model"])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    grad_accum = int(train_config.get("grad_accum", 1))
    if grad_accum <= 0:
        raise ValueError("training.grad_accum 必须大于零。")
    weights = train_config.get("loss_weights", {})
    raw_tier_weights = dict(train_config.get("reference_tier_weights", {}))
    normalized_tier_weights = (
        normalize_reference_tier_weights(
            train_set.reference_tiers,
            raw_tier_weights,
        )
        if is_reference
        else {}
    )
    output_dir = Path(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.jsonl"
    audit = build_training_audit(
        config,
        train_set,
        val_set,
        device,
        raw_tier_weights,
        normalized_tier_weights,
        started_at,
    )

    best_val = float("inf")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        train_totals: dict[str, float] = {}
        train_pages = 0
        accumulated_pages = 0
        accumulated_micro_batches = 0
        for step, batch in enumerate(train_loader, start=1):
            outputs = model(batch)
            annotation_page_weights = None
            if is_reference:
                annotation_page_weights = torch.tensor(
                    [
                        normalized_tier_weights[tier]
                        for tier in batch["reference_tier"]
                    ],
                    dtype=torch.float32,
                )
            losses = compute_intent_losses(
                outputs,
                batch["targets"],
                batch["node_mask"],
                alignment_targets=batch["alignment_targets"],
                weights=weights,
                annotation_page_weights=annotation_page_weights,
            )
            batch_pages = len(batch["sample_id"])
            backward_page_mean(losses["total"], batch_pages)
            accumulated_pages += batch_pages
            accumulated_micro_batches += 1
            if (
                accumulated_micro_batches == grad_accum
                or step == len(train_loader)
            ):
                normalize_accumulated_gradients(
                    model.parameters(), accumulated_pages
                )
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_config["grad_clip"])
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_pages = 0
                accumulated_micro_batches = 0
            _add_loss_totals(train_totals, losses, batch_pages)
            train_pages += batch_pages

        validation = evaluate(
            model,
            val_loader,
            device,
            weights,
            stratify_by_tier=is_reference,
        )
        selection_loss = validation_selection_loss(
            validation,
            model_selection_tier=model_selection_tier if is_reference else None,
        )
        record = {
            "epoch": epoch,
            "train": _mean_loss_totals(train_totals, train_pages),
            "validation": validation,
            "selection": {
                "tier": model_selection_tier if is_reference else "overall",
                "loss": selection_loss,
                "tier_weighted": False,
            },
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))

        checkpoint = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "validation": validation,
            "audit": {
                **audit,
                "checkpoint_created_at": _now(),
            },
        }
        last_path = output_dir / "last.pt"
        best_path = output_dir / "best.pt"
        torch.save(checkpoint, last_path)
        if selection_loss < best_val:
            best_val = selection_loss
            torch.save(checkpoint, best_path)
        write_run_manifest(
            output_dir,
            {"last": last_path, "best": best_path},
            {"training_started_at": started_at, "training_finished_at": None},
        )

    finished_at = _now()
    checkpoint_paths = {
        "last": output_dir / "last.pt",
        "best": output_dir / "best.pt",
    }
    for checkpoint_path in checkpoint_paths.values():
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        payload["audit"]["training_finished_at"] = finished_at
        torch.save(payload, checkpoint_path)
    write_run_manifest(
        output_dir,
        checkpoint_paths,
        {"training_started_at": started_at, "training_finished_at": finished_at},
    )


if __name__ == "__main__":
    main()
