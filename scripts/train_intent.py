"""双流设计意图恢复模型训练入口。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import yaml

from src.data.intent_dataset import IntentDataset, intent_collate_fn
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


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


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


@torch.no_grad()
def evaluate(
    model: DesignIntentRecoveryModel,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = {}
    for batch in loader:
        outputs = model(batch)
        losses = compute_intent_losses(
            outputs,
            batch["targets"],
            batch["node_mask"],
            alignment_targets=batch["alignment_targets"],
            weights=weights,
        )
        for name, value in losses.items():
            totals.setdefault(name, []).append(float(value.detach().cpu()))
    return {name: _mean(values) for name, values in totals.items()}


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

    split_path = data_config.get("split_manifest")
    if not split_path:
        raise ValueError("正式训练必须配置 data.split_manifest，禁止随机页面切分。")
    splits = _load_split_manifest(split_path)
    dataset_args = {
        "data_dir": data_config["data_dir"],
        "max_nodes": int(data_config["max_nodes"]),
        "annotation_name": data_config["annotation_name"],
    }
    train_set = IntentDataset(**dataset_args, sample_ids=splits["train"])
    val_set = IntentDataset(**dataset_args, sample_ids=splits["validation"])
    if not train_set or not val_set:
        raise RuntimeError("训练集和验证集都必须非空。")
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
    init_checkpoint = train_config.get("init_checkpoint")
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
    weights = train_config.get("loss_weights", {})
    output_dir = Path(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.jsonl"

    best_val = float("inf")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        train_totals: dict[str, list[float]] = {}
        for step, batch in enumerate(train_loader, start=1):
            outputs = model(batch)
            losses = compute_intent_losses(
                outputs,
                batch["targets"],
                batch["node_mask"],
                alignment_targets=batch["alignment_targets"],
                weights=weights,
            )
            (losses["total"] / grad_accum).backward()
            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_config["grad_clip"])
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for name, value in losses.items():
                train_totals.setdefault(name, []).append(float(value.detach().cpu()))

        validation = evaluate(model, val_loader, device, weights)
        record = {
            "epoch": epoch,
            "train": {
                name: _mean(values) for name, values in train_totals.items()
            },
            "validation": validation,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))

        checkpoint = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "validation": validation,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if validation.get("total", float("inf")) < best_val:
            best_val = validation["total"]
            torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
