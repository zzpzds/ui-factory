"""在弱标注或来源分层 Reference Intent IR 上评测模型。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.data.intent_dataset import (
    IntentDataset,
    ReferenceIntentDataset,
    intent_collate_fn,
    validate_test_unseal,
)
from src.design_intent.evaluation import (
    build_evaluation_report,
    make_failed_sample_result,
    make_valid_sample_result,
)
from src.design_intent.metrics import compute_intent_metrics
from src.design_intent.solver import decode_intent_ir
from src.design_intent.validation import validate_intent_ir
from src.models.intent import DesignIntentRecoveryModel


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_REFERENCE_PACKAGE = (
    REPO_ROOT / "data/annotations/intent_gold_v1"
).resolve()
FORMAL_TEST_UNSEAL_PATH = (
    REPO_ROOT / "outputs/intent-reference-v1/test_unseal.json"
)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tag(path: str | Path) -> str:
    return f"sha256:{_sha256(path)}"


def _load_json_object(path: str | Path, *, field: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{field} 必须是 JSON object：{path}")
    return payload


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def create_or_read_test_unseal(
    output_dir: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """排他创建首次 test 解封记录；已存在时只读复用。"""
    validated_payload = validate_test_unseal(dict(payload))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "test_unseal.json"
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(
                validated_payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
        return validated_payload
    except FileExistsError:
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        return validate_test_unseal(existing)


def _is_formal_reference_test(package_dir: str | Path, split: str) -> bool:
    return split == "test" and Path(package_dir).resolve() == FORMAL_REFERENCE_PACKAGE


def evaluate_reference_batches(
    model: Any,
    loader: Iterable[dict[str, Any]],
    *,
    decode_fn: Callable[..., Any] | None = None,
    validate_fn: Callable[..., list[str]] | None = None,
    metric_fn: Callable[..., Mapping[str, float | None]] | None = None,
) -> list[dict[str, Any]]:
    """执行 Reference 推理，只降级协议允许的两类逐页失败。"""
    decoder = decode_fn or decode_intent_ir
    validator = validate_fn or validate_intent_ir
    metric_computer = metric_fn or compute_intent_metrics
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            sample_ids = batch.get("sample_id", [])
            tiers = batch.get("reference_tier", [])
            if len(sample_ids) != 1 or len(tiers) != 1:
                raise ValueError("Reference 评测 loader 必须使用 batch_size=1。")
            sample_id = str(sample_ids[0])
            tier = str(tiers[0])
            graph = batch["graph"][0]
            gold = batch["intent_ir"][0]
            outputs = model(batch)
            try:
                predicted = decoder(graph, outputs)
            except Exception as error:
                records.append(
                    make_failed_sample_result(
                        sample_id,
                        tier,
                        stage="decode",
                        error_type=type(error).__name__,
                        error_summary=str(error),
                    )
                )
                continue

            errors = validator(predicted, graph)
            if errors:
                records.append(
                    make_failed_sample_result(
                        sample_id,
                        tier,
                        stage="ir_validation",
                        error_type="IRValidationError",
                        error_summary="; ".join(str(error) for error in errors),
                    )
                )
                continue

            metrics = metric_computer(
                predicted,
                gold,
                graph,
                include_token_metrics=tier == "ai_silver",
            )
            records.append(make_valid_sample_result(sample_id, tier, metrics))
    return records


def _load_model(
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> DesignIntentRecoveryModel:
    model = DesignIntentRecoveryModel(**checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _evaluate_legacy(
    checkpoint: Mapping[str, Any],
    *,
    annotation_name: str | None,
    split: str,
) -> dict[str, Any]:
    data_config = checkpoint["config"]["data"]
    resolved_annotation = annotation_name or data_config["annotation_name"]
    split_path = data_config.get("split_manifest")
    if not split_path:
        raise ValueError("checkpoint 配置缺少 data.split_manifest。")
    with open(split_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if split not in manifest["splits"]:
        raise ValueError(f"未知 split：{split}")
    dataset = IntentDataset(
        data_dir=data_config["data_dir"],
        max_nodes=int(data_config["max_nodes"]),
        annotation_name=resolved_annotation,
        sample_ids=manifest["splits"][split],
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=intent_collate_fn
    )
    model = _load_model(checkpoint, _device())

    values: dict[str, list[float]] = defaultdict(list)
    per_sample: dict[str, dict[str, float | None]] = {}
    failures: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch)
            predicted = decode_intent_ir(batch["graph"][0], outputs)
            errors = validate_intent_ir(predicted, batch["graph"][0])
            if errors:
                failures.append(
                    {"sample_id": batch["sample_id"][0], "errors": errors}
                )
                continue
            metrics = compute_intent_metrics(
                predicted, batch["intent_ir"][0], batch["graph"][0]
            )
            per_sample[batch["sample_id"][0]] = metrics
            for name, value in metrics.items():
                if value is not None:
                    values[name].append(value)

    return {
        "checkpoint": "loaded_checkpoint",
        "annotation_name": resolved_annotation,
        "split": split,
        "samples": len(dataset),
        "valid_predictions": len(dataset) - len(failures),
        "metrics": {
            name: sum(items) / len(items) for name, items in values.items()
        },
        "per_sample": per_sample,
        "failures": failures,
    }


def _evaluate_reference(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    package_dir: str | Path,
    split: str,
    statistics_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples <= 0
    ):
        raise ValueError("bootstrap_samples 必须是正整数。")
    if isinstance(statistics_seed, bool) or not isinstance(statistics_seed, int):
        raise ValueError("statistics_seed 必须是整数。")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"未知 split：{split}")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint 必须是 mapping。")

    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint 缺少完整 config。")
    data_config = config.get("data")
    if not isinstance(data_config, Mapping):
        raise ValueError("checkpoint 配置缺少 data。")
    if not isinstance(config.get("model"), Mapping):
        raise ValueError("checkpoint 配置缺少 model。")
    if not isinstance(checkpoint.get("model"), Mapping):
        raise ValueError("checkpoint 缺少 model state_dict。")

    raw_data_dir = data_config.get("data_dir")
    if not isinstance(raw_data_dir, (str, os.PathLike)) or not str(raw_data_dir):
        raise ValueError("Reference 配置缺少 data.data_dir。")
    data_dir = Path(raw_data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)

    max_nodes = data_config.get("max_nodes")
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes <= 0:
        raise ValueError("data.max_nodes 必须是正整数。")
    verify_hashes = data_config.get("verify_hashes", True)
    if not isinstance(verify_hashes, bool):
        raise ValueError("data.verify_hashes 必须是布尔值。")

    source_split = data_config.get("source_split_manifest") or data_config.get(
        "split_manifest"
    )
    if not isinstance(source_split, (str, os.PathLike)) or not str(source_split):
        raise ValueError(
            "Reference 评测缺少 data.source_split_manifest 或 "
            "data.split_manifest。"
        )
    source_split_path = Path(source_split).resolve()

    package_path = Path(package_dir).resolve()
    if not package_path.is_dir():
        raise FileNotFoundError(package_path)
    is_formal_test = _is_formal_reference_test(package_path, split)
    if is_formal_test and not verify_hashes:
        raise ValueError("正式 Reference test 必须启用 data.verify_hashes。")
    manifest_paths = {
        "assignment": package_path / "assignment.json",
        "selection_manifest": package_path / "selection_manifest.json",
        "ai_reference_manifest": package_path / "ai_reference_manifest.json",
        "source_split": source_split_path,
    }
    for name, path in manifest_paths.items():
        _load_json_object(path, field=name)

    checkpoint_hash = _sha256_tag(checkpoint_path)
    model = _load_model(checkpoint, _device())
    dataset_kwargs = {
        "data_dir": str(data_dir),
        "package_dir": str(package_dir),
        "split": split,
        "max_nodes": max_nodes,
        "tiers": ["human_gold", "ai_silver"],
        "source_split_manifest": str(source_split_path),
        "verify_hashes": verify_hashes,
    }
    metadata_dataset = ReferenceIntentDataset(
        **dataset_kwargs,
        _metadata_only=True,
    )
    preflight_manifest_hashes = dict(metadata_dataset.manifest_hashes)

    unseal: dict[str, Any] | None = None
    if is_formal_test:
        unseal = create_or_read_test_unseal(
            FORMAL_TEST_UNSEAL_PATH.parent,
            {
                "unsealed_at": datetime.now().astimezone().isoformat(),
                "git_commit": _git_commit(),
                "assignment_hash": (
                    f"sha256:{preflight_manifest_hashes['assignment']}"
                ),
                "checkpoint_hash": checkpoint_hash,
            },
        )

    dataset = ReferenceIntentDataset(**dataset_kwargs)
    if dict(dataset.manifest_hashes) != preflight_manifest_hashes:
        raise RuntimeError("Reference manifest 在元数据预检后发生变化。")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=intent_collate_fn
    )
    manifest_hashes = {
        name: f"sha256:{value}"
        for name, value in dataset.manifest_hashes.items()
    }
    run: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "reference_package": str(package_dir),
        "statistics_seed": statistics_seed,
        "bootstrap_samples": bootstrap_samples,
        "hashes": {
            **manifest_hashes,
            "checkpoint": checkpoint_hash,
        },
    }
    if "source_split" not in run["hashes"]:
        run["hashes"]["source_split"] = _sha256_tag(source_split_path)

    if unseal is not None:
        run["test_unseal"] = {
            "path": str(FORMAL_TEST_UNSEAL_PATH),
            **unseal,
        }

    records = evaluate_reference_batches(model, loader)
    expected_ids = sorted(sample.sample_id for sample in dataset.samples)
    return build_evaluation_report(
        records,
        run=run,
        expected_ids=expected_ids,
        statistics_seed=statistics_seed,
        bootstrap_samples=bootstrap_samples,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotation_name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--reference-package", "--reference_package", dest="reference_package"
    )
    parser.add_argument(
        "--statistics-seed", "--statistics_seed", type=int, default=42
    )
    parser.add_argument(
        "--bootstrap-samples", "--bootstrap_samples", type=int, default=10_000
    )
    parser.add_argument("--output", default="outputs/intent-evaluation.json")
    args = parser.parse_args(argv)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    data_config = checkpoint["config"]["data"]
    if args.split is None:
        raise ValueError(
            "评测必须显式提供 --split train、validation 或 test。"
        )
    package_dir = args.reference_package or data_config.get("package_dir")
    if package_dir is not None or data_config.get("dataset_kind") == "reference":
        if package_dir is None:
            raise ValueError("Reference 评测缺少 package_dir。")
        result = _evaluate_reference(
            checkpoint,
            checkpoint_path=args.checkpoint,
            package_dir=package_dir,
            split=args.split,
            statistics_seed=args.statistics_seed,
            bootstrap_samples=args.bootstrap_samples,
        )
    else:
        result = _evaluate_legacy(
            checkpoint,
            annotation_name=args.annotation_name,
            split=args.split,
        )
        result["checkpoint"] = args.checkpoint

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
