"""在人工或弱标注 Intent IR 上评测完整模型与约束求解器。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.data.intent_dataset import IntentDataset, intent_collate_fn
from src.design_intent.metrics import compute_intent_metrics
from src.design_intent.solver import decode_intent_ir
from src.design_intent.validation import validate_intent_ir
from src.models.intent import DesignIntentRecoveryModel


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotation_name", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/intent-evaluation.json")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    data_config = config["data"]
    model_config = config["model"]
    annotation_name = args.annotation_name or data_config["annotation_name"]
    split_path = data_config.get("split_manifest")
    if not split_path:
        raise ValueError("checkpoint 配置缺少 data.split_manifest。")
    with open(split_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if args.split not in manifest["splits"]:
        raise ValueError(f"未知 split：{args.split}")
    dataset = IntentDataset(
        data_dir=data_config["data_dir"],
        max_nodes=int(data_config["max_nodes"]),
        annotation_name=annotation_name,
        sample_ids=manifest["splits"][args.split],
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=intent_collate_fn
    )

    device = _device()
    model = DesignIntentRecoveryModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    values: dict[str, list[float]] = defaultdict(list)
    per_sample: dict[str, dict[str, float]] = {}
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
                values[name].append(value)

    result = {
        "checkpoint": args.checkpoint,
        "annotation_name": annotation_name,
        "split": args.split,
        "samples": len(dataset),
        "valid_predictions": len(dataset) - len(failures),
        "metrics": {
            name: sum(items) / len(items) for name, items in values.items()
        },
        "per_sample": per_sample,
        "failures": failures,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
