"""
GPU 集群兼容训练脚本（accelerate-ready，DDP/AMP 自动）。

本脚本是 scripts/train.py 的 GPU 镜像版：
- 用 `accelerate.Accelerator()` 包装 model/optimizer/dataloader
- 自动分发到多 GPU、自动 bf16 autocast
- 数据/超参完全复用同一份 yaml

启动：
    accelerate config           # 第一次需要 setup
    accelerate launch scripts/train_accelerate.py --config configs/gpu.yaml

也可单卡 fallback：
    python scripts/train_accelerate.py --config configs/default.yaml
"""
import os
import sys
import json
import math
import time
import random
import argparse

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from accelerate.utils import set_seed as accel_set_seed

from src.data.dataset import WebpageDataset, webpage_collate_fn
from src.training.trainer import (
    FigmaGenerationModel, compute_losses, build_type_targets,
)


def _deep_merge(b, o):
    out = dict(b)
    for k, v in o.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict:
    cfg: dict = {}
    if path != "configs/default.yaml" and os.path.exists("configs/default.yaml"):
        with open("configs/default.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    with open(path, encoding="utf-8") as f:
        local = yaml.safe_load(f) or {}
    return _deep_merge(cfg, local)


def cosine_warmup_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


def make_param_groups(model, lr_h, lr_c, wd):
    h, c = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (c if n.startswith("code_encoder.") else h).append(p)
    return [
        {"params": h, "lr": lr_h, "weight_decay": wd},
        {"params": c, "lr": lr_c, "weight_decay": wd},
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["training"].get("grad_accum", 1),
        mixed_precision="bf16" if torch.cuda.is_available() else "no",
    )
    accel_set_seed(cfg["experiment"].get("seed", 42))

    if accelerator.is_main_process:
        print(f"[accelerate] device={accelerator.device}  num_processes={accelerator.num_processes}")
        print(f"[mixed_precision] {accelerator.mixed_precision}")

    # 数据
    full = WebpageDataset(
        data_dir=cfg["data"]["data_dir"],
        max_nodes=cfg["data"]["max_nodes"],
        iou_threshold=cfg["data"]["iou_threshold"],
    )
    n = len(full)
    n_train = max(1, int(n * cfg["data"]["train_ratio"]))
    train_set = Subset(full, list(range(n_train)))
    val_set = Subset(full, list(range(n_train, n))) if n > n_train else train_set

    train_loader = DataLoader(
        train_set, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4) if torch.cuda.is_available() else 0,
        collate_fn=webpage_collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=0, collate_fn=webpage_collate_fn,
    )

    # 模型
    mc = cfg["model"]
    model = FigmaGenerationModel(
        dim=mc["dim"], align_heads=mc["align_heads"],
        ctx_layers=mc["ctx_layers"], ctx_heads=mc["ctx_heads"],
        struct_layers=mc["struct_layers"], struct_heads=mc["struct_heads"],
        freeze_visual=mc["freeze_visual"],
        codebert_unfreeze_last=mc["codebert_unfreeze_last"],
    )

    optimizer = torch.optim.AdamW(make_param_groups(
        model, cfg["training"]["lr_heads"], cfg["training"]["lr_codebert"],
        cfg["training"]["weight_decay"],
    ))

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    out_dir = cfg["experiment"]["output_dir"]
    if accelerator.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
        log_f = open(os.path.join(out_dir, "train.log.jsonl"), "a", encoding="utf-8")

    epochs = cfg["training"]["epochs"]
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * cfg["training"].get("warmup_ratio", 0.05))
    global_step = 0
    best_val = float("inf")
    loss_w = cfg["training"]["loss_weights"]

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = {}
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                target_types = build_type_targets(
                    batch["node_texts"], batch["node_mask"].shape[1]
                )
                outputs = model(
                    batch, return_styles=True, teacher_force=True,
                    target_types=target_types,
                )
                losses = compute_losses(
                    outputs,
                    target_types=target_types.to(accelerator.device),
                    parents=batch["parents"].to(accelerator.device),
                    node_mask=batch["node_mask"].to(accelerator.device),
                    style_targets=batch["styles"],
                    weights=loss_w,
                    enable_styles=epoch >= cfg["training"].get("enable_styles_after_epoch", 0),
                )
                accelerator.backward(losses["total"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        cfg["training"].get("grad_clip", 1.0),
                    )
                # cosine warmup
                lr_scale = cosine_warmup_lr(global_step, total_steps, warmup_steps)
                for i, g in enumerate(optimizer.param_groups):
                    base = cfg["training"]["lr_heads"] if i == 0 else cfg["training"]["lr_codebert"]
                    g["lr"] = base * lr_scale
                optimizer.step()
                optimizer.zero_grad()

            for k, v in losses.items():
                epoch_loss[k] = epoch_loss.get(k, 0.0) + float(v.item())
            global_step += 1

        if accelerator.is_main_process:
            avg = {k: v / max(1, steps_per_epoch) for k, v in epoch_loss.items()}
            log = {"epoch": epoch, "phase": "train", **avg}
            log_f.write(json.dumps(log) + "\n"); log_f.flush()
            print(f"[train e{epoch}] " + "  ".join(f"{k}={v:.3f}" for k, v in avg.items()))

        # eval
        model.eval()
        val_loss = {}
        with torch.no_grad():
            for batch in val_loader:
                target_types = build_type_targets(batch["node_texts"], batch["node_mask"].shape[1])
                outputs = model(batch, return_styles=True, teacher_force=True, target_types=target_types)
                losses = compute_losses(
                    outputs,
                    target_types=target_types.to(accelerator.device),
                    parents=batch["parents"].to(accelerator.device),
                    node_mask=batch["node_mask"].to(accelerator.device),
                    style_targets=batch["styles"],
                    weights=loss_w,
                )
                for k, v in losses.items():
                    val_loss[k] = val_loss.get(k, 0.0) + float(v.item())
        if accelerator.is_main_process:
            n_val = max(1, len(val_loader))
            ev = {k: v / n_val for k, v in val_loss.items()}
            log = {"epoch": epoch, "phase": "val", **ev}
            log_f.write(json.dumps(log) + "\n"); log_f.flush()
            print(f"[val   e{epoch}] " + "  ".join(f"{k}={v:.3f}" for k, v in ev.items()))
            if ev.get("total", 1e9) < best_val:
                best_val = ev["total"]
                accelerator.wait_for_everyone()
                unwrapped = accelerator.unwrap_model(model)
                accelerator.save(unwrapped.state_dict(), os.path.join(out_dir, "best.pt"))

    if accelerator.is_main_process:
        log_f.close()
        print("done")


if __name__ == "__main__":
    main()
