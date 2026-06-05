"""
端到端训练脚本（论文 spec 完整版）。

特性：
- device-agnostic（CUDA / MPS / CPU 自动）
- grad_accum 等效 batch
- cosine + warmup
- 资源保护：内存监控、定期 sleep、低频 cache 清理
- accelerate-ready：标记好 device 与 autocast，迁 GPU 时只需小改
- 单一训练入口（trainer.py 里没有 train()）

用法：
    python scripts/train.py                                # 用 configs/default.yaml
    python scripts/train.py --config configs/M1.yaml
    python scripts/train.py --demo                         # demo 单样本冒烟
    python scripts/train.py --override training.epochs=5   # 临时覆盖单项
"""
import os
import sys
import time
import json
import math
import random
import argparse
import contextlib
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import WebpageDataset, webpage_collate_fn
from src.training.trainer import (
    FigmaGenerationModel,
    compute_losses,
    build_type_targets,
)
from src.models.decoder import build_figma_json


# ──────────────────────────── 工具 ────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def get_memory_pct() -> float:
    try:
        import psutil
        return psutil.virtual_memory().percent
    except ImportError:
        return 0.0


def autocast_ctx(device: torch.device):
    """MPS 还不稳，先 fp32；CUDA 上后续 GPU 训练时再启 bf16。"""
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并：override 中的键覆盖 base，dict 类型会递归合并。"""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, overrides: list[str]) -> dict:
    """
    加载 YAML 配置，自动以 configs/default.yaml 作 base 合并（除非自身就是 default）。
    然后应用 --override key=value。
    """
    default_path = "configs/default.yaml"
    cfg: dict = {}
    if os.path.abspath(path) != os.path.abspath(default_path) and os.path.exists(default_path):
        with open(default_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    with open(path, encoding="utf-8") as f:
        local = yaml.safe_load(f) or {}
    cfg = _deep_merge(cfg, local)
    # 应用 --override 项 形如 training.epochs=5
    for kv in overrides:
        if "=" not in kv:
            continue
        key, value = kv.split("=", 1)
        try:
            value = yaml.safe_load(value)
        except Exception:
            pass
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return cfg


def make_param_groups(model: FigmaGenerationModel, lr_heads: float, lr_codebert: float, wd: float):
    code_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("code_encoder."):
            code_params.append(p)
        else:
            head_params.append(p)
    return [
        {"params": head_params, "lr": lr_heads, "weight_decay": wd},
        {"params": code_params, "lr": lr_codebert, "weight_decay": wd},
    ]


def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    """返回 lr 倍率（0~1）。"""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


# ──────────────────────────── 训练循环 ────────────────────────────

def train_one_epoch(
    model, dataloader, optimizer, device, cfg, epoch, global_step, total_steps,
):
    model.train()
    rs = cfg["resource_safety"]
    tr = cfg["training"]
    loss_w = tr["loss_weights"]
    enable_styles = epoch >= tr.get("enable_styles_after_epoch", 0)
    grad_accum = tr.get("grad_accum", 1)

    epoch_losses: dict[str, float] = {}
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()

    accum_count = 0
    warmup_steps = int(total_steps * tr.get("warmup_ratio", 0.05))

    for step, batch in enumerate(pbar):
        # 学习率调度（per micro-step）
        lr_scale = cosine_warmup_lr(global_step, total_steps, warmup_steps)
        for i, group in enumerate(optimizer.param_groups):
            base_lr = tr["lr_heads"] if i == 0 else tr["lr_codebert"]
            group["lr"] = base_lr * lr_scale

        # 构建 type targets
        target_types = build_type_targets(
            batch["node_texts"], batch["node_mask"].shape[1]
        )

        with autocast_ctx(device):
            outputs = model(
                batch,
                return_styles=enable_styles,
                teacher_force=True,
                target_types=target_types,
            )
            losses = compute_losses(
                outputs,
                target_types=target_types.to(device),
                parents=batch["parents"].to(device),
                node_mask=batch["node_mask"].to(device),
                style_targets=batch["styles"],
                weights=loss_w,
                enable_styles=enable_styles,
            )

        # 梯度累积：每个 micro-step loss / grad_accum
        loss = losses["total"] / grad_accum
        loss.backward()
        accum_count += 1

        if accum_count >= grad_accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=tr.get("grad_clip", 1.0),
            )
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

        # 记录
        for k, v in losses.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + float(v.item())

        pbar.set_postfix({
            "total": f"{losses['total'].item():.3f}",
            "type": f"{losses['type'].item():.3f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

        global_step += 1

        # 资源保护
        if rs.get("empty_cache_every_n_steps") and global_step % rs["empty_cache_every_n_steps"] == 0:
            safe_empty_cache(device)
        if rs.get("sleep_every_n_steps") and global_step % rs["sleep_every_n_steps"] == 0:
            time.sleep(rs.get("sleep_seconds", 0.5))
        if rs.get("mem_check_every_n_steps") and global_step % rs["mem_check_every_n_steps"] == 0:
            mem = get_memory_pct()
            if mem > 0:
                pbar.write(f"[mem] step={global_step} {mem:.1f}%")
                if mem > rs.get("mem_threshold_pct", 85.0):
                    pbar.write(f"⚠️  内存超阈值 {mem:.1f}%，提前结束本 epoch 保存 checkpoint")
                    return epoch_losses, global_step, True

    n = len(dataloader)
    avg = {k: v / max(1, n) for k, v in epoch_losses.items()}
    return avg, global_step, False


@torch.no_grad()
def evaluate(model, dataloader, device, loss_w):
    model.eval()
    total: dict[str, float] = {}
    for batch in dataloader:
        target_types = build_type_targets(
            batch["node_texts"], batch["node_mask"].shape[1]
        )
        outputs = model(
            batch, return_styles=True, teacher_force=True, target_types=target_types,
        )
        losses = compute_losses(
            outputs,
            target_types=target_types.to(device),
            parents=batch["parents"].to(device),
            node_mask=batch["node_mask"].to(device),
            style_targets=batch["styles"],
            weights=loss_w,
            enable_styles=True,
        )
        # 节点类型 acc（用于监控）
        type_pred = outputs["type_logits"].argmax(dim=-1)
        type_acc = ((type_pred == target_types.to(device)) * batch["node_mask"].to(device)).sum() \
                   / batch["node_mask"].to(device).sum().clamp(min=1.0)
        for k, v in losses.items():
            total[k] = total.get(k, 0.0) + float(v.item())
        total["type_acc"] = total.get("type_acc", 0.0) + float(type_acc.item())
    n = len(dataloader)
    return {k: v / max(1, n) for k, v in total.items()}


@torch.no_grad()
def generate_one(model, dataset, idx, device):
    sample = dataset[idx]
    batch = webpage_collate_fn([sample])
    out = model(batch, return_styles=True, teacher_force=False)
    figma = build_figma_json(
        type_indices=out["type_indices"][0].cpu(),
        parent_logits=out["parent_logits"][0].cpu(),
        candidate_mask=out["candidate_mask"][0].cpu(),
        node_mask=out["node_mask"][0].cpu(),
        styles={k: v[0].cpu() for k, v in out["styles"].items()},
        node_texts=batch["node_texts"][0],
        node_boxes=batch["node_boxes"][0].cpu(),
    )
    return figma


# ──────────────────────────── 入口 ────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--override", nargs="*", default=[],
                        help="key=value 临时覆盖，例如 training.epochs=5")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    set_seed(cfg["experiment"].get("seed", 42))
    device = get_device()
    print(f"[device] {device}  [memory] {get_memory_pct():.1f}%")

    # 数据
    if args.demo:
        full = WebpageDataset(use_demo=True, max_nodes=cfg["data"]["max_nodes"])
        train_set = full
        val_set = full
    else:
        full = WebpageDataset(
            data_dir=cfg["data"]["data_dir"],
            max_nodes=cfg["data"]["max_nodes"],
            iou_threshold=cfg["data"]["iou_threshold"],
        )
        n = len(full)
        if n == 0:
            print("❌ 数据集为空"); return
        n_train = max(1, int(n * cfg["data"]["train_ratio"]))
        train_set = Subset(full, list(range(n_train)))
        val_set = Subset(full, list(range(n_train, n))) if n > n_train else train_set
    print(f"[data] train={len(train_set)}  val={len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=webpage_collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=0, collate_fn=webpage_collate_fn,
    )

    # 模型
    model_cfg = cfg["model"]
    model = FigmaGenerationModel(
        dim=model_cfg["dim"],
        align_heads=model_cfg["align_heads"],
        ctx_layers=model_cfg["ctx_layers"],
        ctx_heads=model_cfg["ctx_heads"],
        struct_layers=model_cfg["struct_layers"],
        struct_heads=model_cfg["struct_heads"],
        freeze_visual=model_cfg["freeze_visual"],
        codebert_unfreeze_last=model_cfg["codebert_unfreeze_last"],
    ).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"[model] total={n_total/1e6:.1f}M trainable={n_train/1e6:.1f}M")

    # 优化器
    param_groups = make_param_groups(
        model, cfg["training"]["lr_heads"], cfg["training"]["lr_codebert"],
        cfg["training"]["weight_decay"],
    )
    optimizer = torch.optim.AdamW(param_groups)

    # 日志/输出
    out_dir = cfg["experiment"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log.jsonl")
    log_f = open(log_path, "a", encoding="utf-8")

    # 训练
    epochs = cfg["training"]["epochs"]
    total_steps = epochs * len(train_loader)
    global_step = 0
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        avg, global_step, oom = train_one_epoch(
            model, train_loader, optimizer, device, cfg, epoch, global_step, total_steps,
        )
        log = {"epoch": epoch, "phase": "train", **avg}
        log_f.write(json.dumps(log, ensure_ascii=False) + "\n"); log_f.flush()
        print(f"[train e{epoch}] " + "  ".join(f"{k}={v:.3f}" for k, v in avg.items()))

        if oom:
            ckpt = os.path.join(out_dir, f"oom_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"💾 OOM checkpoint → {ckpt}"); break

        if epoch % cfg["logging"]["eval_every"] == 0:
            ev = evaluate(model, val_loader, device, cfg["training"]["loss_weights"])
            log = {"epoch": epoch, "phase": "val", **ev}
            log_f.write(json.dumps(log, ensure_ascii=False) + "\n"); log_f.flush()
            print(f"[val   e{epoch}] " + "  ".join(f"{k}={v:.3f}" for k, v in ev.items()))
            if ev.get("total", 1e9) < best_val:
                best_val = ev["total"]
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
                print(f"  ✅ best updated  total={best_val:.3f}")

        if epoch % cfg["logging"]["save_every"] == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"epoch{epoch}.pt"))

        if epoch % cfg["logging"]["generate_every"] == 0:
            try:
                figma = generate_one(model, val_set if isinstance(val_set, Subset) else train_set, 0, device)
                with open(os.path.join(out_dir, f"figma_e{epoch}.json"), "w", encoding="utf-8") as f:
                    json.dump(figma, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"  ⚠️  generate failed: {e}")

    log_f.close()
    print("训练完成")


if __name__ == "__main__":
    main()
