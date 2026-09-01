"""Design Intent 多任务损失。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


DEFAULT_LOSS_WEIGHTS = {
    "action": 1.0,
    "element": 0.8,
    "group": 0.8,
    "role": 0.3,
    "layout_mode": 0.5,
    "layout_reg": 0.3,
    "layout_align": 0.2,
    "token": 0.5,
    "tree": 0.8,
    "align": 0.2,
}


def _masked_mean_per_page(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回每页 masked mean 及该页是否至少有一个有效标签。"""
    reduce_dims = tuple(range(1, values.ndim))
    weighted_sum = (values * mask).sum(dim=reduce_dims)
    denominator = mask.sum(dim=reduce_dims)
    return weighted_sum / denominator.clamp(min=1.0), denominator > 0


def _masked_cross_entropy_per_page(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    class_count = logits.shape[-1]
    safe_target = target.clamp(min=0)
    values = F.cross_entropy(
        logits.reshape(-1, class_count),
        safe_target.reshape(-1),
        reduction="none",
    ).reshape(target.shape)
    return _masked_mean_per_page(values, mask * (target >= 0))


def _pairwise_bce_per_page(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    node_count = logits.shape[-1]
    diagonal = ~torch.eye(
        node_count, dtype=torch.bool, device=logits.device
    )
    while diagonal.ndim < logits.ndim:
        diagonal = diagonal.unsqueeze(0)
    effective_mask = mask * diagonal
    values = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return _masked_mean_per_page(values, effective_mask)


def _annotation_page_mean(
    values: torch.Tensor,
    available: torch.Tensor,
    page_weights: torch.Tensor,
) -> torch.Tensor:
    return (values * available * page_weights).sum() / values.shape[0]


def _unweighted_page_mean(
    values: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    return (values * available).sum() / values.shape[0]


def compute_intent_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    node_mask: torch.Tensor,
    alignment_targets: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
    annotation_page_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    loss_weights = dict(DEFAULT_LOSS_WEIGHTS)
    if weights:
        loss_weights.update(weights)
    device = outputs["action_logits"].device
    targets = {key: value.to(device) for key, value in targets.items()}
    node_mask = node_mask.to(device)
    batch_size = node_mask.shape[0]
    if annotation_page_weights is None:
        page_weights = torch.ones(batch_size, device=device)
    else:
        page_weights = torch.as_tensor(
            annotation_page_weights,
            dtype=torch.float32,
            device=device,
        )
        if page_weights.shape != (batch_size,):
            raise ValueError(
                "annotation_page_weights 必须是与 batch 对齐的一维张量。"
            )
        if not torch.isfinite(page_weights).all() or (page_weights <= 0).any():
            raise ValueError("annotation_page_weights 必须有限且严格大于零。")

    action_values = F.cross_entropy(
        outputs["action_logits"].transpose(1, 2),
        targets["action"],
        reduction="none",
    )
    action_mask = (
        node_mask
        * targets["action_confidence"]
        * targets.get("action_mask", node_mask)
    )
    per_page = {
        "action": _masked_mean_per_page(action_values, action_mask),
        "element": _masked_cross_entropy_per_page(
            outputs["element_type_logits"],
            targets["element_type"],
            node_mask,
        ),
        "group": _pairwise_bce_per_page(
            outputs["group_affinity_logits"],
            targets["group_affinity"],
            targets["group_pair_mask"],
        ),
        "role": _masked_cross_entropy_per_page(
            outputs["role_logits"], targets["role"], node_mask
        ),
        "layout_mode": _masked_cross_entropy_per_page(
            outputs["layout_mode_logits"],
            targets["layout_mode"],
            targets["layout_mask"],
        ),
        "token": _pairwise_bce_per_page(
            outputs["token_affinity_logits"],
            targets["token_affinity"],
            targets["token_pair_mask"],
        ),
        "tree": _masked_cross_entropy_per_page(
            outputs["parent_logits"],
            targets["parent"],
            targets.get("tree_mask", targets["selected_mask"]),
        ),
    }

    layout_mask = targets["layout_mask"]
    gap_values, gap_available = _masked_mean_per_page(
        F.smooth_l1_loss(outputs["gap"], targets["gap"], reduction="none"),
        layout_mask,
    )
    padding_values, padding_available = _masked_mean_per_page(
        F.smooth_l1_loss(
            outputs["padding"], targets["padding"], reduction="none"
        ),
        layout_mask.unsqueeze(-1).expand_as(outputs["padding"]),
    )
    per_page["layout_reg"] = (
        gap_values + padding_values,
        gap_available | padding_available,
    )
    primary_values, primary_available = _masked_cross_entropy_per_page(
        outputs["primary_align_logits"],
        targets["primary_align"],
        layout_mask,
    )
    cross_values, cross_available = _masked_cross_entropy_per_page(
        outputs["cross_align_logits"],
        targets["cross_align"],
        layout_mask,
    )
    per_page["layout_align"] = (
        (primary_values + cross_values) / 2,
        primary_available | cross_available,
    )

    losses = {
        name: _annotation_page_mean(values, available, page_weights)
        for name, (values, available) in per_page.items()
    }

    if alignment_targets is not None:
        alignment_targets = alignment_targets.to(device)
        align_mask = node_mask.unsqueeze(-1).expand_as(alignment_targets)
        alignment_values = F.binary_cross_entropy_with_logits(
            outputs["alignment_logits"], alignment_targets, reduction="none"
        )
        align_per_page, align_available = _masked_mean_per_page(
            alignment_values, align_mask
        )
        losses["align"] = _unweighted_page_mean(
            align_per_page, align_available
        )

    total = torch.zeros((), device=device)
    for name, loss in losses.items():
        total = total + loss_weights.get(name, 1.0) * loss
    losses["total"] = total
    return losses
