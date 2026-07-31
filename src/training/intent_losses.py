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


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def _masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    class_count = logits.shape[-1]
    safe_target = target.clamp(min=0)
    values = F.cross_entropy(
        logits.reshape(-1, class_count),
        safe_target.reshape(-1),
        reduction="none",
    ).reshape(target.shape)
    return _masked_mean(values, mask * (target >= 0))


def _pairwise_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    node_count = logits.shape[-1]
    diagonal = ~torch.eye(
        node_count, dtype=torch.bool, device=logits.device
    )
    while diagonal.ndim < logits.ndim:
        diagonal = diagonal.unsqueeze(0)
    effective_mask = mask * diagonal
    values = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return _masked_mean(values, effective_mask)


def compute_intent_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    node_mask: torch.Tensor,
    alignment_targets: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    loss_weights = dict(DEFAULT_LOSS_WEIGHTS)
    if weights:
        loss_weights.update(weights)
    device = outputs["action_logits"].device
    targets = {key: value.to(device) for key, value in targets.items()}
    node_mask = node_mask.to(device)

    action_values = F.cross_entropy(
        outputs["action_logits"].transpose(1, 2),
        targets["action"],
        reduction="none",
    )
    action_mask = node_mask * targets["action_confidence"]
    losses = {
        "action": _masked_mean(action_values, action_mask),
        "element": _masked_cross_entropy(
            outputs["element_type_logits"],
            targets["element_type"],
            node_mask,
        ),
        "group": _pairwise_bce(
            outputs["group_affinity_logits"],
            targets["group_affinity"],
            targets["group_pair_mask"],
        ),
        "role": _masked_cross_entropy(
            outputs["role_logits"], targets["role"], node_mask
        ),
        "layout_mode": _masked_cross_entropy(
            outputs["layout_mode_logits"],
            targets["layout_mode"],
            targets["layout_mask"],
        ),
        "token": _pairwise_bce(
            outputs["token_affinity_logits"],
            targets["token_affinity"],
            targets["token_pair_mask"],
        ),
        "tree": _masked_cross_entropy(
            outputs["parent_logits"],
            targets["parent"],
            targets["selected_mask"],
        ),
    }

    layout_mask = targets["layout_mask"]
    losses["layout_reg"] = _masked_mean(
        F.smooth_l1_loss(outputs["gap"], targets["gap"], reduction="none"),
        layout_mask,
    ) + _masked_mean(
        F.smooth_l1_loss(
            outputs["padding"], targets["padding"], reduction="none"
        ),
        layout_mask.unsqueeze(-1).expand_as(outputs["padding"]),
    )
    losses["layout_align"] = (
        _masked_cross_entropy(
            outputs["primary_align_logits"],
            targets["primary_align"],
            layout_mask,
        )
        + _masked_cross_entropy(
            outputs["cross_align_logits"],
            targets["cross_align"],
            layout_mask,
        )
    ) / 2

    if alignment_targets is not None:
        alignment_targets = alignment_targets.to(device)
        align_mask = node_mask.unsqueeze(-1).expand_as(alignment_targets)
        alignment_values = F.binary_cross_entropy_with_logits(
            outputs["alignment_logits"], alignment_targets, reduction="none"
        )
        losses["align"] = _masked_mean(alignment_values, align_mask)

    total = torch.zeros((), device=device)
    for name, loss in losses.items():
        total = total + loss_weights.get(name, 1.0) * loss
    losses["total"] = total
    return losses
