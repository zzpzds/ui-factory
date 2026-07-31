"""不依赖设计师主观评分的可编辑性行为检查。"""

from __future__ import annotations

from collections import defaultdict

from .schema import DesignIntentIR


def token_reference_integrity(ir: DesignIntentIR) -> float:
    token_members = {token.id: set(token.member_ids) for token in ir.style_tokens}
    checks = []
    for element in ir.elements:
        for token_id in element.style_token_refs:
            checks.append(
                token_id in token_members and element.id in token_members[token_id]
            )
    if not checks:
        return 1.0 if not ir.style_tokens else 0.0
    return sum(checks) / len(checks)


def layout_order_consistency(ir: DesignIntentIR, tolerance: float = 1.0) -> float:
    layout_by_group = {layout.target_id: layout for layout in ir.layouts}
    bbox_by_id = {
        element.id: element.bbox for element in ir.elements
    } | {group.id: group.bbox for group in ir.groups}
    children: dict[str, list[str]] = defaultdict(list)
    for edge in ir.tree:
        children[edge.parent_id].append(edge.child_id)

    checks = []
    for group_id, layout in layout_by_group.items():
        child_boxes = [
            bbox_by_id[child_id]
            for child_id in children.get(group_id, [])
            if child_id in bbox_by_id
        ]
        if len(child_boxes) < 2 or layout.mode not in {"HORIZONTAL", "VERTICAL"}:
            continue
        if layout.mode == "HORIZONTAL":
            ordered = sorted(child_boxes, key=lambda box: box.x)
            checks.extend(
                right.x + tolerance >= left.x2
                for left, right in zip(ordered, ordered[1:])
            )
        else:
            ordered = sorted(child_boxes, key=lambda box: box.y)
            checks.extend(
                lower.y + tolerance >= upper.y2
                for upper, lower in zip(ordered, ordered[1:])
            )
    return sum(checks) / len(checks) if checks else 1.0


def compute_editability_checks(ir: DesignIntentIR) -> dict[str, float]:
    return {
        "token_reference_integrity": token_reference_integrity(ir),
        "layout_order_consistency": layout_order_consistency(ir),
    }
