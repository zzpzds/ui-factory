"""设计树中的直接子实体与 group 后代原子覆盖。"""

from __future__ import annotations

from collections import defaultdict

from .schema import DesignIntentIR


def direct_children(ir: DesignIntentIR, parent_id: str) -> list[str]:
    """按 tree order 返回一个 group 的直接设计子实体。"""
    entity_ids = {
        element.id for element in ir.elements
    } | {group.id for group in ir.groups}
    edges = sorted(
        (
            edge
            for edge in ir.tree
            if edge.parent_id == parent_id and edge.child_id in entity_ids
        ),
        key=lambda edge: (edge.order, edge.child_id),
    )
    return list(dict.fromkeys(edge.child_id for edge in edges))


def descendant_element_ids(
    ir: DesignIntentIR,
    entity_id: str,
) -> list[str]:
    """返回实体覆盖的全部后代原子元素，允许 group 只直接包含子 group。"""
    element_order = {
        element.id: index for index, element in enumerate(ir.elements)
    }
    element_ids = set(element_order)
    group_ids = {group.id for group in ir.groups}
    children: dict[str, list[str]] = defaultdict(list)
    for edge in sorted(ir.tree, key=lambda item: (item.order, item.child_id)):
        if edge.child_id in element_ids | group_ids:
            children[edge.parent_id].append(edge.child_id)

    def collect(current_id: str, visiting: set[str]) -> set[str]:
        if current_id in element_ids:
            return {current_id}
        if current_id not in group_ids or current_id in visiting:
            return set()
        next_visiting = visiting | {current_id}
        result: set[str] = set()
        for child_id in children.get(current_id, []):
            result |= collect(child_id, next_visiting)
        return result

    return sorted(
        collect(entity_id, set()),
        key=lambda element_id: (element_order[element_id], element_id),
    )


def normalize_group_source_elements(
    ir: DesignIntentIR,
) -> dict[str, tuple[list[str], list[str]]]:
    """
    用设计树重建每个 group 的扁平后代原子覆盖。

    tree 表达直接父子关系；source_element_ids 只服务于分组监督和指标签名，
    二者不能再由人工分别维护。
    """
    changes = {}
    for group in ir.groups:
        before = list(group.source_element_ids)
        after = descendant_element_ids(ir, group.id)
        group.source_element_ids = after
        if before != after:
            changes[group.id] = (before, after)
    return changes


def validate_group_structure(ir: DesignIntentIR) -> list[str]:
    """检查 group 的直接子实体规则和扁平覆盖一致性。"""
    errors = []
    element_ids = {element.id for element in ir.elements}
    group_ids = {group.id for group in ir.groups}
    for group in ir.groups:
        children = direct_children(ir, group.id)
        if not children:
            errors.append(f"组 {group.id} 至少需要一个直接设计子实体")
        elif len(children) == 1 and children[0] in element_ids:
            errors.append(
                f"组 {group.id} 不能只包含一个直接原子元素；"
                "请选择至少两个实体或一个子分组"
            )

        expected = descendant_element_ids(ir, group.id)
        if not expected:
            errors.append(f"组 {group.id} 必须覆盖至少一个后代原子元素")
        if set(group.source_element_ids) != set(expected):
            errors.append(
                f"组 {group.id} 的 source_element_ids 与设计树后代不一致"
            )
        missing_group_children = [
            child_id
            for child_id in children
            if child_id not in element_ids | group_ids
        ]
        if missing_group_children:
            errors.append(
                f"组 {group.id} 引用了不存在的直接子实体"
                f" {missing_group_children}"
            )
    return errors
