"""PageGraph 与 Design Intent IR 的确定性校验。"""

from __future__ import annotations

from .grouping import validate_group_structure
from .schema import DesignIntentIR, PageGraph


def validate_page_graph(graph: PageGraph) -> list[str]:
    errors: list[str] = []
    if graph.canvas.width <= 0 or graph.canvas.height <= 0:
        errors.append("canvas 尺寸必须为正数")

    ids = [node.id for node in graph.nodes]
    if len(ids) != len(set(ids)):
        errors.append("PageGraph 节点 id 不唯一")
    if ids != list(range(len(ids))):
        errors.append("PageGraph 节点 id 必须按 0..N-1 连续排列")
    id_set = set(ids)

    for node in graph.nodes:
        if node.parent_id is not None and node.parent_id not in id_set:
            errors.append(f"节点 {node.id} 引用了不存在的父节点 {node.parent_id}")
        if node.parent_id == node.id:
            errors.append(f"节点 {node.id} 不能以自身为父节点")
        if node.bbox.width < 0 or node.bbox.height < 0:
            errors.append(f"节点 {node.id} 的 bbox 尺寸非法")
        tolerance = 1e-3
        if (
            node.bbox.x < -tolerance
            or node.bbox.y < -tolerance
            or node.bbox.x2 > graph.canvas.width + tolerance
            or node.bbox.y2 > graph.canvas.height + tolerance
        ):
            errors.append(f"节点 {node.id} 的 bbox 超出首屏画布")

    parent_of = {node.id: node.parent_id for node in graph.nodes}
    for node_id in ids:
        seen: set[int] = set()
        current: int | None = node_id
        while current is not None:
            if current in seen:
                errors.append(f"PageGraph 存在父子环，涉及节点 {node_id}")
                break
            seen.add(current)
            current = parent_of.get(current)
    return errors


def validate_intent_ir(ir: DesignIntentIR, graph: PageGraph | None = None) -> list[str]:
    errors: list[str] = []
    element_ids = {element.id for element in ir.elements}
    group_ids = {group.id for group in ir.groups}
    if len(element_ids) != len(ir.elements):
        errors.append("DesignIntentIR element id 不唯一")
    if len(group_ids) != len(ir.groups):
        errors.append("DesignIntentIR group id 不唯一")
    if element_ids & group_ids:
        errors.append("element 与 group id 不能重复")

    entity_ids = element_ids | group_ids | {"page_root"}
    source_node_ids = {node.id for node in graph.nodes} if graph is not None else None

    for element in ir.elements:
        if not element.source_node_ids:
            errors.append(f"元素 {element.id} 缺少 source_node_ids")
        if source_node_ids is not None:
            missing = set(element.source_node_ids) - source_node_ids
            if missing:
                errors.append(f"元素 {element.id} 引用了不存在的源节点 {sorted(missing)}")

    for group in ir.groups:
        missing = set(group.source_element_ids) - element_ids
        if missing:
            errors.append(f"组 {group.id} 引用了不存在的元素 {sorted(missing)}")

    parent_count: dict[str, int] = {}
    adjacency: dict[str, list[str]] = {}
    for edge in ir.tree:
        if edge.parent_id not in entity_ids:
            errors.append(f"树边引用不存在的父实体 {edge.parent_id}")
        if edge.child_id not in entity_ids - {"page_root"}:
            errors.append(f"树边引用不存在的子实体 {edge.child_id}")
        if edge.parent_id == edge.child_id:
            errors.append(f"实体 {edge.child_id} 不能以自身为父节点")
        parent_count[edge.child_id] = parent_count.get(edge.child_id, 0) + 1
        adjacency.setdefault(edge.parent_id, []).append(edge.child_id)

    for entity_id in element_ids | group_ids:
        count = parent_count.get(entity_id, 0)
        if count != 1:
            errors.append(f"实体 {entity_id} 必须有且仅有一个父节点，当前为 {count}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(entity_id: str) -> None:
        if entity_id in visiting:
            errors.append(f"DesignIntentIR 设计树存在环，涉及 {entity_id}")
            return
        if entity_id in visited:
            return
        visiting.add(entity_id)
        for child_id in adjacency.get(entity_id, []):
            visit(child_id)
        visiting.remove(entity_id)
        visited.add(entity_id)

    visit("page_root")
    for entity_id in element_ids | group_ids:
        visit(entity_id)

    for layout in ir.layouts:
        if layout.target_id not in group_ids:
            errors.append(f"布局约束引用了非 group 实体 {layout.target_id}")
        if len(layout.padding) != 4:
            errors.append(f"布局 {layout.target_id} 的 padding 必须有 4 个值")

    valid_token_members = element_ids | group_ids
    token_ids = {token.id for token in ir.style_tokens}
    if len(token_ids) != len(ir.style_tokens):
        errors.append("style token id 不唯一")
    for token in ir.style_tokens:
        missing = set(token.member_ids) - valid_token_members
        if missing:
            errors.append(f"token {token.id} 引用了不存在的成员 {sorted(missing)}")
    for element in ir.elements:
        missing = set(element.style_token_refs) - token_ids
        if missing:
            errors.append(f"元素 {element.id} 引用了不存在的 token {sorted(missing)}")

    errors.extend(validate_group_structure(ir))
    return errors
