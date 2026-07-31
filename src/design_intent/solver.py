"""把模型 head 输出约束解码为无环、可验证的 Design Intent IR。"""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any

import torch

from src.data.intent_dataset import (
    ACTION_TYPES,
    CROSS_ALIGNMENTS,
    ELEMENT_TYPES,
    GROUP_ROLES,
    LAYOUT_MODES,
    PRIMARY_ALIGNMENTS,
    TOKEN_RELATION_KINDS,
)

from .schema import (
    BBox,
    DesignElement,
    DesignGroup,
    DesignIntentIR,
    LayoutConstraint,
    PageGraph,
    PageNode,
    StyleToken,
    TreeEdge,
)


class _UnionFind:
    def __init__(self, values: list[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def clusters(self) -> list[list[int]]:
        result: dict[int, list[int]] = defaultdict(list)
        for value in self.parent:
            result[self.find(value)].append(value)
        return list(result.values())


def _name(node: PageNode, prefix: str) -> str:
    for key in ("aria-label", "id", "name", "class"):
        raw = node.attributes.get(key, "").strip()
        if raw:
            value = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw.split()[0]).strip("_")
            if value:
                return value.lower()
    if node.text:
        return re.sub(r"\s+", "_", node.text.strip())[:32]
    return f"{prefix}_{node.id}"


def _ancestors(node_id: int, nodes: dict[int, PageNode]) -> list[int]:
    result = []
    current = nodes[node_id].parent_id
    seen: set[int] = set()
    while current is not None and current in nodes and current not in seen:
        result.append(current)
        seen.add(current)
        current = nodes[current].parent_id
    return result


def _common_group_anchor(
    member_node_ids: list[int],
    nodes: dict[int, PageNode],
) -> int | None:
    if not member_node_ids:
        return None
    common = set(_ancestors(member_node_ids[0], nodes))
    for node_id in member_node_ids[1:]:
        common &= set(_ancestors(node_id, nodes))
    if not common:
        return None
    return max(common, key=lambda node_id: nodes[node_id].depth)


def _clusters_from_logits(
    logits: torch.Tensor,
    eligible: list[int],
    threshold: float,
) -> list[list[int]]:
    if len(eligible) < 2:
        return []
    probabilities = torch.sigmoid(logits)
    union_find = _UnionFind(eligible)
    for left_index, left in enumerate(eligible):
        for right in eligible[left_index + 1:]:
            if float(probabilities[left, right]) >= threshold:
                union_find.union(left, right)
    return [cluster for cluster in union_find.clusters() if len(cluster) >= 2]


def _most_common(values: list[Any], default: Any) -> Any:
    if not values:
        return default
    return Counter(str(value) for value in values).most_common(1)[0][0]


def _color(style: dict[str, Any], key: str) -> list[float] | None:
    value = style.get(key)
    if isinstance(value, (list, tuple)) and len(value) >= 4 and float(value[3]) > 0.01:
        return [round(float(channel), 4) for channel in value[:4]]
    return None


def _design_style(node: PageNode) -> dict[str, Any]:
    style = node.computed_style
    return {
        "background_color": style.get("backgroundColor"),
        "text_color": style.get("color"),
        "border_color": style.get("borderColor"),
        "border_radius": float(style.get("borderRadius", 0) or 0),
        "border_width": float(style.get("borderWidth", 0) or 0),
        "opacity": float(style.get("opacity", 1) or 1),
        "font_size": float(style.get("fontSize", 16) or 16),
        "font_weight": int(float(style.get("fontWeight", 400) or 400)),
        "text_align": str(style.get("textAlign", "start")),
        "image_src": node.attributes.get("src"),
    }


def _token_value(
    kind: str,
    source_ids: list[int],
    nodes: dict[int, PageNode],
    layouts_by_source: dict[int, LayoutConstraint],
) -> dict[str, Any]:
    styles = [nodes[source_id].computed_style for source_id in source_ids]
    if kind.endswith("_COLOR"):
        property_name = kind.removesuffix("_COLOR").lower()
        style_key = {
            "background": "backgroundColor",
            "foreground": "color",
            "border": "borderColor",
        }[property_name]
        colors = [
            color
            for style in styles
            if (color := _color(style, style_key)) is not None
        ]
        value = Counter(tuple(color) for color in colors).most_common(1)
        return {
            "property": property_name,
            "rgba": list(value[0][0]) if value else [0.0, 0.0, 0.0, 1.0],
        }
    if kind == "TEXT":
        sizes = [round(float(style.get("fontSize", 16) or 16), 1) for style in styles]
        weights = [
            int(float(style.get("fontWeight", 400) or 400)) for style in styles
        ]
        return {
            "font_size": float(_most_common(sizes, 16.0)),
            "font_weight": int(float(_most_common(weights, 400))),
        }
    if kind == "RADIUS":
        radii = [round(float(style.get("borderRadius", 0) or 0), 1) for style in styles]
        return {"radius": float(_most_common(radii, 0.0))}
    if kind == "SPACING":
        gaps = [
            round(layouts_by_source[source_id].gap, 1)
            for source_id in source_ids
            if source_id in layouts_by_source
        ]
        return {"spacing": float(_most_common(gaps, 0.0))}
    return {}


def decode_intent_ir(
    graph: PageGraph,
    outputs: dict[str, torch.Tensor],
    batch_index: int = 0,
    group_threshold: float = 0.6,
    token_threshold: float = 0.6,
) -> DesignIntentIR:
    """约束解码单个 batch 样本。"""
    node_count = len(graph.nodes)
    nodes = {node.id: node for node in graph.nodes}
    action_logits = outputs["action_logits"][batch_index, :node_count].detach().cpu()
    action_probs = torch.softmax(action_logits, dim=-1)
    actions = action_probs.argmax(dim=-1).tolist()

    element_type_logits = outputs["element_type_logits"][
        batch_index, :node_count
    ].detach().cpu()
    role_logits = outputs["role_logits"][batch_index, :node_count].detach().cpu()
    group_logits = outputs["group_affinity_logits"][
        batch_index, :node_count, :node_count
    ].detach().cpu()

    atomic_sources = [
        node_id
        for node_id, action in enumerate(actions)
        if ACTION_TYPES[action] == "ATOMIC" and nodes[node_id].bbox.area > 0
    ]
    group_sources = {
        node_id
        for node_id, action in enumerate(actions)
        if ACTION_TYPES[action] == "GROUP"
    }

    # Pairwise grouping 可补出 action head 漏掉的组候选。
    for cluster in _clusters_from_logits(group_logits, atomic_sources, group_threshold):
        anchor = _common_group_anchor(cluster, nodes)
        if anchor is not None:
            group_sources.add(anchor)

    elements: list[DesignElement] = []
    element_id_by_source: dict[int, str] = {}
    for source_id in atomic_sources:
        type_index = int(element_type_logits[source_id].argmax())
        element_type = ELEMENT_TYPES[type_index]
        element_id = f"e_{source_id}"
        element_id_by_source[source_id] = element_id
        elements.append(
            DesignElement(
                id=element_id,
                source_node_ids=[source_id],
                type=element_type,
                bbox=nodes[source_id].bbox,
                name=_name(nodes[source_id], element_type.lower()),
                text=nodes[source_id].text
                if element_type in {"TEXT", "BUTTON_VISUAL"}
                else "",
                style=_design_style(nodes[source_id]),
                confidence=float(action_probs[source_id].max()),
            )
        )

    groups: list[DesignGroup] = []
    group_id_by_source: dict[int, str] = {}
    for source_id in sorted(group_sources):
        group_id = f"g_{source_id}"
        group_id_by_source[source_id] = group_id
        role_probabilities = torch.softmax(role_logits[source_id], dim=-1)
        role = GROUP_ROLES[int(role_probabilities.argmax())]
        groups.append(
            DesignGroup(
                id=group_id,
                source_element_ids=[],
                role=role,
                bbox=nodes[source_id].bbox,
                name=_name(nodes[source_id], role.lower()),
                style=_design_style(nodes[source_id]),
                source_node_id=source_id,
                confidence=float(
                    max(
                        action_probs[source_id].max(),
                        role_probabilities.max(),
                    )
                ),
            )
        )

    entity_id_by_source = {**element_id_by_source, **group_id_by_source}
    parent_logits = outputs["parent_logits"][batch_index].detach().cpu()
    root_index = parent_logits.shape[-1] - 1
    parent_by_entity: dict[str, str] = {}
    for source_id, entity_id in entity_id_by_source.items():
        child_box = nodes[source_id].bbox
        candidates: list[int] = [root_index]
        for group_source in group_id_by_source:
            if group_source == source_id:
                continue
            parent_box = nodes[group_source].bbox
            if (
                parent_box.area > child_box.area
                and parent_box.contains(child_box)
            ):
                candidates.append(group_source)
        best = max(candidates, key=lambda candidate: float(parent_logits[source_id, candidate]))
        parent_by_entity[entity_id] = (
            group_id_by_source[best] if best in group_id_by_source else "page_root"
        )

    def descendant_elements(group_id: str) -> set[str]:
        result: set[str] = set()
        changed = True
        while changed:
            changed = False
            for child_id, parent_id in parent_by_entity.items():
                if parent_id == group_id or parent_id in result:
                    if child_id.startswith("e_") and child_id not in result:
                        result.add(child_id)
                        changed = True
            for nested_group, parent_id in parent_by_entity.items():
                if parent_id == group_id and nested_group.startswith("g_"):
                    before = len(result)
                    result |= descendant_elements(nested_group)
                    changed |= len(result) != before
        return result

    # 删除没有形成实际聚合的预测组，并把其子实体提升到上一级。
    changed = True
    while changed:
        changed = False
        for group in list(groups):
            members = descendant_elements(group.id)
            if len(members) >= 2:
                continue
            parent_id = parent_by_entity.get(group.id, "page_root")
            for child_id, current_parent in list(parent_by_entity.items()):
                if current_parent == group.id:
                    parent_by_entity[child_id] = parent_id
            parent_by_entity.pop(group.id, None)
            groups.remove(group)
            if group.source_node_id is not None:
                group_id_by_source.pop(group.source_node_id, None)
            changed = True

    for group in groups:
        group.source_element_ids = sorted(descendant_elements(group.id))

    layouts: list[LayoutConstraint] = []
    layout_mode_logits = outputs["layout_mode_logits"][batch_index].detach().cpu()
    primary_logits = outputs["primary_align_logits"][batch_index].detach().cpu()
    cross_logits = outputs["cross_align_logits"][batch_index].detach().cpu()
    gap_output = outputs["gap"][batch_index].detach().cpu()
    padding_output = outputs["padding"][batch_index].detach().cpu()
    scale = max(graph.canvas.width, graph.canvas.height)
    layouts_by_source: dict[int, LayoutConstraint] = {}
    for group in groups:
        source_id = group.source_node_id
        if source_id is None:
            continue
        mode_probabilities = torch.softmax(layout_mode_logits[source_id], dim=-1)
        layout = LayoutConstraint(
            target_id=group.id,
            mode=LAYOUT_MODES[int(mode_probabilities.argmax())],
            gap=float(gap_output[source_id]) * scale,
            padding=[float(value) * scale for value in padding_output[source_id]],
            primary_align=PRIMARY_ALIGNMENTS[int(primary_logits[source_id].argmax())],
            cross_align=CROSS_ALIGNMENTS[int(cross_logits[source_id].argmax())],
            horizontal_resize="STRETCH",
            vertical_resize="HUG",
            confidence=float(mode_probabilities.max()),
        )
        layouts.append(layout)
        layouts_by_source[source_id] = layout

    tree = [
        TreeEdge(parent_id=parent_id, child_id=child_id, order=0)
        for child_id, parent_id in parent_by_entity.items()
    ]
    bbox_by_entity = {
        element.id: element.bbox for element in elements
    } | {group.id: group.bbox for group in groups}
    edges_by_parent: dict[str, list[TreeEdge]] = defaultdict(list)
    for edge in tree:
        edges_by_parent[edge.parent_id].append(edge)
    for edges in edges_by_parent.values():
        edges.sort(
            key=lambda edge: (
                bbox_by_entity[edge.child_id].y,
                bbox_by_entity[edge.child_id].x,
            )
        )
        for order, edge in enumerate(edges):
            edge.order = order

    token_logits = outputs["token_affinity_logits"][batch_index].detach().cpu()
    tokens: list[StyleToken] = []
    entity_id_by_source = {**element_id_by_source, **group_id_by_source}
    token_index = 0
    for kind_index, kind in enumerate(TOKEN_RELATION_KINDS):
        eligible = (
            sorted(group_id_by_source)
            if kind == "SPACING"
            else sorted(element_id_by_source)
        )
        for cluster in _clusters_from_logits(
            token_logits[kind_index], eligible, token_threshold
        ):
            member_ids = [
                entity_id_by_source[source_id]
                for source_id in cluster
                if source_id in entity_id_by_source
            ]
            if len(member_ids) < 2:
                continue
            token_id = f"token_{token_index}"
            token_index += 1
            public_kind = "COLOR" if kind.endswith("_COLOR") else kind
            tokens.append(
                StyleToken(
                    id=token_id,
                    kind=public_kind,
                    value=_token_value(kind, cluster, nodes, layouts_by_source),
                    member_ids=member_ids,
                    name=f"{kind.lower()}/{token_index:02d}",
                    confidence=float(
                        torch.sigmoid(token_logits[kind_index][cluster][:, cluster]).mean()
                    ),
                )
            )
            for element in elements:
                if element.id in member_ids:
                    element.style_token_refs.append(token_id)

    return DesignIntentIR(
        schema_version="1.0",
        canvas=graph.canvas,
        elements=elements,
        groups=groups,
        layouts=layouts,
        tree=tree,
        style_tokens=tokens,
        provenance={
            "source_sample_id": graph.sample_id,
            "label_source": "model_prediction",
            "solver": "constraint_solver_v1",
        },
    )
