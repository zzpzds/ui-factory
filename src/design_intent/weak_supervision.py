"""从 PageGraph 构造可解释、带置信度的设计意图弱标签。"""

from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any

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


TEXT_TAGS = {
    "p", "span", "label", "h1", "h2", "h3", "h4", "h5", "h6",
    "strong", "em", "small", "blockquote", "code", "pre", "time",
}
IMAGE_TAGS = {"img", "picture", "video", "canvas"}
INPUT_TAGS = {"input", "textarea", "select"}
ICON_TAGS = {"svg", "i"}
SEMANTIC_GROUP_TAGS = {
    "nav", "header", "footer", "main", "section", "article", "aside",
    "form", "ul", "ol", "li", "table", "thead", "tbody", "tr",
}
ROLE_BY_TAG = {
    "nav": "NAV",
    "header": "SECTION",
    "footer": "SECTION",
    "main": "SECTION",
    "section": "SECTION",
    "article": "CARD",
    "form": "FORM",
    "li": "LIST_ITEM",
    "ul": "LIST",
    "ol": "LIST",
    "table": "TABLE",
    "tr": "TABLE_ROW",
}


def _css_number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else default


def _has_visible_color(style: dict[str, Any], key: str) -> bool:
    value = style.get(key)
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return float(value[3]) > 0.01
    text = str(value or "").lower()
    return bool(text and text not in {"none", "transparent", "rgba(0, 0, 0, 0)"})


def _element_type(node: PageNode) -> tuple[str | None, float]:
    if node.tag in IMAGE_TAGS:
        return "IMAGE", 0.98
    if node.tag in INPUT_TAGS:
        return "INPUT", 0.98
    if node.tag == "button":
        return "BUTTON_VISUAL", 0.98
    if node.tag in ICON_TAGS:
        return "ICON", 0.85
    if node.tag == "hr":
        return "SHAPE", 0.95
    if node.text and (node.tag in TEXT_TAGS or node.child_count == 0):
        if node.tag == "a" and (
            _has_visible_color(node.computed_style, "backgroundColor")
            or _css_number(node.computed_style.get("borderWidth")) > 0
        ):
            return "BUTTON_VISUAL", 0.75
        return "TEXT", 0.95 if node.tag in TEXT_TAGS else 0.75
    if node.child_count == 0 and (
        _has_visible_color(node.computed_style, "backgroundColor")
        or _css_number(node.computed_style.get("borderWidth")) > 0
    ):
        return "SHAPE", 0.65
    return None, 0.0


def _design_style(node: PageNode) -> dict[str, Any]:
    style = node.computed_style
    return {
        "background_color": style.get("backgroundColor"),
        "text_color": style.get("color"),
        "border_color": style.get("borderColor"),
        "border_radius": _css_number(style.get("borderRadius")),
        "border_width": _css_number(style.get("borderWidth")),
        "opacity": _css_number(style.get("opacity"), 1.0),
        "font_size": _css_number(style.get("fontSize"), 16.0),
        "font_weight": int(_css_number(style.get("fontWeight"), 400.0)),
        "text_align": str(style.get("textAlign", "start")),
        "image_src": node.attributes.get("src"),
    }


def _semantic_name(node: PageNode, fallback: str) -> str:
    for key in ("aria-label", "id", "name", "title", "class"):
        value = node.attributes.get(key, "").strip()
        if value:
            clean = re.sub(r"[^a-zA-Z0-9_\-]+", "_", value.split()[0]).strip("_")
            if clean:
                return clean.lower()
    if node.text:
        clean = re.sub(r"\s+", "_", node.text.strip())[:32]
        if clean:
            return clean
    return fallback


def _ancestors(node_id: int, node_by_id: dict[int, PageNode]) -> list[int]:
    result: list[int] = []
    current = node_by_id[node_id].parent_id
    seen: set[int] = set()
    while current is not None and current in node_by_id and current not in seen:
        result.append(current)
        seen.add(current)
        current = node_by_id[current].parent_id
    return result


def _infer_layout(
    group: DesignGroup,
    source_node: PageNode,
    member_boxes: list[BBox],
) -> LayoutConstraint:
    style = source_node.computed_style
    display = str(style.get("display", "")).lower()
    flex_direction = str(style.get("flexDirection", "")).lower()
    confidence = 0.65
    if display in {"flex", "inline-flex"}:
        mode = "VERTICAL" if flex_direction.startswith("column") else "HORIZONTAL"
        confidence = 0.95
    elif display == "grid":
        mode = "GRID"
        confidence = 0.95
    else:
        mode = _spatial_layout(member_boxes)

    gap = _css_number(style.get("gap"), default=-1.0)
    if gap < 0:
        gap = _infer_gap(member_boxes, mode)
        confidence = min(confidence, 0.75)

    explicit_padding = any(
        key in style for key in ("paddingTop", "paddingRight", "paddingBottom", "paddingLeft")
    )
    if explicit_padding:
        padding = [
            _css_number(style.get("paddingTop")),
            _css_number(style.get("paddingRight")),
            _css_number(style.get("paddingBottom")),
            _css_number(style.get("paddingLeft")),
        ]
    else:
        content = BBox.union(member_boxes)
        padding = [
            max(0.0, content.y - group.bbox.y),
            max(0.0, group.bbox.x2 - content.x2),
            max(0.0, group.bbox.y2 - content.y2),
            max(0.0, content.x - group.bbox.x),
        ]
        confidence = min(confidence, 0.7)

    primary = {
        "flex-start": "START",
        "center": "CENTER",
        "flex-end": "END",
        "space-between": "SPACE_BETWEEN",
    }.get(str(style.get("justifyContent", "")).lower(), "START")
    cross = {
        "flex-start": "START",
        "center": "CENTER",
        "flex-end": "END",
        "stretch": "STRETCH",
    }.get(str(style.get("alignItems", "")).lower(), "START")

    return LayoutConstraint(
        target_id=group.id,
        mode=mode,
        gap=round(gap, 3),
        padding=[round(value, 3) for value in padding],
        primary_align=primary,
        cross_align=cross,
        horizontal_resize="STRETCH" if mode in {"HORIZONTAL", "GRID"} else "FIXED",
        vertical_resize="HUG" if mode in {"HORIZONTAL", "VERTICAL"} else "FIXED",
        confidence=confidence,
    )


def _spatial_layout(boxes: list[BBox]) -> str:
    if len(boxes) < 2:
        return "FREE"
    x_centers = [box.x + box.width / 2 for box in boxes]
    y_centers = [box.y + box.height / 2 for box in boxes]
    x_spread = max(x_centers) - min(x_centers)
    y_spread = max(y_centers) - min(y_centers)
    avg_w = sum(box.width for box in boxes) / len(boxes)
    avg_h = sum(box.height for box in boxes) / len(boxes)
    if x_spread > y_spread * 1.5 and y_spread <= avg_h:
        return "HORIZONTAL"
    if y_spread > x_spread * 1.5 and x_spread <= avg_w:
        return "VERTICAL"
    return "FREE"


def _infer_gap(boxes: list[BBox], mode: str) -> float:
    if len(boxes) < 2 or mode not in {"HORIZONTAL", "VERTICAL"}:
        return 0.0
    if mode == "HORIZONTAL":
        ordered = sorted(boxes, key=lambda box: box.x)
        gaps = [max(0.0, right.x - left.x2) for left, right in zip(ordered, ordered[1:])]
    else:
        ordered = sorted(boxes, key=lambda box: box.y)
        gaps = [max(0.0, lower.y - upper.y2) for upper, lower in zip(ordered, ordered[1:])]
    return float(sum(gaps) / len(gaps)) if gaps else 0.0


def _normalize_color(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    rgba = tuple(round(float(channel), 3) for channel in value[:4])
    return rgba if rgba[3] > 0.01 else None


def _build_style_tokens(
    elements: list[DesignElement],
    groups: list[DesignGroup],
    layouts: list[LayoutConstraint],
    node_by_id: dict[int, PageNode],
) -> list[StyleToken]:
    buckets: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    values: dict[tuple[Any, ...], dict[str, Any]] = {}
    kinds: dict[tuple[Any, ...], str] = {}

    for element in elements:
        style = node_by_id[element.source_node_ids[0]].computed_style
        for style_key, name in (
            ("backgroundColor", "background"),
            ("color", "foreground"),
            ("borderColor", "border"),
        ):
            color = _normalize_color(style.get(style_key))
            if color is not None:
                signature = ("COLOR", name, *color)
                buckets[signature].append(element.id)
                values[signature] = {"property": name, "rgba": list(color)}
                kinds[signature] = "COLOR"

        if element.type == "TEXT":
            font_size = round(_css_number(style.get("fontSize"), 16.0), 1)
            font_weight = int(round(_css_number(style.get("fontWeight"), 400.0) / 100) * 100)
            signature = ("TEXT", font_size, font_weight)
            buckets[signature].append(element.id)
            values[signature] = {"font_size": font_size, "font_weight": font_weight}
            kinds[signature] = "TEXT"

        radius = round(_css_number(style.get("borderRadius")), 1)
        if radius > 0:
            signature = ("RADIUS", radius)
            buckets[signature].append(element.id)
            values[signature] = {"radius": radius}
            kinds[signature] = "RADIUS"

    for layout in layouts:
        if layout.gap > 0:
            gap = round(layout.gap, 1)
            signature = ("SPACING", gap)
            buckets[signature].append(layout.target_id)
            values[signature] = {"spacing": gap}
            kinds[signature] = "SPACING"

    tokens: list[StyleToken] = []
    token_index = 0
    for signature in sorted(buckets, key=str):
        members = sorted(set(buckets[signature]))
        if len(members) < 2:
            continue
        token_id = f"token_{token_index}"
        token_index += 1
        kind = kinds[signature]
        name = f"{kind.lower()}/{token_index:02d}"
        tokens.append(
            StyleToken(
                id=token_id,
                kind=kind,
                value=values[signature],
                member_ids=members,
                name=name,
                confidence=0.8,
            )
        )
        for element in elements:
            if element.id in members:
                element.style_token_refs.append(token_id)
    return tokens


def build_weak_intent(graph: PageGraph) -> DesignIntentIR:
    """构造第一版规则教师输出，用于闭环和后续弱监督预训练。"""
    node_by_id = {node.id: node for node in graph.nodes}
    elements: list[DesignElement] = []
    element_by_node: dict[int, DesignElement] = {}

    for node in graph.nodes:
        element_type, confidence = _element_type(node)
        if element_type is None or node.bbox.area <= 0:
            continue
        element = DesignElement(
            id=f"e_{node.id}",
            source_node_ids=[node.id],
            type=element_type,
            bbox=node.bbox,
            name=_semantic_name(node, f"{element_type.lower()}_{node.id}"),
            text=node.text if element_type in {"TEXT", "BUTTON_VISUAL"} else "",
            style=_design_style(node),
            confidence=confidence,
        )
        elements.append(element)
        element_by_node[node.id] = element

    descendant_elements: dict[int, list[DesignElement]] = defaultdict(list)
    for node_id, element in element_by_node.items():
        for ancestor_id in _ancestors(node_id, node_by_id):
            descendant_elements[ancestor_id].append(element)

    group_candidates: list[tuple[PageNode, list[DesignElement], float]] = []
    for node in graph.nodes:
        members = descendant_elements.get(node.id, [])
        display = str(node.computed_style.get("display", "")).lower()
        semantic = node.tag in SEMANTIC_GROUP_TAGS
        layout_container = display in {"flex", "inline-flex", "grid"}
        if len(members) < 2 or not (semantic or layout_container or node.child_count >= 2):
            continue
        confidence = 0.9 if semantic or layout_container else 0.65
        group_candidates.append((node, members, confidence))

    # 相同成员集合只保留更深、语义更明确的候选，避免生成重复 wrapper。
    selected_by_members: dict[tuple[str, ...], tuple[PageNode, list[DesignElement], float]] = {}
    for candidate in sorted(group_candidates, key=lambda value: value[0].depth):
        node, members, confidence = candidate
        key = tuple(sorted(element.id for element in members))
        previous = selected_by_members.get(key)
        if previous is None:
            selected_by_members[key] = candidate
            continue
        prev_node = previous[0]
        node_score = int(node.tag in SEMANTIC_GROUP_TAGS) * 10 + node.depth
        prev_score = int(prev_node.tag in SEMANTIC_GROUP_TAGS) * 10 + prev_node.depth
        if node_score > prev_score:
            selected_by_members[key] = candidate

    groups: list[DesignGroup] = []
    group_by_node: dict[int, DesignGroup] = {}
    for node, members, confidence in sorted(
        selected_by_members.values(), key=lambda value: value[0].id
    ):
        role = ROLE_BY_TAG.get(node.tag, "CONTAINER")
        group = DesignGroup(
            id=f"g_{node.id}",
            source_element_ids=sorted(element.id for element in members),
            role=role,
            bbox=node.bbox if node.bbox.area > 0 else BBox.union([m.bbox for m in members]),
            name=_semantic_name(node, f"{role.lower()}_{node.id}"),
            style=_design_style(node),
            source_node_id=node.id,
            confidence=confidence,
        )
        groups.append(group)
        group_by_node[node.id] = group

    layouts = [
        _infer_layout(
            group,
            node_by_id[group.source_node_id],
            [element.bbox for element in elements if element.id in group.source_element_ids],
        )
        for group in groups
        if group.source_node_id is not None
    ]

    tree: list[TreeEdge] = []
    entities: list[tuple[str, BBox, int | None, int]] = []
    for group in groups:
        entities.append((group.id, group.bbox, group.source_node_id, 0))
    for element in elements:
        entities.append((element.id, element.bbox, element.source_node_ids[0], 1))

    for entity_id, bbox, source_node_id, entity_kind in entities:
        parent_id = "page_root"
        if source_node_id is not None:
            for ancestor_id in _ancestors(source_node_id, node_by_id):
                candidate = group_by_node.get(ancestor_id)
                if candidate is not None and candidate.id != entity_id:
                    parent_id = candidate.id
                    break
        tree.append(
            TreeEdge(
                parent_id=parent_id,
                child_id=entity_id,
                order=0,
                confidence=0.9 if parent_id != "page_root" else 0.75,
            )
        )

    entity_bbox = {entity_id: bbox for entity_id, bbox, _, _ in entities}
    children_by_parent: dict[str, list[TreeEdge]] = defaultdict(list)
    for edge in tree:
        children_by_parent[edge.parent_id].append(edge)
    for edges in children_by_parent.values():
        edges.sort(
            key=lambda edge: (
                round(entity_bbox[edge.child_id].y, 3),
                round(entity_bbox[edge.child_id].x, 3),
            )
        )
        for order, edge in enumerate(edges):
            edge.order = order

    tokens = _build_style_tokens(elements, groups, layouts, node_by_id)
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
            "label_source": "weak_supervision_v1",
            "warnings": [
                "该文件是规则教师输出，不能作为最终测试真值。",
            ],
        },
    )
