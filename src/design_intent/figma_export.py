"""将 Design Intent IR 确定性导出为可检查的 Figma-like JSON。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schema import BBox, DesignElement, DesignGroup, DesignIntentIR, LayoutConstraint


def _absolute_box(box: BBox) -> dict[str, float]:
    return {
        "x": box.x,
        "y": box.y,
        "width": box.width,
        "height": box.height,
    }


def _paint(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    rgba = [float(channel) for channel in value[:4]]
    if rgba[3] <= 0.01:
        return []
    return [
        {
            "type": "SOLID",
            "color": {
                "r": rgba[0],
                "g": rgba[1],
                "b": rgba[2],
                "a": rgba[3],
            },
        }
    ]


def _attach_visual_style(node: dict[str, Any], style: dict[str, Any]) -> None:
    node["opacity"] = max(0.0, min(1.0, float(style.get("opacity", 1.0))))
    node["cornerRadius"] = max(0.0, float(style.get("border_radius", 0.0)))
    node["strokeWeight"] = max(0.0, float(style.get("border_width", 0.0)))
    node["fills"] = _paint(style.get("background_color"))
    node["strokes"] = _paint(style.get("border_color"))
    node["style"] = {
        "fontSize": max(0.0, float(style.get("font_size", 16.0))),
        "fontWeight": int(style.get("font_weight", 400)),
        "textAlignHorizontal": str(style.get("text_align", "left")).upper(),
        "fontColor": style.get("text_color"),
    }


def _element_node(element: DesignElement) -> dict[str, Any]:
    figma_type = {
        "TEXT": "TEXT",
        "IMAGE": "RECTANGLE",
        "ICON": "VECTOR",
        "SHAPE": "RECTANGLE",
        "INPUT": "FRAME",
        "BUTTON_VISUAL": "FRAME",
    }.get(element.type, "FRAME")
    node: dict[str, Any] = {
        "id": element.id,
        "name": element.name,
        "type": figma_type,
        "absoluteBoundingBox": _absolute_box(element.bbox),
        "sourceNodeIds": element.source_node_ids,
        "styleTokenRefs": element.style_token_refs,
    }
    if element.text:
        node["characters"] = element.text
    _attach_visual_style(node, element.style)
    if figma_type == "RECTANGLE" and element.type == "IMAGE":
        node["fills"] = [
            {
                "type": "IMAGE",
                "scaleMode": "FILL",
                "imageRef": element.style.get("image_src"),
            }
        ]
    return node


def _group_node(
    group: DesignGroup,
    layout: LayoutConstraint | None,
) -> dict[str, Any]:
    node: dict[str, Any] = {
        "id": group.id,
        "name": group.name,
        "type": "FRAME",
        "role": group.role,
        "absoluteBoundingBox": _absolute_box(group.bbox),
        "clipsContent": False,
        "children": [],
    }
    _attach_visual_style(node, group.style)
    if layout is not None:
        if layout.mode in {"HORIZONTAL", "VERTICAL"}:
            node["layoutMode"] = layout.mode
        elif layout.mode == "GRID":
            # Figma REST 节点缺少统一 GRID 写法，保留在扩展字段中。
            node["layoutMode"] = "NONE"
            node["designIntentLayoutMode"] = "GRID"
        else:
            node["layoutMode"] = "NONE"
        node.update(
            {
                "itemSpacing": layout.gap,
                "paddingTop": layout.padding[0],
                "paddingRight": layout.padding[1],
                "paddingBottom": layout.padding[2],
                "paddingLeft": layout.padding[3],
                "primaryAxisAlignItems": layout.primary_align,
                "counterAxisAlignItems": layout.cross_align,
                "designIntentResize": {
                    "horizontal": layout.horizontal_resize,
                    "vertical": layout.vertical_resize,
                },
            }
        )
    return node


def export_figma_json(ir: DesignIntentIR) -> dict[str, Any]:
    """生成结构合法且可追溯的 Figma-like 文档树。"""
    layouts = {layout.target_id: layout for layout in ir.layouts}
    nodes: dict[str, dict[str, Any]] = {
        element.id: _element_node(element) for element in ir.elements
    }
    nodes.update(
        {
            group.id: _group_node(group, layouts.get(group.id))
            for group in ir.groups
        }
    )

    child_edges: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for edge in ir.tree:
        child_edges[edge.parent_id].append((edge.order, edge.child_id))
    for edges in child_edges.values():
        edges.sort(key=lambda item: item[0])

    def build(entity_id: str, ancestors: frozenset[str] = frozenset()) -> dict[str, Any]:
        if entity_id in ancestors:
            raise ValueError(f"设计树存在环：{entity_id}")
        node = dict(nodes[entity_id])
        children = child_edges.get(entity_id, [])
        if children:
            next_ancestors = ancestors | {entity_id}
            node["children"] = [
                build(child_id, next_ancestors) for _, child_id in children
            ]
        return node

    root_children = [
        build(child_id) for _, child_id in child_edges.get("page_root", [])
    ]
    canvas = {
        "id": "canvas_0",
        "name": "Recovered Design",
        "type": "CANVAS",
        "children": root_children,
        "backgroundColor": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
    }
    return {
        "schemaVersion": 0,
        "name": f"Recovered-{ir.provenance.get('source_sample_id', 'page')}",
        "document": {
            "id": "document_0",
            "name": "Document",
            "type": "DOCUMENT",
            "children": [canvas],
        },
        "components": {},
        "componentSets": {},
        "styles": {
            token.id: {
                "key": token.id,
                "name": token.name,
                "styleType": token.kind,
                "value": token.value,
                "members": token.member_ids,
            }
            for token in ir.style_tokens
        },
        "designIntent": {
            "schemaVersion": ir.schema_version,
            "canvas": {
                "width": ir.canvas.width,
                "height": ir.canvas.height,
            },
            "provenance": ir.provenance,
        },
    }
