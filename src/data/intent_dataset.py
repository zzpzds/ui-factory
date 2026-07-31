"""Design Intent 数据集：PageGraph + Intent IR → 多任务监督张量。"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.design_intent.schema import DesignIntentIR, PageGraph, PageNode
from src.design_intent.validation import validate_intent_ir, validate_page_graph
from src.design_intent.weak_supervision import build_weak_intent


ACTION_TYPES = ["DROP", "ATOMIC", "GROUP", "MERGE"]
ELEMENT_TYPES = ["TEXT", "IMAGE", "ICON", "SHAPE", "INPUT", "BUTTON_VISUAL"]
GROUP_ROLES = [
    "CONTAINER", "CARD", "NAV", "FORM", "LIST", "LIST_ITEM",
    "TABLE", "TABLE_ROW", "SECTION", "UNKNOWN",
]
LAYOUT_MODES = ["HORIZONTAL", "VERTICAL", "GRID", "FREE"]
PRIMARY_ALIGNMENTS = ["START", "CENTER", "END", "SPACE_BETWEEN"]
CROSS_ALIGNMENTS = ["START", "CENTER", "END", "STRETCH"]
TOKEN_RELATION_KINDS = [
    "BACKGROUND_COLOR",
    "FOREGROUND_COLOR",
    "BORDER_COLOR",
    "TEXT",
    "RADIUS",
    "SPACING",
]

ACTION_TO_IDX = {value: index for index, value in enumerate(ACTION_TYPES)}
ELEMENT_TO_IDX = {value: index for index, value in enumerate(ELEMENT_TYPES)}
ROLE_TO_IDX = {value: index for index, value in enumerate(GROUP_ROLES)}
LAYOUT_TO_IDX = {value: index for index, value in enumerate(LAYOUT_MODES)}
PRIMARY_TO_IDX = {value: index for index, value in enumerate(PRIMARY_ALIGNMENTS)}
CROSS_TO_IDX = {value: index for index, value in enumerate(CROSS_ALIGNMENTS)}
TOKEN_KIND_TO_IDX = {
    value: index for index, value in enumerate(TOKEN_RELATION_KINDS)
}

TAG_VOCAB = [
    "<unk>", "div", "span", "p", "a", "button", "img", "svg", "input",
    "textarea", "select", "label", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "footer", "nav", "main", "section", "article", "aside",
    "form", "ul", "ol", "li", "table", "thead", "tbody", "tr", "td", "th",
]
TAG_TO_IDX = {value: index for index, value in enumerate(TAG_VOCAB)}
STRUCTURED_FEATURE_DIM = 25


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = "".join(ch for ch in str(value) if ch in "-.0123456789")
        return float(text) if text else default
    except (TypeError, ValueError):
        return default


def _node_features(node: PageNode, graph: PageGraph) -> list[float]:
    style = node.computed_style
    width = max(graph.canvas.width, 1.0)
    height = max(graph.canvas.height, 1.0)
    area_ratio = node.bbox.area / (width * height)
    aspect = math.log1p(node.bbox.width / max(node.bbox.height, 1e-6))
    display = str(style.get("display", "block")).lower()
    position = str(style.get("position", "static")).lower()
    return [
        node.bbox.x / width,
        node.bbox.y / height,
        node.bbox.width / width,
        node.bbox.height / height,
        area_ratio,
        aspect / 5.0,
        min(node.depth, 32) / 32.0,
        min(node.sibling_index, 32) / 32.0,
        min(node.child_count, 32) / 32.0,
        _number(style.get("borderRadius")) / 50.0,
        _number(style.get("borderWidth")) / 10.0,
        min(max(_number(style.get("opacity"), 1.0), 0.0), 1.0),
        _number(style.get("fontSize")) / 64.0,
        _number(style.get("fontWeight"), 400.0) / 900.0,
        _number(style.get("gap")) / 100.0,
        _number(style.get("paddingTop")) / 100.0,
        _number(style.get("paddingRight")) / 100.0,
        _number(style.get("paddingBottom")) / 100.0,
        _number(style.get("paddingLeft")) / 100.0,
        float(display == "block"),
        float(display in {"inline", "inline-block"}),
        float(display in {"flex", "inline-flex"}),
        float(display == "grid"),
        float(position in {"absolute", "fixed"}),
        float(bool(node.text)),
    ]


def _serialize_node(node: PageNode) -> str:
    attrs = " ".join(
        f'{key}="{value}"'
        for key, value in sorted(node.attributes.items())
        if key in {"id", "class", "role", "aria-label", "type", "name"}
    )
    prefix = f"<{node.tag}{(' ' + attrs) if attrs else ''}>"
    return f"{prefix}{node.text[:100]}</{node.tag}>"


def _normalized_xyxy(node: PageNode, graph: PageGraph) -> list[float]:
    width = max(graph.canvas.width, 1.0)
    height = max(graph.canvas.height, 1.0)
    return [
        node.bbox.x / width,
        node.bbox.y / height,
        node.bbox.x2 / width,
        node.bbox.y2 / height,
    ]


def _alignment_targets(boxes: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    centers = torch.stack(
        torch.meshgrid(
            (torch.arange(14, dtype=torch.float32) + 0.5) / 14.0,
            (torch.arange(14, dtype=torch.float32) + 0.5) / 14.0,
            indexing="ij",
        ),
        dim=-1,
    )
    patch_y = centers[..., 0].reshape(-1)
    patch_x = centers[..., 1].reshape(-1)
    labels = (
        (patch_x.unsqueeze(0) >= boxes[:, 0:1])
        & (patch_x.unsqueeze(0) <= boxes[:, 2:3])
        & (patch_y.unsqueeze(0) >= boxes[:, 1:2])
        & (patch_y.unsqueeze(0) <= boxes[:, 3:4])
    ).float()
    return labels * node_mask.unsqueeze(-1)


def build_intent_targets(
    graph: PageGraph,
    ir: DesignIntentIR,
    max_nodes: int,
) -> dict[str, torch.Tensor]:
    n = min(len(graph.nodes), max_nodes)
    action = torch.full((max_nodes,), ACTION_TO_IDX["DROP"], dtype=torch.long)
    action_confidence = torch.zeros(max_nodes)
    action_confidence[:n] = 0.5
    element_type = torch.full((max_nodes,), -100, dtype=torch.long)
    role = torch.full((max_nodes,), -100, dtype=torch.long)
    layout_mode = torch.full((max_nodes,), -100, dtype=torch.long)
    primary_align = torch.full((max_nodes,), -100, dtype=torch.long)
    cross_align = torch.full((max_nodes,), -100, dtype=torch.long)
    gap = torch.zeros(max_nodes)
    padding = torch.zeros(max_nodes, 4)
    layout_mask = torch.zeros(max_nodes)
    selected_mask = torch.zeros(max_nodes)

    entity_source: dict[str, int] = {}
    element_source: dict[str, int] = {}
    for element in ir.elements:
        if not element.source_node_ids:
            continue
        source_id = element.source_node_ids[0]
        if source_id >= n:
            continue
        action[source_id] = ACTION_TO_IDX["ATOMIC"]
        action_confidence[source_id] = element.confidence
        element_type[source_id] = ELEMENT_TO_IDX.get(element.type, -100)
        selected_mask[source_id] = 1.0
        entity_source[element.id] = source_id
        element_source[element.id] = source_id

    for group in ir.groups:
        if group.source_node_id is None or group.source_node_id >= n:
            continue
        source_id = group.source_node_id
        action[source_id] = ACTION_TO_IDX["GROUP"]
        action_confidence[source_id] = group.confidence
        role[source_id] = ROLE_TO_IDX.get(group.role, ROLE_TO_IDX["UNKNOWN"])
        selected_mask[source_id] = 1.0
        entity_source[group.id] = source_id

    layout_by_target = {layout.target_id: layout for layout in ir.layouts}
    for group in ir.groups:
        if group.id not in layout_by_target or group.id not in entity_source:
            continue
        source_id = entity_source[group.id]
        layout = layout_by_target[group.id]
        layout_mode[source_id] = LAYOUT_TO_IDX.get(layout.mode, LAYOUT_TO_IDX["FREE"])
        primary_align[source_id] = PRIMARY_TO_IDX.get(layout.primary_align, 0)
        cross_align[source_id] = CROSS_TO_IDX.get(layout.cross_align, 0)
        normalizer = max(graph.canvas.width, graph.canvas.height, 1.0)
        gap[source_id] = layout.gap / normalizer
        padding[source_id] = torch.tensor(layout.padding) / normalizer
        layout_mask[source_id] = layout.confidence

    parent_target = torch.full((max_nodes,), -100, dtype=torch.long)
    parent_by_child = {edge.child_id: edge.parent_id for edge in ir.tree}
    for entity_id, source_id in entity_source.items():
        parent_id = parent_by_child.get(entity_id, "page_root")
        parent_target[source_id] = (
            entity_source[parent_id] if parent_id in entity_source else max_nodes
        )

    group_affinity = torch.zeros(max_nodes, max_nodes)
    group_pair_mask = torch.zeros(max_nodes, max_nodes)
    direct_group: dict[str, str] = {}
    for element_id in element_source:
        parent_id = parent_by_child.get(element_id, "page_root")
        direct_group[element_id] = parent_id
    element_ids = sorted(element_source)
    for left_id in element_ids:
        left = element_source[left_id]
        for right_id in element_ids:
            right = element_source[right_id]
            group_pair_mask[left, right] = 1.0
            if direct_group[left_id] == direct_group[right_id]:
                group_affinity[left, right] = 1.0

    token_affinity = torch.zeros(
        len(TOKEN_RELATION_KINDS), max_nodes, max_nodes
    )
    token_pair_mask = torch.zeros_like(token_affinity)
    token_members_by_kind: dict[str, list[set[int]]] = {
        kind: [] for kind in TOKEN_RELATION_KINDS
    }
    for token in ir.style_tokens:
        if token.kind == "COLOR":
            property_name = str(token.value.get("property", "foreground")).upper()
            relation_kind = f"{property_name}_COLOR"
        else:
            relation_kind = token.kind
        if relation_kind not in token_members_by_kind:
            continue
        source_members = {
            entity_source[member_id]
            for member_id in token.member_ids
            if member_id in entity_source
        }
        if source_members:
            token_members_by_kind[relation_kind].append(source_members)

    element_source_ids = set(element_source.values())
    group_source_ids = {
        entity_source[group.id] for group in ir.groups if group.id in entity_source
    }
    for kind, kind_index in TOKEN_KIND_TO_IDX.items():
        eligible = group_source_ids if kind == "SPACING" else element_source_ids
        for left in eligible:
            for right in eligible:
                token_pair_mask[kind_index, left, right] = 1.0
        for cluster in token_members_by_kind[kind]:
            for left in cluster:
                for right in cluster:
                    token_affinity[kind_index, left, right] = 1.0

    return {
        "action": action,
        "action_confidence": action_confidence,
        "element_type": element_type,
        "role": role,
        "layout_mode": layout_mode,
        "primary_align": primary_align,
        "cross_align": cross_align,
        "gap": gap,
        "padding": padding,
        "layout_mask": layout_mask,
        "selected_mask": selected_mask,
        "parent": parent_target,
        "group_affinity": group_affinity,
        "group_pair_mask": group_pair_mask,
        "token_affinity": token_affinity,
        "token_pair_mask": token_pair_mask,
    }


def truncate_graph_and_ir(
    graph: PageGraph,
    ir: DesignIntentIR,
    max_nodes: int,
) -> tuple[PageGraph, DesignIntentIR]:
    """同步截断输入图与标注，并重接设计树。"""
    if len(graph.nodes) <= max_nodes:
        return graph, ir

    graph = deepcopy(graph)
    ir = deepcopy(ir)
    graph.nodes = graph.nodes[:max_nodes]
    kept_node_ids = {node.id for node in graph.nodes}
    for node in graph.nodes:
        if node.parent_id not in kept_node_ids:
            node.parent_id = None

    ir.elements = [
        element
        for element in ir.elements
        if any(source_id in kept_node_ids for source_id in element.source_node_ids)
    ]
    for element in ir.elements:
        element.source_node_ids = [
            source_id
            for source_id in element.source_node_ids
            if source_id in kept_node_ids
        ]
    kept_element_ids = {element.id for element in ir.elements}

    ir.groups = [
        group
        for group in ir.groups
        if group.source_node_id in kept_node_ids
    ]
    for group in ir.groups:
        group.source_element_ids = [
            element_id
            for element_id in group.source_element_ids
            if element_id in kept_element_ids
        ]
    ir.groups = [
        group for group in ir.groups if len(group.source_element_ids) >= 2
    ]
    kept_group_ids = {group.id for group in ir.groups}
    kept_entity_ids = kept_element_ids | kept_group_ids

    original_parent = {
        edge.child_id: edge.parent_id for edge in ir.tree
    }
    new_tree = []
    for child_id in kept_entity_ids:
        parent_id = original_parent.get(child_id, "page_root")
        seen = {child_id}
        while parent_id not in kept_group_ids and parent_id != "page_root":
            if parent_id in seen:
                parent_id = "page_root"
                break
            seen.add(parent_id)
            parent_id = original_parent.get(parent_id, "page_root")
        new_tree.append(
            {
                "parent_id": parent_id,
                "child_id": child_id,
            }
        )

    bbox_by_entity = {
        element.id: element.bbox for element in ir.elements
    } | {group.id: group.bbox for group in ir.groups}
    new_tree.sort(
        key=lambda edge: (
            edge["parent_id"],
            bbox_by_entity[edge["child_id"]].y,
            bbox_by_entity[edge["child_id"]].x,
        )
    )
    order_by_parent: dict[str, int] = {}
    ir.tree = []
    from src.design_intent.schema import TreeEdge

    for edge in new_tree:
        parent_id = edge["parent_id"]
        order = order_by_parent.get(parent_id, 0)
        order_by_parent[parent_id] = order + 1
        ir.tree.append(
            TreeEdge(
                parent_id=parent_id,
                child_id=edge["child_id"],
                order=order,
            )
        )

    ir.layouts = [
        layout for layout in ir.layouts if layout.target_id in kept_group_ids
    ]
    ir.style_tokens = [
        token for token in ir.style_tokens
        if len(
            [
                member_id
                for member_id in token.member_ids
                if member_id in kept_entity_ids
            ]
        ) >= 2
    ]
    kept_token_ids = {token.id for token in ir.style_tokens}
    for token in ir.style_tokens:
        token.member_ids = [
            member_id
            for member_id in token.member_ids
            if member_id in kept_entity_ids
        ]
    for element in ir.elements:
        element.style_token_refs = [
            token_id
            for token_id in element.style_token_refs
            if token_id in kept_token_ids
        ]
    from src.design_intent.grouping import normalize_group_source_elements

    normalize_group_source_elements(ir)
    return graph, ir


class IntentDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        max_nodes: int = 128,
        annotation_name: str = "weak_intent.json",
        build_missing_weak_labels: bool = False,
        sample_ids: list[str] | None = None,
    ):
        self.max_nodes = max_nodes
        self.annotation_name = annotation_name
        self.build_missing_weak_labels = build_missing_weak_labels
        root = Path(data_dir)
        selected = set(sample_ids) if sample_ids is not None else None
        self.samples = [
            sample_dir
            for sample_dir in sorted(root.iterdir())
            if sample_dir.is_dir()
            and (selected is None or sample_dir.name in selected)
            and (sample_dir / "screenshot.png").exists()
            and (sample_dir / "page_graph.json").exists()
            and (
                build_missing_weak_labels
                or (sample_dir / annotation_name).exists()
            )
        ]
        if selected is not None:
            found = {sample_dir.name for sample_dir in self.samples}
            missing = sorted(selected - found)
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} 个 manifest 样本缺少所需文件：{missing[:5]}"
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_dir = self.samples[index]
        graph = PageGraph.load(sample_dir / "page_graph.json")
        graph_errors = validate_page_graph(graph)
        if graph_errors:
            raise ValueError(f"{sample_dir}: " + "; ".join(graph_errors))

        annotation_path = sample_dir / self.annotation_name
        if annotation_path.exists():
            ir = DesignIntentIR.load(annotation_path)
        elif self.build_missing_weak_labels:
            ir = build_weak_intent(graph)
        else:
            raise FileNotFoundError(annotation_path)
        graph, ir = truncate_graph_and_ir(graph, ir, self.max_nodes)
        ir_errors = validate_intent_ir(ir, graph)
        if ir_errors:
            raise ValueError(f"{annotation_path}: " + "; ".join(ir_errors))

        n = len(graph.nodes)
        node_mask = torch.zeros(self.max_nodes)
        node_mask[:n] = 1.0
        tag_ids = torch.zeros(self.max_nodes, dtype=torch.long)
        features = torch.zeros(self.max_nodes, STRUCTURED_FEATURE_DIM)
        boxes = torch.zeros(self.max_nodes, 4)
        dom_parents = torch.full((self.max_nodes,), -1, dtype=torch.long)
        node_texts: list[str] = []
        for index, node in enumerate(graph.nodes):
            tag_ids[index] = TAG_TO_IDX.get(node.tag, 0)
            features[index] = torch.tensor(_node_features(node, graph))
            boxes[index] = torch.tensor(_normalized_xyxy(node, graph))
            dom_parents[index] = node.parent_id if node.parent_id is not None else -1
            node_texts.append(_serialize_node(node))

        return {
            # 实验清单以目录名寻址；不能让复制或迁移后的图内 ID 覆盖页面结果。
            "sample_id": sample_dir.name,
            "graph": graph,
            "intent_ir": ir,
            "image": Image.open(sample_dir / "screenshot.png").convert("RGB"),
            "node_texts": node_texts,
            "tag_ids": tag_ids,
            "structured_features": features,
            "boxes": boxes,
            "dom_parents": dom_parents,
            "node_mask": node_mask,
            "alignment_targets": _alignment_targets(boxes, node_mask),
            "targets": build_intent_targets(graph, ir, self.max_nodes),
        }


def intent_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [
        "tag_ids", "structured_features", "boxes", "dom_parents",
        "node_mask", "alignment_targets",
    ]
    target_keys = batch[0]["targets"].keys()
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "graph": [item["graph"] for item in batch],
        "intent_ir": [item["intent_ir"] for item in batch],
        "image": [item["image"] for item in batch],
        "node_texts": [item["node_texts"] for item in batch],
        **{
            key: torch.stack([item[key] for item in batch])
            for key in tensor_keys
        },
        "targets": {
            key: torch.stack([item["targets"][key] for item in batch])
            for key in target_keys
        },
    }
