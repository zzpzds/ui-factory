"""Design Intent 数据集：PageGraph + Intent IR → 多任务监督张量。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path, PurePosixPath
import re
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
REFERENCE_TIERS = ("human_gold", "ai_silver")
REFERENCE_LABEL_SOURCES = {
    "human_gold": "human_ai_adjudicated_gold",
    "ai_silver": "ai_multiview_silver",
}
REFERENCE_ANNOTATION_CONDITIONS = {
    "human_gold": "existing_human_adjudicated_gold",
    "ai_silver": "ai_multiview_silver",
}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORMAL_REFERENCE_PACKAGE = (
    _REPO_ROOT / "data/annotations/intent_gold_v1"
).resolve()
_FORMAL_TEST_UNSEAL_PATH = (
    _REPO_ROOT / "outputs/intent-reference-v1/test_unseal.json"
)
_TEST_UNSEAL_FIELDS = {
    "unsealed_at",
    "git_commit",
    "assignment_hash",
    "checkpoint_hash",
}
_SHA256_TAG_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_SHA256_HEX_PATTERN = re.compile(r"[0-9a-f]{64}")


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
    *,
    reference_tier: str | None = None,
) -> dict[str, torch.Tensor]:
    if reference_tier not in {None, *REFERENCE_TIERS}:
        raise ValueError(f"未知 reference tier：{reference_tier}")
    n = min(len(graph.nodes), max_nodes)
    action = torch.full((max_nodes,), ACTION_TO_IDX["DROP"], dtype=torch.long)
    action_confidence = torch.zeros(max_nodes)
    action_confidence[:n] = 0.5
    action_mask = torch.zeros(max_nodes)
    if reference_tier != "human_gold":
        action_mask[:n] = 1.0
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
        action_mask[source_id] = 1.0
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
        action_mask[source_id] = 1.0
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
    tree_mask = torch.zeros(max_nodes)
    parent_by_child = {edge.child_id: edge.parent_id for edge in ir.tree}
    for entity_id, source_id in entity_source.items():
        parent_id = parent_by_child.get(entity_id)
        if parent_id == "page_root":
            parent_target[source_id] = max_nodes
            tree_mask[source_id] = 1.0
        elif parent_id in entity_source:
            parent_target[source_id] = entity_source[parent_id]
            tree_mask[source_id] = 1.0

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
            if (
                direct_group[left_id] != "page_root"
                and direct_group[left_id] == direct_group[right_id]
            ):
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

    if reference_tier == "human_gold":
        token_pair_mask.zero_()

    return {
        "action": action,
        "action_confidence": action_confidence,
        "action_mask": action_mask,
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
        "tree_mask": tree_mask,
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


def _build_intent_sample(
    *,
    sample_id: str,
    screenshot_path: Path,
    graph_path: Path,
    annotation_path: Path,
    max_nodes: int,
    build_missing_weak_labels: bool = False,
    reference_tier: str | None = None,
) -> dict[str, Any]:
    """从显式路径构造模型样本，供目录数据和 Reference 数据共用。"""
    graph = PageGraph.load(graph_path)
    graph_errors = validate_page_graph(graph)
    if graph_errors:
        raise ValueError(f"{graph_path}: " + "; ".join(graph_errors))

    if annotation_path.exists():
        ir = DesignIntentIR.load(annotation_path)
    elif build_missing_weak_labels:
        ir = build_weak_intent(graph)
    else:
        raise FileNotFoundError(annotation_path)
    graph, ir = truncate_graph_and_ir(graph, ir, max_nodes)
    ir_errors = validate_intent_ir(ir, graph)
    if ir_errors:
        raise ValueError(f"{annotation_path}: " + "; ".join(ir_errors))

    n = len(graph.nodes)
    node_mask = torch.zeros(max_nodes)
    node_mask[:n] = 1.0
    tag_ids = torch.zeros(max_nodes, dtype=torch.long)
    features = torch.zeros(max_nodes, STRUCTURED_FEATURE_DIM)
    boxes = torch.zeros(max_nodes, 4)
    dom_parents = torch.full((max_nodes,), -1, dtype=torch.long)
    node_texts: list[str] = []
    for index, node in enumerate(graph.nodes):
        tag_ids[index] = TAG_TO_IDX.get(node.tag, 0)
        features[index] = torch.tensor(_node_features(node, graph))
        boxes[index] = torch.tensor(_normalized_xyxy(node, graph))
        dom_parents[index] = node.parent_id if node.parent_id is not None else -1
        node_texts.append(_serialize_node(node))

    with Image.open(screenshot_path) as source_image:
        image = source_image.convert("RGB")
    return {
        "sample_id": sample_id,
        "graph": graph,
        "intent_ir": ir,
        "image": image,
        "node_texts": node_texts,
        "tag_ids": tag_ids,
        "structured_features": features,
        "boxes": boxes,
        "dom_parents": dom_parents,
        "node_mask": node_mask,
        "alignment_targets": _alignment_targets(boxes, node_mask),
        "targets": build_intent_targets(
            graph,
            ir,
            max_nodes,
            reference_tier=reference_tier,
        ),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 顶层必须是 JSON object")
    return payload


def _load_json_object_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    """从同一字节快照解析 JSON 并计算审计哈希。"""
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 顶层必须是 JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256_hex(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256_HEX_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(f"{field} 必须是 64 位小写 SHA-256。")
    return value


def validate_test_unseal(
    payload: Any,
    *,
    assignment_hash: str | None = None,
) -> dict[str, Any]:
    """校验正式 test 解封审计；不接受仅有一个 JSON 文件的弱门禁。"""
    if not isinstance(payload, dict) or set(payload) != _TEST_UNSEAL_FIELDS:
        raise ValueError("test unseal 字段不完整或包含未知字段。")
    timestamp = payload["unsealed_at"]
    if not isinstance(timestamp, str):
        raise ValueError("test unseal unsealed_at 必须是带时区时间。")
    try:
        parsed_time = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise ValueError("test unseal unsealed_at 不是 ISO 时间。") from error
    if parsed_time.utcoffset() is None:
        raise ValueError("test unseal unsealed_at 必须包含时区。")
    if not isinstance(payload["git_commit"], str) or not payload["git_commit"]:
        raise ValueError("test unseal git_commit 不能为空。")
    for field in ("assignment_hash", "checkpoint_hash"):
        value = payload[field]
        if (
            not isinstance(value, str)
            or _SHA256_TAG_PATTERN.fullmatch(value) is None
        ):
            raise ValueError(f"test unseal {field} 必须是 sha256:<64hex>。")
    if assignment_hash is not None and payload["assignment_hash"] != assignment_hash:
        raise ValueError("test unseal assignment_hash 与当前 Reference 包不一致。")
    return dict(payload)


def _records_by_id(
    records: Any,
    *,
    source: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError(f"{source}.samples 必须是列表")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or "sample_id" not in record:
            raise ValueError(f"{source}.samples 包含无效记录")
        sample_id = str(record["sample_id"])
        if not sample_id or Path(sample_id).name != sample_id:
            raise ValueError(f"{source} 包含非法 sample ID：{sample_id!r}")
        if sample_id in result:
            raise ValueError(f"{source} 包含重复 sample ID：{sample_id}")
        result[sample_id] = record
    return result


def _split_membership(payload: Any, *, source: str) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError(f"{source}.splits 必须是 object")
    allowed_splits = {"train", "validation", "test"}
    unknown = set(payload) - allowed_splits
    if unknown:
        raise ValueError(f"{source}.splits 包含未知 split：{sorted(unknown)}")
    membership: dict[str, str] = {}
    for split, raw_ids in payload.items():
        if not isinstance(raw_ids, list):
            raise ValueError(f"{source}.splits.{split} 必须是列表")
        for raw_id in raw_ids:
            sample_id = str(raw_id)
            if sample_id in membership:
                raise ValueError(
                    f"{source} 样本 {sample_id} 同时出现在 "
                    f"{membership[sample_id]} 和 {split}"
                )
            membership[sample_id] = split
    return membership


def _id_set(raw_ids: Any, *, source: str) -> set[str]:
    if not isinstance(raw_ids, list):
        raise ValueError(f"{source} 必须是列表")
    values = [str(value) for value in raw_ids]
    if len(values) != len(set(values)):
        raise ValueError(f"{source} 包含重复 sample ID")
    return set(values)


def _manifest_path(
    raw_path: Any,
    *,
    source: str,
    strict_repo_relative: bool = False,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{source} 必须是非空路径")
    if strict_repo_relative:
        pure_path = PurePosixPath(raw_path)
        if (
            pure_path.is_absolute()
            or "\\" in raw_path
            or ".." in pure_path.parts
            or pure_path.as_posix() != raw_path
        ):
            raise ValueError(
                f"{source} 必须使用规范仓库相对 POSIX 路径，实际 {raw_path!r}"
            )
        return (_REPO_ROOT / Path(*pure_path.parts)).resolve()
    path = Path(raw_path)
    return (path if path.is_absolute() else _REPO_ROOT / path).resolve()


def _expect_equal(
    sample_id: str,
    field: str,
    expected: Any,
    actual: Any,
) -> None:
    if actual != expected:
        raise ValueError(
            f"样本 {sample_id} 的 {field} 不一致："
            f"期望 {expected!r}，实际 {actual!r}"
        )


def _expect_id_sets_equal(
    field: str,
    expected: set[str],
    actual: set[str],
) -> None:
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    sample_id = (missing or unexpected or ["<manifest>"])[0]
    raise ValueError(
        f"样本 {sample_id} 的 {field} 不一致："
        f"期望集合缺失项为 []、多余项为 []，"
        f"实际缺失 {missing[:5]}、多余 {unexpected[:5]}"
    )


@dataclass(frozen=True)
class _ReferenceSample:
    sample_id: str
    reference_tier: str
    label_source: str
    screenshot_path: Path
    graph_path: Path
    annotation_path: Path


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
        return _build_intent_sample(
            # 实验清单以目录名寻址；不能让复制后的图内 ID 覆盖页面结果。
            sample_id=sample_dir.name,
            screenshot_path=sample_dir / "screenshot.png",
            graph_path=sample_dir / "page_graph.json",
            annotation_path=sample_dir / self.annotation_name,
            max_nodes=self.max_nodes,
            build_missing_weak_labels=self.build_missing_weak_labels,
        )


class ReferenceIntentDataset(Dataset):
    """直接消费只读 Reference 包，并在初始化时完成来源与切分校验。"""

    def __init__(
        self,
        data_dir: str,
        package_dir: str,
        split: str,
        max_nodes: int,
        tiers: list[str] | None = None,
        source_split_manifest: str | None = None,
        verify_hashes: bool = True,
        *,
        _metadata_only: bool = False,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"未知 split：{split}")
        if max_nodes <= 0:
            raise ValueError("max_nodes 必须大于 0")
        requested_tiers = list(REFERENCE_TIERS) if tiers is None else list(tiers)
        if not requested_tiers:
            raise ValueError("tiers 不能为空")
        if len(requested_tiers) != len(set(requested_tiers)):
            raise ValueError("tiers 不能重复")
        unknown_tiers = set(requested_tiers) - set(REFERENCE_TIERS)
        if unknown_tiers:
            raise ValueError(f"未知 reference tier：{sorted(unknown_tiers)}")

        self.data_dir = Path(data_dir).resolve()
        self.package_dir = Path(package_dir).resolve()
        self.max_nodes = max_nodes
        self.split = split
        self.verify_hashes = verify_hashes
        if not isinstance(_metadata_only, bool):
            raise TypeError("_metadata_only 必须是布尔值")
        self._metadata_only = _metadata_only
        if not self.data_dir.is_dir():
            raise FileNotFoundError(self.data_dir)
        if not self.package_dir.is_dir():
            raise FileNotFoundError(self.package_dir)
        is_formal_test = (
            split == "test" and self.package_dir == _FORMAL_REFERENCE_PACKAGE
        )
        if is_formal_test and not verify_hashes:
            raise ValueError("正式 Reference test 必须启用 verify_hashes。")
        strict_repo_paths = (
            self.data_dir.is_relative_to(_REPO_ROOT)
            and self.package_dir.is_relative_to(_REPO_ROOT)
        )

        manifest_paths = {
            "assignment": self.package_dir / "assignment.json",
            "selection_manifest": self.package_dir / "selection_manifest.json",
            "ai_reference_manifest": self.package_dir / "ai_reference_manifest.json",
        }
        assignment, assignment_manifest_hash = _load_json_object_snapshot(
            manifest_paths["assignment"]
        )
        selection, selection_manifest_hash = _load_json_object_snapshot(
            manifest_paths["selection_manifest"]
        )
        reference_manifest, reference_manifest_hash = _load_json_object_snapshot(
            manifest_paths["ai_reference_manifest"]
        )
        manifest_hashes = {
            "assignment": assignment_manifest_hash,
            "selection_manifest": selection_manifest_hash,
            "ai_reference_manifest": reference_manifest_hash,
        }
        assignment_records = _records_by_id(
            assignment.get("samples"), source="assignment"
        )
        selection_records = _records_by_id(
            selection.get("samples"), source="selection_manifest"
        )
        _expect_id_sets_equal(
            "assignment/selection_manifest sample ID 集合",
            set(assignment_records),
            set(selection_records),
        )

        assignment_splits: dict[str, str] = {}
        for sample_id, record in assignment_records.items():
            item_split = str(record.get("split", ""))
            if item_split not in {"train", "validation", "test"}:
                raise ValueError(
                    f"样本 {sample_id} 的 assignment.split 非法：{item_split!r}"
                )
            assignment_splits[sample_id] = item_split
            _expect_equal(
                sample_id,
                "selection_manifest.samples[].split",
                item_split,
                str(selection_records[sample_id].get("split", "")),
            )

        selection_splits = _split_membership(
            selection.get("splits"), source="selection_manifest"
        )
        _expect_id_sets_equal(
            "assignment/selection_manifest.splits sample ID 集合",
            set(assignment_splits),
            set(selection_splits),
        )
        for sample_id, expected_split in assignment_splits.items():
            _expect_equal(
                sample_id,
                "selection_manifest.splits split",
                expected_split,
                selection_splits.get(sample_id),
            )

        human_ids = _id_set(
            assignment.get("human_gold_sample_ids"),
            source="assignment.human_gold_sample_ids",
        )
        silver_ids = _id_set(
            assignment.get("ai_silver_sample_ids"),
            source="assignment.ai_silver_sample_ids",
        )
        if human_ids & silver_ids:
            sample_id = sorted(human_ids & silver_ids)[0]
            raise ValueError(
                f"样本 {sample_id} 的 assignment reference_tier 不一致："
                "期望只属于一个 tier，实际同时属于 human_gold 和 ai_silver"
            )
        _expect_id_sets_equal(
            "assignment tier sample ID 集合",
            set(assignment_records),
            human_ids | silver_ids,
        )
        all_split_tier_counts = {
            split_name: {tier: 0 for tier in REFERENCE_TIERS}
            for split_name in ("train", "validation", "test")
        }
        for sample_id, item_split in assignment_splits.items():
            tier = "human_gold" if sample_id in human_ids else "ai_silver"
            all_split_tier_counts[item_split][tier] += 1

        silver_manifest_records = _records_by_id(
            reference_manifest.get("samples"), source="ai_reference_manifest"
        )
        human_manifest_records = _records_by_id(
            reference_manifest.get("human_gold_files"),
            source="ai_reference_manifest.human_gold_files",
        )
        _expect_id_sets_equal(
            "ai_reference_manifest Silver sample ID 集合",
            silver_ids,
            set(silver_manifest_records),
        )
        _expect_id_sets_equal(
            "ai_reference_manifest Human sample ID 集合",
            human_ids,
            set(human_manifest_records),
        )
        _expect_equal(
            "<manifest>",
            "ai_reference_manifest.ai_silver_samples",
            len(silver_ids),
            reference_manifest.get("ai_silver_samples"),
        )
        _expect_equal(
            "<manifest>",
            "ai_reference_manifest.human_gold_samples",
            len(human_ids),
            reference_manifest.get("human_gold_samples"),
        )
        _expect_equal(
            "<manifest>",
            "ai_reference_manifest.human_gold_token_labels_available",
            False,
            reference_manifest.get("human_gold_token_labels_available"),
        )
        _expect_equal(
            "<manifest>",
            "ai_reference_manifest.human_gold_token_metrics_reportable",
            False,
            reference_manifest.get("human_gold_token_metrics_reportable"),
        )
        for sample_id, record in human_manifest_records.items():
            _expect_equal(
                sample_id,
                (
                    "ai_reference_manifest.human_gold_files[]."
                    "token_labels_available"
                ),
                False,
                record.get("token_labels_available"),
            )

        token_policy = reference_manifest.get("token_policy")
        if not isinstance(token_policy, dict):
            raise ValueError("ai_reference_manifest.token_policy 必须是 object")
        _expect_equal(
            "<manifest>",
            "ai_reference_manifest.token_policy.generator",
            "codex_compact_style_tokens_v1",
            token_policy.get("generator"),
        )
        _expect_equal(
            "<manifest>",
            "ai_reference_manifest.token_policy.maximum_tokens_per_page",
            12,
            token_policy.get("maximum_tokens_per_page"),
        )
        _expect_equal(
            "<manifest>",
            "ai_reference_manifest.token_policy.evaluation_scope",
            "ai_silver_proxy_consistency_only",
            token_policy.get("evaluation_scope"),
        )

        if source_split_manifest is not None:
            source_split_path = Path(source_split_manifest).resolve()
            source_split_payload, source_split_hash = _load_json_object_snapshot(
                source_split_path
            )
            source_splits = _split_membership(
                source_split_payload.get("splits"),
                source="source_split_manifest",
            )
            for sample_id, expected_split in assignment_splits.items():
                _expect_equal(
                    sample_id,
                    "source_split_manifest split",
                    expected_split,
                    source_splits.get(sample_id),
                )
            manifest_paths["source_split"] = source_split_path
            manifest_hashes["source_split"] = source_split_hash

        reference_dir = (self.package_dir / "reference").resolve()
        if not reference_dir.is_relative_to(self.package_dir):
            raise ValueError("Reference 标注目录越出 package_dir")
        configured_reference_dirs = [
            assignment.get("annotation_workflow", {}).get(
                "canonical_reference_dir"
            ),
            reference_manifest.get("canonical_reference_dir"),
        ]
        for raw_path in configured_reference_dirs:
            if raw_path is not None:
                configured = _manifest_path(
                    raw_path,
                    source="canonical_reference_dir",
                    strict_repo_relative=strict_repo_paths,
                )
                if configured != reference_dir:
                    raise ValueError(
                        "manifest canonical_reference_dir 与规范目录不一致"
                    )

        samples_by_id: dict[str, _ReferenceSample] = {}
        for sample_id in sorted(assignment_records):
            record = assignment_records[sample_id]
            tier = "human_gold" if sample_id in human_ids else "ai_silver"
            expected_label_source = REFERENCE_LABEL_SOURCES[tier]
            _expect_equal(
                sample_id,
                "assignment.annotation_condition",
                REFERENCE_ANNOTATION_CONDITIONS[tier],
                record.get("annotation_condition"),
            )

            screenshot_path = (self.data_dir / sample_id / "screenshot.png").resolve()
            graph_path = (self.data_dir / sample_id / "page_graph.json").resolve()
            annotation_path = (reference_dir / f"{sample_id}.json").resolve()
            if not screenshot_path.is_relative_to(self.data_dir):
                raise ValueError(f"样本 {sample_id} 的截图路径越出 data_dir")
            if not graph_path.is_relative_to(self.data_dir):
                raise ValueError(f"样本 {sample_id} 的 PageGraph 路径越出 data_dir")
            if not annotation_path.is_relative_to(reference_dir):
                raise ValueError(f"样本 {sample_id} 的标注路径越出 reference 目录")
            _expect_equal(
                sample_id,
                "assignment.screenshot",
                screenshot_path,
                _manifest_path(
                    record.get("screenshot"),
                    source=f"assignment.samples[{sample_id}].screenshot",
                    strict_repo_relative=strict_repo_paths,
                ),
            )
            _expect_equal(
                sample_id,
                "assignment.page_graph",
                graph_path,
                _manifest_path(
                    record.get("page_graph"),
                    source=f"assignment.samples[{sample_id}].page_graph",
                    strict_repo_relative=strict_repo_paths,
                ),
            )

            reference_record = (
                human_manifest_records[sample_id]
                if tier == "human_gold"
                else silver_manifest_records[sample_id]
            )
            asset_hashes = selection_records[sample_id].get("asset_sha256")
            if not isinstance(asset_hashes, dict):
                raise ValueError(
                    f"样本 {sample_id} 缺少 selection asset_sha256"
                )
            for asset_name in ("screenshot", "page_graph"):
                _require_sha256_hex(
                    asset_hashes.get(asset_name),
                    (
                        f"样本 {sample_id} 的 selection_manifest."
                        f"asset_sha256.{asset_name}"
                    ),
                )
            _require_sha256_hex(
                reference_record.get("reference_sha256"),
                (
                    f"样本 {sample_id} 的 ai_reference_manifest."
                    "reference_sha256"
                ),
            )
            _expect_equal(
                sample_id,
                "ai_reference_manifest.reference_path",
                annotation_path,
                _manifest_path(
                    reference_record.get("reference_path"),
                    source=f"ai_reference_manifest[{sample_id}].reference_path",
                    strict_repo_relative=strict_repo_paths,
                ),
            )

            samples_by_id[sample_id] = _ReferenceSample(
                sample_id=sample_id,
                reference_tier=tier,
                label_source=expected_label_source,
                screenshot_path=screenshot_path,
                graph_path=graph_path,
                annotation_path=annotation_path,
            )

        requested_tier_set = set(requested_tiers)
        active_ids = [
            sample_id
            for sample_id in sorted(samples_by_id)
            if assignment_splits[sample_id] == split
            and samples_by_id[sample_id].reference_tier in requested_tier_set
        ]
        if not active_ids:
            raise ValueError(
                f"split={split}、tiers={requested_tiers} 没有可用 Reference 样本"
            )
        self.samples = [samples_by_id[sample_id] for sample_id in active_ids]
        self.reference_tiers = [sample.reference_tier for sample in self.samples]
        counts = Counter(self.reference_tiers)
        self.tier_counts = {
            tier: counts[tier] for tier in REFERENCE_TIERS if counts[tier]
        }
        self.all_split_tier_counts = all_split_tier_counts
        self.manifest_hashes = dict(manifest_hashes)

        for sample in self.samples:
            for path, field in [
                (sample.screenshot_path, "screenshot"),
                (sample.graph_path, "page_graph"),
                (sample.annotation_path, "reference"),
            ]:
                if not path.is_file():
                    raise FileNotFoundError(
                        f"样本 {sample.sample_id} 缺少 {field}：{path}"
                    )

        # 正式 test 解封前只允许校验清单语义、路径和文件存在性。
        if self._metadata_only:
            return

        if is_formal_test:
            if not _FORMAL_TEST_UNSEAL_PATH.is_file():
                raise PermissionError(
                    "正式 Reference test 尚未解封；请通过 "
                    "scripts/evaluate_intent.py 完成预检并创建固定 unseal。"
                )
            validate_test_unseal(
                _load_json_object(_FORMAL_TEST_UNSEAL_PATH),
                assignment_hash=(
                    f"sha256:{self.manifest_hashes['assignment']}"
                ),
            )

        # test 解封边界：只有 active split/tier 可以接触实际资源与 IR。
        for sample in self.samples:
            sample_id = sample.sample_id
            tier = sample.reference_tier
            reference_record = (
                human_manifest_records[sample_id]
                if tier == "human_gold"
                else silver_manifest_records[sample_id]
            )
            annotation_payload = _load_json_object(sample.annotation_path)
            provenance = annotation_payload.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"样本 {sample_id} 缺少 provenance")
            _expect_equal(
                sample_id,
                "provenance.source_sample_id",
                sample_id,
                str(provenance.get("source_sample_id", "")),
            )
            _expect_equal(
                sample_id,
                "provenance.reference_tier",
                tier,
                provenance.get("reference_tier"),
            )
            _expect_equal(
                sample_id,
                "provenance.label_source",
                sample.label_source,
                provenance.get("label_source"),
            )
            _expect_equal(
                sample_id,
                "provenance.status",
                "complete",
                provenance.get("status"),
            )
            if tier == "human_gold":
                _expect_equal(
                    sample_id,
                    "provenance.token_labels_available",
                    False,
                    provenance.get("token_labels_available"),
                )

            if verify_hashes:
                asset_hashes = selection_records[sample_id].get("asset_sha256")
                if not isinstance(asset_hashes, dict):
                    raise ValueError(
                        f"样本 {sample_id} 缺少 selection asset_sha256"
                    )
                _expect_equal(
                    sample_id,
                    "screenshot SHA-256",
                    asset_hashes.get("screenshot"),
                    _sha256(sample.screenshot_path),
                )
                _expect_equal(
                    sample_id,
                    "page_graph SHA-256",
                    asset_hashes.get("page_graph"),
                    _sha256(sample.graph_path),
                )
                _expect_equal(
                    sample_id,
                    "reference SHA-256",
                    reference_record.get("reference_sha256"),
                    _sha256(sample.annotation_path),
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._metadata_only:
            raise RuntimeError("仅元数据 ReferenceIntentDataset 不能读取样本")
        sample = self.samples[index]
        item = _build_intent_sample(
            sample_id=sample.sample_id,
            screenshot_path=sample.screenshot_path,
            graph_path=sample.graph_path,
            annotation_path=sample.annotation_path,
            max_nodes=self.max_nodes,
            reference_tier=sample.reference_tier,
        )
        is_human = sample.reference_tier == "human_gold"
        item.update(
            {
                "reference_tier": sample.reference_tier,
                "label_source": sample.label_source,
                "token_supervision_available": not is_human,
                "token_supervision_scope": (
                    "not_collected" if is_human else "ai_computed_style_proxy"
                ),
                "negative_action_supervision_available": not is_human,
            }
        )
        return item


def intent_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch 不能为空")
    tensor_keys = [
        "tag_ids", "structured_features", "boxes", "dom_parents",
        "node_mask", "alignment_targets",
    ]
    target_keys = batch[0]["targets"].keys()
    result = {
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
    optional_list_keys = [
        "reference_tier",
        "label_source",
        "token_supervision_scope",
    ]
    optional_bool_keys = [
        "token_supervision_available",
        "negative_action_supervision_available",
    ]
    for key in [*optional_list_keys, *optional_bool_keys]:
        presence = [key in item for item in batch]
        if any(presence) and not all(presence):
            raise ValueError(f"batch 中的可选字段 {key} 不完整")
    for key in optional_list_keys:
        if key in batch[0]:
            result[key] = [item[key] for item in batch]
    for key in optional_bool_keys:
        if key in batch[0]:
            result[key] = torch.tensor(
                [bool(item[key]) for item in batch], dtype=torch.bool
            )
    return result
