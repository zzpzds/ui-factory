"""页面图构建与旧数据格式迁移。"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import torch

from .schema import BBox, Canvas, PageGraph, PageNode


_TAG_RE = re.compile(r"^\s*<([a-zA-Z][\w:-]*)")
_ATTR_RE = re.compile(r"""([\w:-]+)=["']([^"']*)["']""")
_TEXT_RE = re.compile(r"^[^>]*>(.*?)</[\w:-]+>\s*$", re.DOTALL)


def parse_serialized_node(value: str) -> tuple[str, str, dict[str, str]]:
    tag_match = _TAG_RE.search(value)
    tag = tag_match.group(1).lower() if tag_match else "div"
    attributes = {key: val for key, val in _ATTR_RE.findall(value)}
    text_match = _TEXT_RE.search(value)
    text = text_match.group(1).strip() if text_match else ""
    return tag, text, attributes


def _tree_features(parents: list[int | None]) -> tuple[list[int], list[int], list[int]]:
    n = len(parents)
    children: dict[int | None, list[int]] = {}
    for node_id, parent_id in enumerate(parents):
        children.setdefault(parent_id, []).append(node_id)

    depth = [0] * n
    for node_id in range(n):
        current = parents[node_id]
        seen = {node_id}
        while current is not None and current >= 0 and current not in seen:
            depth[node_id] += 1
            seen.add(current)
            current = parents[current] if current < n else None

    sibling = [0] * n
    for child_ids in children.values():
        for index, node_id in enumerate(child_ids):
            sibling[node_id] = index

    child_count = [len(children.get(node_id, [])) for node_id in range(n)]
    return depth, sibling, child_count


def page_graph_from_legacy_sample(sample_dir: str | Path) -> PageGraph:
    """读取现有五文件格式，生成 PageGraph v1。"""
    sample_dir = Path(sample_dir)
    with open(sample_dir / "node_texts.json", encoding="utf-8") as f:
        node_texts: list[str] = json.load(f)
    with open(sample_dir / "styles.json", encoding="utf-8") as f:
        styles: list[dict[str, Any]] = json.load(f)
    boxes = torch.load(sample_dir / "nodes.pt", weights_only=True).tolist()
    raw_parents: list[int] = torch.load(
        sample_dir / "parents.pt", weights_only=True
    ).tolist()

    n = min(len(node_texts), len(boxes), len(raw_parents))
    parents: list[int | None] = [
        int(parent) if 0 <= int(parent) < n else None for parent in raw_parents[:n]
    ]
    depth, sibling, child_count = _tree_features(parents)

    nodes: list[PageNode] = []
    for node_id in range(n):
        tag, text, attributes = parse_serialized_node(node_texts[node_id])
        nodes.append(
            PageNode(
                id=node_id,
                parent_id=parents[node_id],
                tag=tag,
                text=text,
                attributes=attributes,
                bbox=BBox.from_xyxy(boxes[node_id]),
                computed_style=styles[node_id] if node_id < len(styles) else {},
                depth=depth[node_id],
                sibling_index=sibling[node_id],
                child_count=child_count[node_id],
            )
        )

    metadata: dict[str, Any] = {
        "source_format": "legacy_render_v1",
        "coordinate_space": "vit_224",
    }
    meta_path = sample_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            metadata["source_meta"] = json.load(f)

    return PageGraph(
        schema_version="1.0",
        sample_id=sample_dir.name,
        canvas=Canvas(width=224.0, height=224.0),
        nodes=nodes,
        metadata=metadata,
    )


def page_graph_from_browser_payload(
    sample_id: str,
    payload: list[dict[str, Any]],
    canvas_width: float,
    canvas_height: float,
    metadata: dict[str, Any] | None = None,
) -> PageGraph:
    """把 Playwright 返回的结构化节点列表转换为 PageGraph。"""
    parents = [
        None if node.get("parent_id") is None or int(node["parent_id"]) < 0
        else int(node["parent_id"])
        for node in payload
    ]
    depth, sibling, child_count = _tree_features(parents)
    nodes = []
    for node_id, raw in enumerate(payload):
        bbox = raw["bbox"]
        nodes.append(
            PageNode(
                id=node_id,
                parent_id=parents[node_id],
                tag=str(raw.get("tag", "div")).lower(),
                text=str(raw.get("text", "")),
                attributes={
                    str(key): str(value)
                    for key, value in raw.get("attributes", {}).items()
                },
                bbox=BBox.from_dict(bbox),
                computed_style=dict(raw.get("computed_style", {})),
                depth=depth[node_id],
                sibling_index=sibling[node_id],
                child_count=child_count[node_id],
            )
        )
    return PageGraph(
        schema_version="1.0",
        sample_id=sample_id,
        canvas=Canvas(width=canvas_width, height=canvas_height),
        nodes=nodes,
        metadata=dict(metadata or {}),
    )
