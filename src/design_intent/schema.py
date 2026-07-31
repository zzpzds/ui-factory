"""Design Intent IR 与 Page Implementation Graph 的稳定 JSON 契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class BBox:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_xyxy(cls, values: list[float] | tuple[float, ...]) -> "BBox":
        x1, y1, x2, y2 = (float(v) for v in values)
        return cls(x=x1, y=y1, width=max(0.0, x2 - x1), height=max(0.0, y2 - y1))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BBox":
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            width=float(data["width"]),
            height=float(data["height"]),
        )

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains(self, other: "BBox", tolerance: float = 1.0) -> bool:
        return (
            self.x <= other.x + tolerance
            and self.y <= other.y + tolerance
            and self.x2 + tolerance >= other.x2
            and self.y2 + tolerance >= other.y2
        )

    @classmethod
    def union(cls, boxes: list["BBox"]) -> "BBox":
        if not boxes:
            return cls(0.0, 0.0, 0.0, 0.0)
        x1 = min(b.x for b in boxes)
        y1 = min(b.y for b in boxes)
        x2 = max(b.x2 for b in boxes)
        y2 = max(b.y2 for b in boxes)
        return cls(x1, y1, x2 - x1, y2 - y1)


@dataclass
class Canvas:
    width: float
    height: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Canvas":
        return cls(width=float(data["width"]), height=float(data["height"]))


@dataclass
class PageNode:
    id: int
    parent_id: int | None
    tag: str
    text: str
    attributes: dict[str, str]
    bbox: BBox
    computed_style: dict[str, Any]
    depth: int = 0
    sibling_index: int = 0
    child_count: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PageNode":
        return cls(
            id=int(data["id"]),
            parent_id=None if data.get("parent_id") is None else int(data["parent_id"]),
            tag=str(data.get("tag", "div")).lower(),
            text=str(data.get("text", "")),
            attributes={str(k): str(v) for k, v in data.get("attributes", {}).items()},
            bbox=BBox.from_dict(data["bbox"]),
            computed_style=dict(data.get("computed_style", {})),
            depth=int(data.get("depth", 0)),
            sibling_index=int(data.get("sibling_index", 0)),
            child_count=int(data.get("child_count", 0)),
        )


@dataclass
class PageGraph:
    schema_version: str
    sample_id: str
    canvas: Canvas
    nodes: list[PageNode]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PageGraph":
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            sample_id=str(data["sample_id"]),
            canvas=Canvas.from_dict(data["canvas"]),
            nodes=[PageNode.from_dict(node) for node in data.get("nodes", [])],
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PageGraph":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


@dataclass
class DesignElement:
    id: str
    source_node_ids: list[int]
    type: str
    bbox: BBox
    name: str
    text: str = ""
    style: dict[str, Any] = field(default_factory=dict)
    style_token_refs: list[str] = field(default_factory=list)
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DesignElement":
        return cls(
            id=str(data["id"]),
            source_node_ids=[int(v) for v in data.get("source_node_ids", [])],
            type=str(data["type"]),
            bbox=BBox.from_dict(data["bbox"]),
            name=str(data.get("name", data["id"])),
            text=str(data.get("text", "")),
            style=dict(data.get("style", {})),
            style_token_refs=[str(v) for v in data.get("style_token_refs", [])],
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class DesignGroup:
    id: str
    source_element_ids: list[str]
    role: str
    bbox: BBox
    name: str
    style: dict[str, Any] = field(default_factory=dict)
    source_node_id: int | None = None
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DesignGroup":
        return cls(
            id=str(data["id"]),
            source_element_ids=[str(v) for v in data.get("source_element_ids", [])],
            role=str(data.get("role", "UNKNOWN")),
            bbox=BBox.from_dict(data["bbox"]),
            name=str(data.get("name", data["id"])),
            style=dict(data.get("style", {})),
            source_node_id=(
                None if data.get("source_node_id") is None else int(data["source_node_id"])
            ),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class LayoutConstraint:
    target_id: str
    mode: str
    gap: float
    padding: list[float]
    primary_align: str = "START"
    cross_align: str = "START"
    horizontal_resize: str = "FIXED"
    vertical_resize: str = "FIXED"
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayoutConstraint":
        return cls(
            target_id=str(data["target_id"]),
            mode=str(data.get("mode", "FREE")),
            gap=float(data.get("gap", 0.0)),
            padding=[float(v) for v in data.get("padding", [0, 0, 0, 0])],
            primary_align=str(data.get("primary_align", "START")),
            cross_align=str(data.get("cross_align", "START")),
            horizontal_resize=str(data.get("horizontal_resize", "FIXED")),
            vertical_resize=str(data.get("vertical_resize", "FIXED")),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class TreeEdge:
    parent_id: str
    child_id: str
    order: int
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TreeEdge":
        return cls(
            parent_id=str(data["parent_id"]),
            child_id=str(data["child_id"]),
            order=int(data.get("order", 0)),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class StyleToken:
    id: str
    kind: str
    value: dict[str, Any]
    member_ids: list[str]
    name: str
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StyleToken":
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            value=dict(data.get("value", {})),
            member_ids=[str(v) for v in data.get("member_ids", [])],
            name=str(data.get("name", data["id"])),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class DesignIntentIR:
    schema_version: str
    canvas: Canvas
    elements: list[DesignElement]
    groups: list[DesignGroup]
    layouts: list[LayoutConstraint]
    tree: list[TreeEdge]
    style_tokens: list[StyleToken]
    provenance: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DesignIntentIR":
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            canvas=Canvas.from_dict(data["canvas"]),
            elements=[DesignElement.from_dict(v) for v in data.get("elements", [])],
            groups=[DesignGroup.from_dict(v) for v in data.get("groups", [])],
            layouts=[LayoutConstraint.from_dict(v) for v in data.get("layouts", [])],
            tree=[TreeEdge.from_dict(v) for v in data.get("tree", [])],
            style_tokens=[StyleToken.from_dict(v) for v in data.get("style_tokens", [])],
            provenance=dict(data.get("provenance", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DesignIntentIR":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
