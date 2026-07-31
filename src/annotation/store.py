"""标注包读取、保存与提交校验。"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

from src.design_intent.grouping import (
    direct_children,
    normalize_group_source_elements,
)
from src.design_intent.schema import DesignIntentIR, PageGraph
from src.design_intent.validation import validate_intent_ir, validate_page_graph


ELEMENT_TYPES = {
    "TEXT", "IMAGE", "ICON", "SHAPE", "INPUT", "BUTTON_VISUAL",
}
GROUP_ROLES = {
    "CONTAINER", "CARD", "NAV", "FORM", "LIST", "LIST_ITEM",
    "TABLE", "TABLE_ROW", "SECTION", "UNKNOWN",
}
LAYOUT_MODES = {"HORIZONTAL", "VERTICAL", "GRID", "FREE"}
PRIMARY_ALIGNMENTS = {"START", "CENTER", "END", "SPACE_BETWEEN"}
CROSS_ALIGNMENTS = {"START", "CENTER", "END", "STRETCH"}
RESIZE_MODES = {"FIXED", "HUG", "STRETCH"}
TOKEN_KINDS = {"COLOR", "TEXT", "RADIUS", "SPACING"}


class AnnotationStore:
    def __init__(self, repo_root: Path, package_dir: Path):
        self.repo_root = repo_root.resolve()
        self.package_dir = package_dir.resolve()
        assignment_path = self.package_dir / "assignment.json"
        self.assignment = json.loads(
            assignment_path.read_text(encoding="utf-8")
        )
        self.samples = {
            item["sample_id"]: item for item in self.assignment["samples"]
        }
        self.annotators = ("annotator_a", "annotator_b")

    def _sample(self, sample_id: str) -> dict[str, Any]:
        if sample_id not in self.samples:
            raise KeyError(f"未知样本：{sample_id}")
        return self.samples[sample_id]

    def _source_path(self, sample_id: str, key: str) -> Path:
        raw = Path(self._sample(sample_id)[key])
        path = raw if raw.is_absolute() else self.repo_root / raw
        path = path.resolve()
        if not path.is_relative_to(self.repo_root):
            raise ValueError("样本路径越出仓库。")
        return path

    def graph(self, sample_id: str) -> PageGraph:
        return PageGraph.load(self._source_path(sample_id, "page_graph"))

    def screenshot_path(self, sample_id: str) -> Path:
        return self._source_path(sample_id, "screenshot")

    def annotation_path(self, annotator: str, sample_id: str) -> Path:
        if annotator not in self.annotators:
            raise KeyError(f"未知标注者：{annotator}")
        self._sample(sample_id)
        return self.package_dir / annotator / f"{sample_id}.json"

    def annotation(self, annotator: str, sample_id: str) -> DesignIntentIR:
        return DesignIntentIR.load(
            self.annotation_path(annotator, sample_id)
        )

    def progress(self) -> dict[str, dict[str, int]]:
        result = {}
        for annotator in self.annotators:
            statuses = Counter(
                str(
                    self.annotation(annotator, sample_id).provenance.get(
                        "status", "draft"
                    )
                )
                for sample_id in self.samples
            )
            result[annotator] = {
                "complete": statuses["complete"],
                "draft": len(self.samples) - statuses["complete"],
                "total": len(self.samples),
            }
        return result

    def assignment_payload(self) -> dict[str, Any]:
        payload = dict(self.assignment)
        payload["annotators"] = list(self.annotators)
        payload["progress"] = self.progress()
        payload["sample_status"] = {
            annotator: {
                sample_id: str(
                    self.annotation(annotator, sample_id).provenance.get(
                        "status", "draft"
                    )
                )
                for sample_id in self.samples
            }
            for annotator in self.annotators
        }
        return payload

    @staticmethod
    def _bbox_errors(ir: DesignIntentIR) -> list[str]:
        errors = []
        for entity in [*ir.elements, *ir.groups]:
            bbox = entity.bbox
            if bbox.width <= 0 or bbox.height <= 0:
                errors.append(f"{entity.id} 的 bbox 必须具有正面积")
            if (
                bbox.x < 0
                or bbox.y < 0
                or bbox.x2 > ir.canvas.width + 1e-3
                or bbox.y2 > ir.canvas.height + 1e-3
            ):
                errors.append(f"{entity.id} 的 bbox 超出画布")
        return errors

    @staticmethod
    def _submission_errors(
        ir: DesignIntentIR, graph: PageGraph
    ) -> list[str]:
        errors = validate_page_graph(graph) + validate_intent_ir(ir, graph)
        errors.extend(AnnotationStore._bbox_errors(ir))

        if not ir.elements:
            errors.append("完整标注至少需要一个原子元素")
        for element in ir.elements:
            if element.type not in ELEMENT_TYPES:
                errors.append(f"元素 {element.id} 类型非法：{element.type}")
        for group in ir.groups:
            children = direct_children(ir, group.id)
            if not children:
                errors.append(f"组 {group.id} 至少需要一个直接设计子实体")
            if group.role not in GROUP_ROLES:
                errors.append(f"组 {group.id} 角色非法：{group.role}")

        layout_counts = Counter(layout.target_id for layout in ir.layouts)
        for group in ir.groups:
            if layout_counts[group.id] != 1:
                errors.append(f"组 {group.id} 必须且只能有一个布局约束")
        for layout in ir.layouts:
            if layout.mode not in LAYOUT_MODES:
                errors.append(f"布局 {layout.target_id} mode 非法")
            if layout.primary_align not in PRIMARY_ALIGNMENTS:
                errors.append(f"布局 {layout.target_id} 主轴对齐非法")
            if layout.cross_align not in CROSS_ALIGNMENTS:
                errors.append(f"布局 {layout.target_id} 交叉轴对齐非法")
            if (
                layout.horizontal_resize not in RESIZE_MODES
                or layout.vertical_resize not in RESIZE_MODES
            ):
                errors.append(f"布局 {layout.target_id} resize 非法")

        source_counts = Counter(
            source_id
            for element in ir.elements
            for source_id in element.source_node_ids
        )
        repeated_sources = sorted(
            source_id
            for source_id, count in source_counts.items()
            if count > 1
        )
        if repeated_sources:
            errors.append(
                f"源节点不能属于多个原子元素：{repeated_sources[:10]}"
            )

        for token in ir.style_tokens:
            if token.kind not in TOKEN_KINDS:
                errors.append(f"token {token.id} kind 非法：{token.kind}")
            if len(set(token.member_ids)) < 2:
                errors.append(f"token {token.id} 至少需要两个成员")
        return list(dict.fromkeys(errors))

    @staticmethod
    def _normalize_token_references(ir: DesignIntentIR) -> None:
        token_members = {
            token.id: set(token.member_ids) for token in ir.style_tokens
        }
        for element in ir.elements:
            element.style_token_refs = sorted(
                token_id
                for token_id, members in token_members.items()
                if element.id in members
            )

    def save(
        self,
        annotator: str,
        sample_id: str,
        payload: dict[str, Any],
        submit: bool = False,
    ) -> dict[str, Any]:
        graph = self.graph(sample_id)
        ir = DesignIntentIR.from_dict(payload)
        if (
            ir.canvas.width != graph.canvas.width
            or ir.canvas.height != graph.canvas.height
        ):
            raise ValueError("标注画布尺寸与 PageGraph 不一致。")

        normalize_group_source_elements(ir)
        self._normalize_token_references(ir)
        ir.provenance = {
            **ir.provenance,
            "source_sample_id": sample_id,
            "label_source": "human_annotation",
            "annotator": annotator,
            "weak_labels_viewed": False,
            "status": "draft",
        }
        errors = self._submission_errors(ir, graph)
        if submit and not errors:
            ir.provenance["status"] = "complete"

        path = self.annotation_path(annotator, sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(ir.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return {
            "annotation": ir.to_dict(),
            "errors": errors,
            "submitted": submit and not errors,
        }
