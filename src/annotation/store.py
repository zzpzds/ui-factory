"""标注包读取、保存与提交校验。"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
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
        configured_annotators = self.assignment.get(
            "annotators", ["annotator_a", "annotator_b"]
        )
        if not configured_annotators:
            raise ValueError("标注包至少需要配置一名标注者。")
        self.annotators = tuple(str(value) for value in configured_annotators)
        self.locked_sample_ids = {
            str(value) for value in self.assignment.get("locked_sample_ids", [])
        }
        unknown_locked = self.locked_sample_ids - set(self.samples)
        if unknown_locked:
            raise ValueError(f"冻结列表包含未知样本：{sorted(unknown_locked)}")

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

    def _token_review_errors(
        self, ir: DesignIntentIR, sample_id: str
    ) -> list[str]:
        if (
            self.assignment.get("token_annotation_mode") != "candidate_review"
            or sample_id in self.locked_sample_ids
        ):
            return []
        review = ir.provenance.get("token_review", {})
        if (
            review.get("status") != "reviewed"
            or not isinstance(review.get("reviewed_candidate_keys"), list)
        ):
            return ["请先完成 Token 样式候选检查"]
        return []

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _assisted_review_errors(
        self, ir: DesignIntentIR, sample_id: str
    ) -> list[str]:
        workflow = self.assignment.get("annotation_workflow", {})
        mode = ir.provenance.get("ai_assistance_mode")
        if not mode or sample_id in self.locked_sample_ids:
            return []
        errors = []
        if (
            workflow.get("require_human_review_confirmation")
            and ir.provenance.get("human_review_confirmed") is not True
        ):
            errors.append("请确认设计师已完成逐项复核")
        if mode not in {"preannotation", "blind_control"}:
            errors.append(f"未知 AI 辅助标注模式：{mode}")
            return errors
        expected_visible = mode == "preannotation"
        if ir.provenance.get("ai_preannotation_visible") is not expected_visible:
            errors.append("AI 初稿可见性与标注条件不一致")
        raw_path = ir.provenance.get("ai_preannotation_path")
        expected_hash = ir.provenance.get("ai_preannotation_sha256")
        if not raw_path or not expected_hash:
            errors.append("缺少 AI 初稿来源或哈希")
            return errors
        path = (self.repo_root / str(raw_path)).resolve()
        if not path.is_relative_to(self.repo_root) or not path.exists():
            errors.append("AI 初稿来源文件不存在或越出仓库")
        elif self._sha256(path) != expected_hash:
            errors.append("AI 初稿来源哈希已变化")
        return errors

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
        self.annotation_path(annotator, sample_id)
        if sample_id in self.locked_sample_ids:
            raise ValueError("该样本已冻结为金标准，不能继续修改")
        graph = self.graph(sample_id)
        ir = DesignIntentIR.from_dict(payload)
        if (
            ir.canvas.width != graph.canvas.width
            or ir.canvas.height != graph.canvas.height
        ):
            raise ValueError("标注画布尺寸与 PageGraph 不一致。")

        normalize_group_source_elements(ir)
        self._normalize_token_references(ir)
        assistance_mode = ir.provenance.get("ai_assistance_mode")
        label_source = (
            "human_corrected_ai_preannotation"
            if assistance_mode == "preannotation"
            else "human_annotation"
        )
        ir.provenance = {
            **ir.provenance,
            "source_sample_id": sample_id,
            "label_source": label_source,
            "annotator": annotator,
            "weak_labels_viewed": assistance_mode == "preannotation",
            "ai_assistance_disclosed": bool(assistance_mode),
            "status": "draft",
        }
        errors = self._submission_errors(ir, graph)
        errors.extend(self._token_review_errors(ir, sample_id))
        errors.extend(self._assisted_review_errors(ir, sample_id))
        if submit and not errors:
            ir.provenance["status"] = "complete"
            if assistance_mode and not ir.provenance.get("human_reviewed_at"):
                ir.provenance["human_reviewed_at"] = (
                    datetime.now().astimezone().isoformat()
                )

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
