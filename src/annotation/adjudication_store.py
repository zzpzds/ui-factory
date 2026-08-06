"""人机差异仲裁记录与金标准草稿的读写和审核门槛。"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.annotation.store import AnnotationStore
from src.design_intent.grouping import normalize_group_source_elements
from src.design_intent.schema import DesignIntentIR, PageGraph


class AdjudicationStore:
    def __init__(self, repo_root: Path, package_dir: Path):
        self.repo_root = repo_root.resolve()
        self.package_dir = package_dir.resolve()
        self.assignment = json.loads(
            (self.package_dir / "assignment.json").read_text(encoding="utf-8")
        )
        self.samples = {
            str(item["sample_id"]): item for item in self.assignment["samples"]
        }

    def _sample(self, sample_id: str) -> dict[str, Any]:
        if sample_id not in self.samples:
            raise KeyError(f"未知样本：{sample_id}")
        return self.samples[sample_id]

    def _path(self, directory: str, sample_id: str) -> Path:
        self._sample(sample_id)
        return self.package_dir / directory / f"{sample_id}.json"

    def record_path(self, sample_id: str) -> Path:
        return self._path("adjudication", sample_id)

    def draft_path(self, sample_id: str) -> Path:
        return self._path("gold_drafts", sample_id)

    def human_path(self, sample_id: str) -> Path:
        return self._path("annotator_a", sample_id)

    def ai_path(self, sample_id: str) -> Path:
        return self._path("annotator_ai", sample_id)

    def graph(self, sample_id: str) -> PageGraph:
        raw = Path(self._sample(sample_id)["page_graph"])
        path = raw if raw.is_absolute() else self.repo_root / raw
        path = path.resolve()
        if not path.is_relative_to(self.repo_root):
            raise ValueError("样本路径越出仓库")
        return PageGraph.load(path)

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _record(self, sample_id: str) -> dict[str, Any]:
        return json.loads(self.record_path(sample_id).read_text(encoding="utf-8"))

    def _input_errors(
        self, sample_id: str, record: dict[str, Any], draft: DesignIntentIR
    ) -> list[str]:
        errors = []
        human_hash = self._sha256(self.human_path(sample_id))
        ai_hash = self._sha256(self.ai_path(sample_id))
        if record.get("inputs", {}).get("human", {}).get("sha256") != human_hash:
            errors.append("人工原始标注哈希已变化，请重新生成仲裁材料")
        if record.get("inputs", {}).get("ai", {}).get("sha256") != ai_hash:
            errors.append("AI 原始标注哈希已变化，请重新生成仲裁材料")
        if draft.provenance.get("human_annotation_sha256") != human_hash:
            errors.append("金标准草稿记录的人工输入哈希不一致")
        if draft.provenance.get("ai_annotation_sha256") != ai_hash:
            errors.append("金标准草稿记录的 AI 输入哈希不一致")
        return errors

    def progress(self) -> dict[str, int]:
        statuses = Counter(
            str(self._record(sample_id).get("status", "pending"))
            for sample_id in self.samples
        )
        reviewed = statuses["reviewed"] + statuses["finalized"]
        return {
            "reviewed": reviewed,
            "pending": len(self.samples) - reviewed,
            "total": len(self.samples),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "comparison_kind": "human-ai",
            "progress": self.progress(),
            "sample_status": {
                sample_id: str(self._record(sample_id).get("status", "pending"))
                for sample_id in self.samples
            },
        }

    def payload(self, sample_id: str) -> dict[str, Any]:
        self._sample(sample_id)
        return {
            "record": self._record(sample_id),
            "draft": DesignIntentIR.load(self.draft_path(sample_id)).to_dict(),
            "human": DesignIntentIR.load(self.human_path(sample_id)).to_dict(),
            "ai": DesignIntentIR.load(self.ai_path(sample_id)).to_dict(),
        }

    def save(
        self,
        sample_id: str,
        draft_payload: dict[str, Any],
        review_payload: dict[str, Any] | None = None,
        submit: bool = False,
    ) -> dict[str, Any]:
        record = self._record(sample_id)
        if record.get("status") in {"reviewed", "finalized"}:
            raise ValueError("该样本已完成审核，不能继续修改")

        existing = DesignIntentIR.load(self.draft_path(sample_id))
        draft = DesignIntentIR.from_dict(draft_payload)
        graph = self.graph(sample_id)
        if (
            draft.canvas.width != graph.canvas.width
            or draft.canvas.height != graph.canvas.height
        ):
            raise ValueError("金标准草稿画布尺寸与 PageGraph 不一致")

        normalize_group_source_elements(draft)
        AnnotationStore._normalize_token_references(draft)
        protected = existing.provenance
        draft.provenance = {
            **draft.provenance,
            "source_sample_id": sample_id,
            "label_source": "human_ai_adjudication_draft",
            "status": "draft",
            "adjudication_status": "pending",
            "gold_finalized": False,
            "base_annotation": protected.get("base_annotation", "annotator_a"),
            "base_annotation_provenance": protected.get(
                "base_annotation_provenance", {}
            ),
            "human_annotation_sha256": protected.get(
                "human_annotation_sha256"
            ),
            "ai_annotation_sha256": protected.get("ai_annotation_sha256"),
            "ai_used_for_disagreement_review": True,
            "ai_treated_as_human_annotator": False,
            "reviewed_by": None,
            "reviewed_at": None,
            "review_notes": "",
        }
        errors = self._input_errors(sample_id, record, draft)
        errors.extend(AnnotationStore._submission_errors(draft, graph))
        errors = list(dict.fromkeys(errors))

        reviewed = False
        if submit:
            review = review_payload or {}
            reviewer = str(review.get("reviewed_by", "")).strip()
            notes = str(review.get("notes", "")).strip()
            if not reviewer:
                errors.append("请填写真实审核人标识")
            if reviewer == "annotator_ai":
                errors.append("AI 代理不能作为金标准审核人")
            if not notes:
                errors.append("请填写审核说明")
            if not errors:
                reviewed_at = datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )
                record["status"] = "reviewed"
                record["review"] = {
                    **record.get("review", {}),
                    "reviewed_by": reviewer,
                    "reviewed_at": reviewed_at,
                    "notes": notes,
                }
                draft.provenance.update(
                    {
                        "status": "complete",
                        "adjudication_status": "reviewed",
                        "reviewed_by": reviewer,
                        "reviewed_at": reviewed_at,
                        "review_notes": notes,
                    }
                )
                reviewed = True

        self._write_json(self.draft_path(sample_id), draft.to_dict())
        if reviewed:
            self._write_json(self.record_path(sample_id), record)
        return {
            "record": record,
            "draft": draft.to_dict(),
            "errors": errors,
            "reviewed": reviewed,
            "progress": self.progress(),
        }
