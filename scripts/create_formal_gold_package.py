"""从成功渲染池创建可复现、分层且去近重复的正式 Gold 标注包。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageOps, ImageStat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.create_annotation_pilot import (  # noqa: E402
    empty_annotation,
    size_bin,
    structure_fingerprint,
)
from scripts.detect_near_duplicates import (  # noqa: E402
    difference_hash,
    hamming_distance,
)
from src.design_intent.schema import PageGraph  # noqa: E402


SPLIT_TARGETS = {"train": 30, "validation": 10, "test": 20}
SIZE_TARGETS = {
    "train": {"small": 10, "medium": 15, "large": 5},
    "validation": {"small": 3, "medium": 4, "large": 3},
    "test": {"small": 7, "medium": 12, "large": 1},
}
TOKEN_TARGETS = {"train": 12, "validation": 4, "test": 8}
RESERVE_TARGETS = {"train": 10, "validation": 5, "test": 5}
PAGE_TYPES = ("table", "form", "commerce", "list", "navigation", "content", "generic")
UNSAFE_CONTENT_TERMS = {
    "adult": ("porn", "порно", "секс", "влагин", "пенис", "траха"),
    "gambling": ("online casino", "casino games", "slots", "jackpots", "bingo cards"),
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _stable_noise(seed: int, sample_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / (2**32 - 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rounded_color(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        color = tuple(round(float(channel), 3) for channel in value[:4])
    except (TypeError, ValueError):
        return None
    if len(color) == 4 and color[3] <= 0.05:
        return None
    return color


def style_repeat_evidence(graph: PageGraph) -> dict[str, Any]:
    """统计潜在设计 Token 的重复样式证据，不读取任何弱标签。"""
    buckets: dict[str, Counter[tuple[Any, ...]]] = {
        kind: Counter() for kind in ("COLOR", "TEXT", "RADIUS", "SPACING")
    }
    for node in graph.nodes:
        style = node.computed_style
        background = _rounded_color(style.get("backgroundColor"))
        if background is not None:
            buckets["COLOR"][("background", *background)] += 1
        text_color = _rounded_color(style.get("color"))
        if node.text.strip() and text_color is not None:
            buckets["COLOR"][("text", *text_color)] += 1
            buckets["TEXT"][(
                round(float(style.get("fontSize", 0) or 0), 2),
                int(float(style.get("fontWeight", 400) or 400)),
                *text_color,
            )] += 1
        radius = round(float(style.get("borderRadius", 0) or 0), 2)
        if radius > 0:
            buckets["RADIUS"][(radius,)] += 1
        gap = round(float(style.get("gap", 0) or 0), 2)
        if gap > 0:
            buckets["SPACING"][("gap", gap)] += 1
        padding = tuple(
            round(float(style.get(key, 0) or 0), 2)
            for key in ("paddingTop", "paddingRight", "paddingBottom", "paddingLeft")
        )
        if any(value > 0 for value in padding):
            buckets["SPACING"][("padding", *padding)] += 1

    kinds: dict[str, dict[str, int]] = {}
    for kind, counts in buckets.items():
        repeated = [count for count in counts.values() if count >= 2]
        if repeated:
            kinds[kind] = {
                "repeated_patterns": len(repeated),
                "repeated_members": sum(repeated),
            }
    score = sum(
        evidence["repeated_patterns"] * 2
        + min(evidence["repeated_members"], 30) / 6
        for evidence in kinds.values()
    )
    return {
        "kind_count": len(kinds),
        "score": round(score, 3),
        "kinds": kinds,
    }


def primary_page_type(graph: PageGraph) -> str:
    tags = Counter(node.tag for node in graph.nodes)
    if tags["table"] or tags["tr"] >= 2 or tags["td"] >= 4:
        return "table"
    if sum(tags[tag] for tag in ("input", "select", "textarea")) >= 2:
        return "form"
    if tags["img"] >= 3 and (tags["button"] or tags["a"] >= 4):
        return "commerce"
    if tags["li"] >= 3:
        return "list"
    if tags["nav"] and tags["a"] >= 3:
        return "navigation"
    if tags["article"] or tags["main"] or tags["p"] >= 3:
        return "content"
    return "generic"


def content_safety_flags(graph: PageGraph) -> list[str]:
    text = " ".join(node.text.lower() for node in graph.nodes if node.text)
    return [
        category for category, terms in UNSAFE_CONTENT_TERMS.items()
        if any(term in text for term in terms)
    ]


def broken_image_proxy(graph: PageGraph) -> dict[str, Any]:
    images = [node for node in graph.nodes if node.tag == "img"]
    tiny = [
        node for node in images
        if node.bbox.width <= 20 and node.bbox.height <= 20
    ]
    ratio = len(tiny) / len(images) if images else 0.0
    return {
        "image_count": len(images),
        "tiny_image_count": len(tiny),
        "tiny_image_ratio": round(ratio, 4),
        "mass_failure_suspected": len(images) >= 4 and ratio >= 0.6,
    }


def screenshot_quality(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        gray = image.convert("L")
        stats = ImageStat.Stat(gray)
        width, height = image.size
    contrast = float(stats.stddev[0])
    mean = float(stats.mean[0])
    contrast_score = min(contrast / 64.0, 1.0)
    exposure_score = 1.0 - min(abs(mean - 150.0) / 150.0, 1.0)
    return {
        "width": float(width),
        "height": float(height),
        "luminance_mean": round(mean, 3),
        "luminance_std": round(contrast, 3),
        "score": round(0.75 * contrast_score + 0.25 * exposure_score, 4),
    }


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def duplicate_clusters(
    screenshots: dict[str, Path], split_by_id: dict[str, str], threshold: int
) -> tuple[dict[str, str], dict[str, Any]]:
    hashes = {sample_id: difference_hash(path) for sample_id, path in screenshots.items()}
    disjoint = DisjointSet()
    matches = []
    sample_ids = sorted(hashes)
    for index, left in enumerate(sample_ids):
        for right in sample_ids[index + 1 :]:
            distance = hamming_distance(hashes[left], hashes[right])
            if distance > threshold:
                continue
            disjoint.union(left, right)
            matches.append({
                "left": left,
                "right": right,
                "distance": distance,
                "left_split": split_by_id[left],
                "right_split": split_by_id[right],
                "cross_split": split_by_id[left] != split_by_id[right],
            })
    cluster_by_id = {
        sample_id: disjoint.find(sample_id)
        if sample_id in disjoint.parent else sample_id
        for sample_id in sample_ids
    }
    return cluster_by_id, {
        "method": "dhash_16x16",
        "threshold": threshold,
        "samples": len(sample_ids),
        "matches": matches,
        "cross_split_matches": sum(match["cross_split"] for match in matches),
    }


def _rank_candidates(
    candidates: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    scores = sorted(
        item["style_repeat_evidence"]["score"]
        for item in candidates
        if item["style_repeat_evidence"]["kind_count"] >= 2
    )
    if not scores:
        raise ValueError("没有样本具备至少两类重复样式证据。")
    token_threshold = scores[math.floor(0.55 * (len(scores) - 1))]
    for item in candidates:
        evidence = item["style_repeat_evidence"]
        item["token_rich_candidate"] = (
            evidence["kind_count"] >= 2 and evidence["score"] >= token_threshold
        )
        item["selection_noise"] = round(
            _stable_noise(seed, item["sample_id"]), 6
        )
    return candidates


def _select_split(
    split: str,
    candidates: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    used_clusters: set[str],
    used_fingerprints: set[str],
) -> list[dict[str, Any]]:
    size_remaining = dict(SIZE_TARGETS[split])
    for item in existing:
        size_remaining[item["size_bin"]] -= 1
    if any(value < 0 for value in size_remaining.values()):
        raise ValueError(f"{split} 现有 Gold 已超过规模层目标。")

    required = SPLIT_TARGETS[split] - len(existing)
    chosen: list[dict[str, Any]] = []
    type_counts = Counter(item["page_type"] for item in existing)
    token_count = 0
    while len(chosen) < required:
        available = [
            item for item in candidates
            if item["size_bin"] in size_remaining
            and size_remaining[item["size_bin"]] > 0
            and item["duplicate_cluster"] not in used_clusters
            and item["structure_fingerprint"] not in used_fingerprints
            and item not in chosen
        ]
        if not available:
            raise ValueError(f"{split} 无法满足去重后的分层配额。")
        token_needed = token_count < TOKEN_TARGETS[split]

        def score(item: dict[str, Any]) -> float:
            token_bonus = 8.0 if token_needed and item["token_rich_candidate"] else 0.0
            type_bonus = 2.5 / (1 + type_counts[item["page_type"]])
            evidence = min(item["style_repeat_evidence"]["score"] / 40.0, 1.5)
            return (
                2.0 * item["quality"]["score"]
                + token_bonus
                + type_bonus
                + 0.35 * evidence
                + 0.05 * item["selection_noise"]
            )

        selected = max(available, key=lambda item: (score(item), item["sample_id"]))
        chosen.append(selected)
        size_remaining[selected["size_bin"]] -= 1
        type_counts[selected["page_type"]] += 1
        token_count += int(selected["token_rich_candidate"])
        used_clusters.add(selected["duplicate_cluster"])
        used_fingerprints.add(selected["structure_fingerprint"])

    if token_count < TOKEN_TARGETS[split]:
        raise ValueError(
            f"{split} 仅选出 {token_count} 个 Token 富集候选，"
            f"目标为 {TOKEN_TARGETS[split]}。"
        )
    return chosen


def _reserve_pool(
    candidates: list[dict[str, Any]],
    used_clusters: set[str],
    used_fingerprints: set[str],
    seed: int,
) -> list[dict[str, Any]]:
    reserves: list[dict[str, Any]] = []
    for split in SPLIT_TARGETS:
        split_pool = [item for item in candidates if item["split"] == split]
        type_counts: Counter[str] = Counter()
        size_counts: Counter[str] = Counter()
        for _ in range(RESERVE_TARGETS[split]):
            available = [
                item for item in split_pool
                if item["duplicate_cluster"] not in used_clusters
                and item["structure_fingerprint"] not in used_fingerprints
                and item not in reserves
            ]
            if not available:
                raise ValueError(
                    f"{split} 无法构造 {RESERVE_TARGETS[split]} 个独立备用样本。"
                )

            def score(item: dict[str, Any]) -> float:
                return (
                    2 * item["quality"]["score"]
                    + 1.4 / (1 + type_counts[item["page_type"]])
                    + 1.0 / (1 + size_counts[item["size_bin"]])
                    + int(item["token_rich_candidate"])
                    + 0.05 * _stable_noise(seed + 1, item["sample_id"])
                )

            selected = max(available, key=lambda item: (score(item), item["sample_id"]))
            reserves.append(selected)
            type_counts[selected["page_type"]] += 1
            size_counts[selected["size_bin"]] += 1
            used_clusters.add(selected["duplicate_cluster"])
            used_fingerprints.add(selected["structure_fingerprint"])
    return reserves


def _contact_sheet(
    records: list[dict[str, Any]], data_dir: Path, output: Path, title: str
) -> None:
    columns, cell_width, image_height, label_height = 5, 240, 150, 46
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * cell_width, 34 + rows * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 9), title, fill="#111111")
    for index, record in enumerate(records):
        x = (index % columns) * cell_width
        y = 34 + (index // columns) * (image_height + label_height)
        with Image.open(data_dir / record["sample_id"] / "screenshot.png") as source:
            thumbnail = ImageOps.contain(source.convert("RGB"), (cell_width - 8, image_height - 8))
        px = x + (cell_width - thumbnail.width) // 2
        py = y + (image_height - thumbnail.height) // 2
        sheet.paste(thumbnail, (px, py))
        token = "T" if record["token_rich_candidate"] else "-"
        draw.text(
            (x + 6, y + image_height + 4),
            f"{record['sample_id']}  {record['split']}  {record['size_bin']}  {token}",
            fill="#111111",
        )
        draw.text(
            (x + 6, y + image_height + 21),
            f"{record['page_type']}  n={record['node_count']}  q={record['quality']['score']:.2f}",
            fill="#444444",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=88, optimize=True)


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(records),
        "by_split": dict(sorted(Counter(item["split"] for item in records).items())),
        "by_size": dict(sorted(Counter(item["size_bin"] for item in records).items())),
        "by_page_type": dict(sorted(Counter(item["page_type"] for item in records).items())),
        "token_rich_candidates": sum(item["token_rich_candidate"] for item in records),
    }


def create_package(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    data_dir = (repo_root / args.data_dir).resolve()
    package_dir = (repo_root / args.output_dir).resolve()
    source_manifest = _json(repo_root / args.source_manifest)
    split_manifest = _json(repo_root / args.split_manifest)
    existing_assignment = _json(repo_root / args.existing_package / "assignment.json")
    exclusion_path = repo_root / args.visual_exclusions
    visual_exclusions = _json(exclusion_path) if exclusion_path.exists() else {"samples": []}
    excluded_ids = {str(item["sample_id"]) for item in visual_exclusions["samples"]}
    prior_rejections = [
        {
            "sample_id": str(item["old_sample_id"]),
            "reason": str(item["reason"]),
        }
        for item in existing_assignment.get("replacement_history", [])
    ]
    excluded_ids.update(item["sample_id"] for item in prior_rejections)
    existing_ids = [item["sample_id"] for item in existing_assignment["samples"]]
    split_by_id = {
        sample_id: split
        for split, ids in split_manifest["splits"].items()
        for sample_id in ids
    }
    successful_ids = sorted(set(source_manifest["sample_ids"]) & set(split_by_id))
    screenshots = {
        sample_id: data_dir / sample_id / "screenshot.png"
        for sample_id in successful_ids
    }
    missing = [
        sample_id for sample_id in successful_ids
        if not screenshots[sample_id].is_file()
        or not (data_dir / sample_id / "page_graph.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"缺少 {len(missing)} 个成功样本资产：{missing[:5]}")

    cluster_by_id, duplicate_report = duplicate_clusters(
        screenshots, split_by_id, args.duplicate_threshold
    )
    if duplicate_report["cross_split_matches"]:
        raise ValueError("固定切分中仍存在跨 split 截图近重复，不能继续。")

    candidates = []
    for sample_id in successful_ids:
        graph = PageGraph.load(data_dir / sample_id / "page_graph.json")
        quality = screenshot_quality(screenshots[sample_id])
        count = len(graph.nodes)
        safety_flags = content_safety_flags(graph)
        image_proxy = broken_image_proxy(graph)
        candidates.append({
            "sample_id": sample_id,
            "split": split_by_id[sample_id],
            "node_count": count,
            "size_bin": size_bin(count),
            "page_type": primary_page_type(graph),
            "structure_fingerprint": structure_fingerprint(graph),
            "duplicate_cluster": cluster_by_id[sample_id],
            "style_repeat_evidence": style_repeat_evidence(graph),
            "quality": quality,
            "asset_sha256": {
                "screenshot": _sha256(screenshots[sample_id]),
                "page_graph": _sha256(data_dir / sample_id / "page_graph.json"),
            },
            "content_safety_flags": safety_flags,
            "broken_image_proxy": image_proxy,
            "eligible": (
                8 <= count <= 180
                and quality["luminance_std"] >= 10
                and not safety_flags
                and not image_proxy["mass_failure_suspected"]
            ),
        })
    _rank_candidates(candidates, args.seed)
    by_id = {item["sample_id"]: item for item in candidates}

    existing = []
    for sample_id in existing_ids:
        if sample_id not in by_id:
            raise ValueError(f"现有 Gold {sample_id} 不在成功渲染池中。")
        item = dict(by_id[sample_id])
        item["status"] = "existing_gold"
        item["gold_style_token_count"] = len(
            _json(data_dir / sample_id / "gold_intent.json").get("style_tokens", [])
        )
        existing.append(item)

    used_clusters = {item["duplicate_cluster"] for item in existing}
    used_fingerprints = {item["structure_fingerprint"] for item in existing}
    pool = [
        item for item in candidates
        if item["eligible"]
        and item["sample_id"] not in existing_ids
        and item["sample_id"] not in excluded_ids
        and item["duplicate_cluster"] not in used_clusters
        and item["structure_fingerprint"] not in used_fingerprints
    ]
    selected_new = []
    for split in SPLIT_TARGETS:
        split_existing = [item for item in existing if item["split"] == split]
        split_pool = [item for item in pool if item["split"] == split]
        selected_new.extend(_select_split(
            split, split_pool, split_existing, used_clusters, used_fingerprints
        ))
    for item in selected_new:
        item["status"] = "annotation_required"
    selected = sorted([*existing, *selected_new], key=lambda item: item["sample_id"])
    reserves = _reserve_pool(
        pool, set(used_clusters), set(used_fingerprints), args.seed
    )

    for split, target in SPLIT_TARGETS.items():
        split_records = [item for item in selected if item["split"] == split]
        if len(split_records) != target:
            raise AssertionError(f"{split} 数量不是 {target}")
        actual_sizes = Counter(item["size_bin"] for item in split_records)
        if dict(actual_sizes) != SIZE_TARGETS[split]:
            raise AssertionError(f"{split} 规模分层错误：{actual_sizes}")
    if len({item["duplicate_cluster"] for item in selected}) != len(selected):
        raise AssertionError("正式样本中存在截图近重复簇冲突。")
    if len({item["structure_fingerprint"] for item in selected}) != len(selected):
        raise AssertionError("正式样本中存在相同结构指纹。")

    manifest = {
        "schema_version": "1.0",
        "purpose": "formal_design_intent_gold_v1",
        "created_date": args.created_date,
        "seed": args.seed,
        "sampling_strategy": {
            "fixed_split": str(args.split_manifest),
            "split_targets": SPLIT_TARGETS,
            "size_targets": SIZE_TARGETS,
            "token_rich_minimums_for_new_samples": TOKEN_TARGETS,
            "token_evidence": "page_graph_repeated_computed_styles_without_weak_labels",
            "near_duplicate": f"dhash_16x16_hamming_le_{args.duplicate_threshold}",
            "quality_gate": (
                "8_to_180_nodes_luminance_std_ge_10_no_explicit_adult_or_"
                "gambling_terms_no_mass_tiny_image_failure_then_visual_review"
            ),
            "warning": "样式富集用于覆盖 Token 标注，不代表自然网页总体分布。",
        },
        "visual_review": {
            "status": "complete",
            "completed_date": args.created_date,
            "scope": "contact_sheet_and_full_resolution_review_of_suspected_failures",
        },
        "existing_gold_ids": sorted(existing_ids),
        "annotation_required_ids": sorted(item["sample_id"] for item in selected_new),
        "splits": {
            split: sorted(item["sample_id"] for item in selected if item["split"] == split)
            for split in SPLIT_TARGETS
        },
        "summary": _summary(selected),
        "visual_exclusions": visual_exclusions["samples"],
        "prior_pilot_rejections": prior_rejections,
        "samples": selected,
        "reserve_summary": _summary(reserves),
        "reserve_samples": sorted(reserves, key=lambda item: (item["split"], item["sample_id"])),
    }
    assignment_order = sorted(selected_new, key=lambda item: item["sample_id"]) + sorted(
        existing, key=lambda item: item["sample_id"]
    )
    assignment = {
        "schema_version": "1.0",
        "purpose": "formal_design_intent_gold_v1",
        "seed": args.seed,
        "annotators": ["annotator_a"],
        "locked_sample_ids": sorted(existing_ids),
        "blind_double_annotation": False,
        "weak_labels_visible": False,
        "token_annotation_mode": "candidate_review",
        "near_duplicate_policy": "one_representative_per_dhash_cluster",
        "protocol": "docs/thesis/05_annotation_protocol.md",
        "selection_manifest": str(
            (Path(args.output_dir) / "selection_manifest.json").as_posix()
        ),
        "samples": [{
            **{key: item[key] for key in (
                "sample_id", "split", "node_count", "size_bin", "page_type",
                "structure_fingerprint", "token_rich_candidate", "status",
            )},
            "screenshot": str(Path(args.data_dir) / item["sample_id"] / "screenshot.png"),
            "page_graph": str(Path(args.data_dir) / item["sample_id"] / "page_graph.json"),
        } for item in assignment_order],
    }
    assignment_path = package_dir / "assignment.json"
    annotation_dir = package_dir / "annotator_a"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    if assignment_path.exists():
        old_assignment = _json(assignment_path)
        old_ids = [item["sample_id"] for item in old_assignment["samples"]]
        new_ids = [item["sample_id"] for item in assignment["samples"]]
        if old_ids != new_ids:
            if not args.refresh_unstarted:
                raise ValueError("现有正式标注包样本清单不同，拒绝覆盖。")
            for item in old_assignment["samples"]:
                if item.get("status") != "annotation_required":
                    continue
                old_annotation = _json(
                    annotation_dir / f"{item['sample_id']}.json"
                )
                if any(old_annotation.get(key) for key in (
                    "elements", "groups", "layouts", "tree", "style_tokens"
                )):
                    raise ValueError("正式标注包已有人工内容，拒绝刷新样本。")
            stale_ids = set(old_ids) - set(new_ids)
            for sample_id in stale_ids:
                stale_path = annotation_dir / f"{sample_id}.json"
                if stale_path.exists():
                    stale_path.unlink()
        # AI 辅助标注启动后，重跑采样器不得丢失条件分配和审计配置。
        if old_assignment.get("annotation_workflow"):
            assignment["annotation_workflow"] = old_assignment[
                "annotation_workflow"
            ]
            conditions = {
                item["sample_id"]: item.get("annotation_condition")
                for item in old_assignment["samples"]
            }
            for item in assignment["samples"]:
                condition = conditions.get(item["sample_id"])
                if condition:
                    item["annotation_condition"] = condition
    _write_json(assignment_path, assignment)
    for item in selected:
        target = annotation_dir / f"{item['sample_id']}.json"
        if target.exists():
            continue
        if item["status"] == "existing_gold":
            shutil.copy2(data_dir / item["sample_id"] / "gold_intent.json", target)
        else:
            graph = PageGraph.load(data_dir / item["sample_id"] / "page_graph.json")
            annotation = empty_annotation(graph, "annotator_a")
            annotation.provenance["annotation_phase"] = "formal_gold_expansion"
            annotation.dump(target)
    _write_json(package_dir / "selection_manifest.json", manifest)
    _write_json(package_dir / "near_duplicate_report.json", duplicate_report)

    _contact_sheet(
        sorted(selected_new, key=lambda item: (item["split"], item["size_bin"], item["sample_id"])),
        data_dir,
        package_dir / "annotation_candidates.jpg",
        "Formal Gold v1: 50 annotation candidates (T = token-rich evidence)",
    )
    _contact_sheet(
        sorted(reserves, key=lambda item: (item["split"], item["size_bin"], item["sample_id"])),
        data_dir,
        package_dir / "reserve_candidates.jpg",
        "Formal Gold v1: 30 reserve candidates",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument(
        "--source_manifest", default="data/intent_pilot_success_manifest.json"
    )
    parser.add_argument("--split_manifest", default="data/intent_pilot_split.json")
    parser.add_argument(
        "--existing_package", default="data/annotations/intent_pilot_v1"
    )
    parser.add_argument(
        "--output_dir", default="data/annotations/intent_gold_v1"
    )
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--duplicate_threshold", type=int, default=12)
    parser.add_argument(
        "--visual_exclusions",
        default="data/annotations/intent_gold_v1/visual_exclusions.json",
    )
    parser.add_argument("--refresh_unstarted", action="store_true")
    parser.add_argument("--created_date", default="2026-08-13")
    args = parser.parse_args()
    manifest = create_package(args)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
