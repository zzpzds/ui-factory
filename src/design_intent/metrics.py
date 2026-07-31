"""设计意图恢复任务的无第三方依赖核心指标。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .schema import DesignIntentIR, PageGraph


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def set_f1(predicted: set[Any], gold: set[Any]) -> dict[str, float]:
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def pairwise_cluster_f1(
    predicted_clusters: Iterable[Iterable[str]],
    gold_clusters: Iterable[Iterable[str]],
) -> dict[str, float]:
    def pairs(clusters: Iterable[Iterable[str]]) -> set[tuple[str, str]]:
        result: set[tuple[str, str]] = set()
        for cluster in clusters:
            members = sorted(set(cluster))
            for i, left in enumerate(members):
                for right in members[i + 1:]:
                    result.add((left, right))
        return result

    return set_f1(pairs(predicted_clusters), pairs(gold_clusters))


def bcubed_f1(
    predicted_clusters: Iterable[Iterable[str]],
    gold_clusters: Iterable[Iterable[str]],
) -> dict[str, float]:
    """按实体平均的 B-cubed precision/recall/F1。"""

    def membership(clusters: Iterable[Iterable[str]]) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for cluster in clusters:
            members = set(cluster)
            for member in members:
                # 重叠分组时取最小集合，保证指标行为稳定且可解释。
                if member not in result or len(members) < len(result[member]):
                    result[member] = members
        return result

    pred = membership(predicted_clusters)
    gold = membership(gold_clusters)
    universe = set(pred) | set(gold)
    if not universe:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    precisions = []
    recalls = []
    for item in universe:
        pred_cluster = pred.get(item, {item})
        gold_cluster = gold.get(item, {item})
        overlap = len(pred_cluster & gold_cluster)
        precisions.append(overlap / len(pred_cluster))
        recalls.append(overlap / len(gold_cluster))
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def group_match_f1(
    predicted_clusters: Iterable[Iterable[str]],
    gold_clusters: Iterable[Iterable[str]],
    jaccard_threshold: float = 0.5,
) -> dict[str, float]:
    """贪心一对一组匹配；匹配条件为成员 Jaccard 达到阈值。"""
    predicted = [set(cluster) for cluster in predicted_clusters if cluster]
    gold = [set(cluster) for cluster in gold_clusters if cluster]
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(predicted):
        for gold_index, target in enumerate(gold):
            union = pred | target
            score = len(pred & target) / len(union) if union else 1.0
            if score >= jaccard_threshold:
                candidates.append((score, pred_index, gold_index))
    candidates.sort(reverse=True)

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matches = 0
    for _, pred_index, gold_index in candidates:
        if pred_index in used_pred or gold_index in used_gold:
            continue
        used_pred.add(pred_index)
        used_gold.add(gold_index)
        matches += 1

    precision = matches / len(predicted) if predicted else float(not gold)
    recall = matches / len(gold) if gold else float(not predicted)
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def _source_signature(ir: DesignIntentIR, entity_id: str) -> tuple[int, ...]:
    elements = {element.id: element for element in ir.elements}
    groups = {group.id: group for group in ir.groups}
    if entity_id in elements:
        return tuple(sorted(elements[entity_id].source_node_ids))
    if entity_id in groups:
        element_by_id = elements
        source_ids = {
            node_id
            for element_id in groups[entity_id].source_element_ids
            for node_id in element_by_id[element_id].source_node_ids
        }
        return tuple(sorted(source_ids))
    return ()


def parent_edge_f1(predicted: DesignIntentIR, gold: DesignIntentIR) -> dict[str, float]:
    """使用源节点签名对齐实体后计算父子边，避免任意 IR id 影响指标。"""

    def edges(ir: DesignIntentIR) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
        result = set()
        for edge in ir.tree:
            child = _source_signature(ir, edge.child_id)
            parent = () if edge.parent_id == "page_root" else _source_signature(
                ir, edge.parent_id
            )
            result.add((parent, child))
        return result

    return set_f1(edges(predicted), edges(gold))


def normalized_tree_edit_distance(
    predicted: DesignIntentIR,
    gold: DesignIntentIR,
) -> float:
    """
    固定源实体标签下的归一化树编辑距离。

    每个缺失/新增实体计一次编辑，每个直接父节点变化计一次 move。
    该任务的实体可通过 source_node_ids 对齐，因此不需要求解无标签树匹配。
    """

    def parent_map(ir: DesignIntentIR) -> dict[tuple[int, ...], tuple[int, ...]]:
        result = {}
        for edge in ir.tree:
            child = _source_signature(ir, edge.child_id)
            parent = () if edge.parent_id == "page_root" else _source_signature(
                ir, edge.parent_id
            )
            result[child] = parent
        return result

    pred = parent_map(predicted)
    target = parent_map(gold)
    universe = set(pred) | set(target)
    if not universe:
        return 0.0
    edits = sum(pred.get(entity) != target.get(entity) for entity in universe)
    return edits / len(universe)


def leaf_preservation_f1(predicted: DesignIntentIR, gold: DesignIntentIR) -> dict[str, float]:
    pred = {
        tuple(sorted(element.source_node_ids))
        for element in predicted.elements
    }
    target = {
        tuple(sorted(element.source_node_ids))
        for element in gold.elements
    }
    return set_f1(pred, target)


def _layout_by_signature(ir: DesignIntentIR) -> dict[tuple[int, ...], Any]:
    group_by_id = {group.id: group for group in ir.groups}
    result = {}
    for layout in ir.layouts:
        if layout.target_id not in group_by_id:
            continue
        result[_source_signature(ir, layout.target_id)] = layout
    return result


def layout_mode_accuracy(predicted: DesignIntentIR, gold: DesignIntentIR) -> float:
    pred_by_members = _layout_by_signature(predicted)
    gold_by_members = _layout_by_signature(gold)
    common = set(pred_by_members) & set(gold_by_members)
    if not common:
        return 0.0
    return sum(
        pred_by_members[key].mode == gold_by_members[key].mode for key in common
    ) / len(common)


def layout_mode_macro_f1(predicted: DesignIntentIR, gold: DesignIntentIR) -> float:
    pred = _layout_by_signature(predicted)
    target = _layout_by_signature(gold)
    common = set(pred) & set(target)
    if not common:
        return 0.0
    classes = sorted(
        {pred[key].mode for key in common} | {target[key].mode for key in common}
    )
    f1_values = []
    for mode in classes:
        true_positive = sum(
            pred[key].mode == mode and target[key].mode == mode for key in common
        )
        false_positive = sum(
            pred[key].mode == mode and target[key].mode != mode for key in common
        )
        false_negative = sum(
            pred[key].mode != mode and target[key].mode == mode for key in common
        )
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1_values.append(_f1(precision, recall))
    return sum(f1_values) / len(f1_values)


def layout_numeric_errors(
    predicted: DesignIntentIR,
    gold: DesignIntentIR,
) -> tuple[float, float]:
    pred = _layout_by_signature(predicted)
    target = _layout_by_signature(gold)
    common = set(pred) & set(target)
    if not common:
        return 0.0, 0.0
    normalizer = max(gold.canvas.width, gold.canvas.height, 1.0)
    gap_error = sum(
        abs(pred[key].gap - target[key].gap) / normalizer for key in common
    ) / len(common)
    padding_error = sum(
        sum(
            abs(left - right)
            for left, right in zip(pred[key].padding, target[key].padding)
        )
        / (4 * normalizer)
        for key in common
    ) / len(common)
    return gap_error, padding_error


def token_coverage(ir: DesignIntentIR) -> float:
    if not ir.elements:
        return 1.0
    covered = sum(bool(element.style_token_refs) for element in ir.elements)
    return covered / len(ir.elements)


def semantic_naming_rate(ir: DesignIntentIR) -> float:
    entities = [*ir.elements, *ir.groups]
    if not entities:
        return 1.0
    generic_prefixes = ("e_", "g_", "node_", "text_", "image_", "container_")
    semantic = sum(
        bool(entity.name)
        and not entity.name.lower().startswith(generic_prefixes)
        for entity in entities
    )
    return semantic / len(entities)


def layer_compression_ratio(ir: DesignIntentIR, graph: PageGraph) -> float:
    if not graph.nodes:
        return 0.0
    generated = len(ir.elements) + len(ir.groups)
    return 1.0 - generated / len(graph.nodes)


def compute_intent_metrics(
    predicted: DesignIntentIR,
    gold: DesignIntentIR,
    graph: PageGraph | None = None,
) -> dict[str, float]:
    def signature_text(ir: DesignIntentIR, entity_id: str) -> str:
        return ",".join(str(value) for value in _source_signature(ir, entity_id))

    predicted_groups = [
        [signature_text(predicted, element_id) for element_id in group.source_element_ids]
        for group in predicted.groups
    ]
    gold_groups = [
        [signature_text(gold, element_id) for element_id in group.source_element_ids]
        for group in gold.groups
    ]
    predicted_tokens = [
        [signature_text(predicted, member_id) for member_id in token.member_ids]
        for token in predicted.style_tokens
    ]
    gold_tokens = [
        [signature_text(gold, member_id) for member_id in token.member_ids]
        for token in gold.style_tokens
    ]

    result: dict[str, float] = {}
    for prefix, values in (
        ("leaf", leaf_preservation_f1(predicted, gold)),
        ("group_pair", pairwise_cluster_f1(predicted_groups, gold_groups)),
        ("group_bcubed", bcubed_f1(predicted_groups, gold_groups)),
        ("group_match", group_match_f1(predicted_groups, gold_groups)),
        ("parent", parent_edge_f1(predicted, gold)),
        ("token_bcubed", bcubed_f1(predicted_tokens, gold_tokens)),
    ):
        for key, value in values.items():
            result[f"{prefix}_{key}"] = value
    result["layout_mode_accuracy"] = layout_mode_accuracy(predicted, gold)
    result["layout_mode_macro_f1"] = layout_mode_macro_f1(predicted, gold)
    gap_mae, padding_mae = layout_numeric_errors(predicted, gold)
    result["gap_normalized_mae"] = gap_mae
    result["padding_normalized_mae"] = padding_mae
    result["tree_normalized_edit_distance"] = normalized_tree_edit_distance(
        predicted, gold
    )
    result["token_coverage"] = token_coverage(predicted)
    result["semantic_naming_rate"] = semantic_naming_rate(predicted)
    if graph is not None:
        result["layer_compression_ratio"] = layer_compression_ratio(
            predicted, graph
        )
    return result
