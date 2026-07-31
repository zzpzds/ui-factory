from copy import deepcopy

from src.design_intent.figma_export import export_figma_json
from src.design_intent.editability import compute_editability_checks
from src.design_intent.html_export import export_editable_html
from src.design_intent.metrics import compute_intent_metrics
from src.design_intent.solver import decode_intent_ir
from src.design_intent.schema import BBox, Canvas, PageGraph, PageNode, TreeEdge
from src.design_intent.validation import validate_intent_ir, validate_page_graph
from src.design_intent.weak_supervision import build_weak_intent
from src.design_intent.visual_metrics import compute_visual_metrics
from src.data.intent_dataset import (
    STRUCTURED_FEATURE_DIM,
    TAG_VOCAB,
    TOKEN_RELATION_KINDS,
    build_intent_targets,
    truncate_graph_and_ir,
)
from src.models.intent import DesignIntentRecoveryCore
from src.training.intent_losses import compute_intent_losses
from scripts.evaluate_annotation_agreement import cohen_kappa
import torch
from PIL import Image


def _style(**overrides):
    value = {
        "display": "block",
        "backgroundColor": [0.0, 0.0, 0.0, 0.0],
        "color": [0.1, 0.1, 0.1, 1.0],
        "borderColor": [0.0, 0.0, 0.0, 0.0],
        "borderRadius": 0,
        "borderWidth": 0,
        "fontSize": 16,
        "fontWeight": 400,
    }
    value.update(overrides)
    return value


def _node(node_id, parent_id, tag, text, box, style, depth, sibling, child_count):
    return PageNode(
        id=node_id,
        parent_id=parent_id,
        tag=tag,
        text=text,
        attributes={},
        bbox=BBox(*box),
        computed_style=style,
        depth=depth,
        sibling_index=sibling,
        child_count=child_count,
    )


def make_page_graph():
    nodes = [
        _node(
            0, None, "main", "", (0, 0, 224, 224),
            _style(
                display="flex", flexDirection="column", gap=12,
                paddingTop=8, paddingRight=8, paddingBottom=8, paddingLeft=8,
            ),
            0, 0, 2,
        ),
        _node(
            1, 0, "article", "", (8, 8, 208, 92),
            _style(display="flex", flexDirection="column", gap=4),
            1, 0, 3,
        ),
        _node(2, 1, "h2", "标题一", (16, 16, 160, 20), _style(), 2, 0, 0),
        _node(3, 1, "p", "说明一", (16, 42, 160, 18), _style(), 2, 1, 0),
        _node(
            4, 1, "button", "操作", (16, 66, 64, 20),
            _style(backgroundColor=[0.1, 0.4, 0.9, 1.0], borderRadius=4),
            2, 2, 0,
        ),
        _node(
            5, 0, "article", "", (8, 112, 208, 92),
            _style(display="flex", flexDirection="column", gap=4),
            1, 1, 3,
        ),
        _node(6, 5, "h2", "标题二", (16, 120, 160, 20), _style(), 2, 0, 0),
        _node(7, 5, "p", "说明二", (16, 146, 160, 18), _style(), 2, 1, 0),
        _node(
            8, 5, "button", "操作", (16, 170, 64, 20),
            _style(backgroundColor=[0.1, 0.4, 0.9, 1.0], borderRadius=4),
            2, 2, 0,
        ),
    ]
    return PageGraph(
        schema_version="1.0",
        sample_id="unit-test",
        canvas=Canvas(224, 224),
        nodes=nodes,
    )


def test_page_graph_and_weak_intent_are_valid():
    graph = make_page_graph()
    assert validate_page_graph(graph) == []

    intent = build_weak_intent(graph)
    assert validate_intent_ir(intent, graph) == []
    assert len(intent.elements) == 6
    assert len(intent.groups) == 3
    assert len(intent.layouts) == 3
    assert len(intent.tree) == len(intent.elements) + len(intent.groups)
    assert any(token.kind == "TEXT" for token in intent.style_tokens)
    assert any(token.kind == "COLOR" for token in intent.style_tokens)


def test_tree_uses_design_groups_instead_of_dom_wrappers():
    intent = build_weak_intent(make_page_graph())
    parent = {edge.child_id: edge.parent_id for edge in intent.tree}

    assert parent["g_0"] == "page_root"
    assert parent["g_1"] == "g_0"
    assert parent["g_5"] == "g_0"
    assert parent["e_2"] == "g_1"
    assert parent["e_6"] == "g_5"


def test_figma_export_contains_layout_and_tokens():
    intent = build_weak_intent(make_page_graph())
    figma = export_figma_json(intent)

    canvas = figma["document"]["children"][0]
    assert canvas["type"] == "CANVAS"
    assert canvas["children"][0]["id"] == "g_0"
    assert canvas["children"][0]["layoutMode"] == "VERTICAL"
    assert figma["styles"]
    assert figma["designIntent"]["provenance"]["label_source"] == "weak_supervision_v1"


def test_identical_ir_has_perfect_metrics():
    intent = build_weak_intent(make_page_graph())
    metrics = compute_intent_metrics(intent, deepcopy(intent))

    assert metrics
    lower_is_better = {
        "gap_normalized_mae",
        "padding_normalized_mae",
        "tree_normalized_edit_distance",
    }
    for name, value in metrics.items():
        assert value == (0.0 if name in lower_is_better else 1.0)


def test_validator_detects_design_tree_cycle():
    graph = make_page_graph()
    intent = build_weak_intent(graph)
    parent = {edge.child_id: edge for edge in intent.tree}
    parent["g_0"].parent_id = "g_1"
    parent["g_1"].parent_id = "g_0"

    errors = validate_intent_ir(intent, graph)
    assert any("存在环" in error for error in errors)


def test_validator_detects_missing_tree_parent():
    graph = make_page_graph()
    intent = build_weak_intent(graph)
    intent.tree = [edge for edge in intent.tree if edge.child_id != "e_2"]

    errors = validate_intent_ir(intent, graph)
    assert any("e_2 必须有且仅有一个父节点" in error for error in errors)


def test_intent_targets_cover_all_tasks():
    graph = make_page_graph()
    intent = build_weak_intent(graph)
    targets = build_intent_targets(graph, intent, max_nodes=12)

    assert targets["action"].shape == (12,)
    assert targets["group_affinity"].shape == (12, 12)
    assert targets["token_affinity"].shape == (
        len(TOKEN_RELATION_KINDS), 12, 12
    )
    assert targets["padding"].shape == (12, 4)
    assert (targets["element_type"] >= 0).sum() == 6
    assert (targets["role"] >= 0).sum() == 3
    assert (targets["parent"] >= 0).sum() == 9


def test_design_intent_core_and_losses_backward():
    torch.manual_seed(7)
    batch_size = 2
    node_count = 12
    text_dim = 32
    visual_dim = 48
    model = DesignIntentRecoveryCore(
        dim=64,
        text_dim=text_dim,
        visual_dim=visual_dim,
        structured_dim=STRUCTURED_FEATURE_DIM,
        tag_vocab_size=len(TAG_VOCAB),
    )
    node_mask = torch.zeros(batch_size, node_count)
    node_mask[:, :9] = 1
    dom_parents = torch.full((batch_size, node_count), -1, dtype=torch.long)
    dom_parents[:, 1:9] = torch.tensor([0, 1, 1, 1, 0, 5, 5, 5])
    boxes = torch.zeros(batch_size, node_count, 4)
    boxes[:, :9] = torch.rand(batch_size, 9, 4)
    boxes[..., 2:] = torch.maximum(boxes[..., :2], boxes[..., 2:])

    outputs = model(
        text_features=torch.randn(batch_size, node_count, text_dim),
        visual_features=torch.randn(batch_size, 196, visual_dim),
        tag_ids=torch.randint(0, len(TAG_VOCAB), (batch_size, node_count)),
        structured_features=torch.randn(
            batch_size, node_count, STRUCTURED_FEATURE_DIM
        ),
        dom_parents=dom_parents,
        boxes=boxes,
        node_mask=node_mask,
    )
    assert outputs["action_logits"].shape == (batch_size, node_count, 4)
    assert outputs["group_affinity_logits"].shape == (
        batch_size, node_count, node_count
    )
    assert outputs["token_affinity_logits"].shape == (
        batch_size, len(TOKEN_RELATION_KINDS), node_count, node_count
    )
    assert outputs["parent_logits"].shape == (
        batch_size, node_count, node_count + 1
    )
    assert outputs["alignment_logits"].shape == (batch_size, node_count, 196)

    graph = make_page_graph()
    targets = build_intent_targets(
        graph, build_weak_intent(graph), max_nodes=node_count
    )
    batched_targets = {
        key: torch.stack([value, value]) for key, value in targets.items()
    }
    alignment_targets = outputs["roi_weights"].detach().gt(0).float()
    losses = compute_intent_losses(
        outputs,
        batched_targets,
        node_mask,
        alignment_targets=alignment_targets,
    )
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.decoder.action_head.weight.grad is not None


def test_constraint_solver_decodes_valid_oracle_structure():
    graph = make_page_graph()
    gold = build_weak_intent(graph)
    node_count = 12
    targets = build_intent_targets(graph, gold, max_nodes=node_count)

    def classification_logits(target, class_count):
        logits = torch.full((1, node_count, class_count), -6.0)
        for node_id, class_id in enumerate(target.tolist()):
            if class_id >= 0:
                logits[0, node_id, class_id] = 6.0
        return logits

    parent_logits = torch.full((1, node_count, node_count + 1), -6.0)
    for node_id, parent_id in enumerate(targets["parent"].tolist()):
        if parent_id >= 0:
            parent_logits[0, node_id, parent_id] = 6.0

    outputs = {
        "action_logits": classification_logits(targets["action"], 4),
        "element_type_logits": classification_logits(targets["element_type"], 6),
        "role_logits": classification_logits(targets["role"], 10),
        "layout_mode_logits": classification_logits(targets["layout_mode"], 4),
        "primary_align_logits": classification_logits(targets["primary_align"], 4),
        "cross_align_logits": classification_logits(targets["cross_align"], 4),
        "gap": targets["gap"].unsqueeze(0),
        "padding": targets["padding"].unsqueeze(0),
        "group_affinity_logits": (
            targets["group_affinity"] * 12 - 6
        ).unsqueeze(0),
        "token_affinity_logits": (
            targets["token_affinity"] * 12 - 6
        ).unsqueeze(0),
        "parent_logits": parent_logits,
    }
    predicted = decode_intent_ir(graph, outputs)
    assert validate_intent_ir(predicted, graph) == []

    metrics = compute_intent_metrics(predicted, gold)
    assert metrics["leaf_f1"] == 1.0
    assert metrics["group_match_f1"] == 1.0
    assert metrics["parent_f1"] == 1.0


def test_graph_and_ir_truncation_remain_consistent():
    graph = make_page_graph()
    intent = build_weak_intent(graph)
    truncated_graph, truncated_intent = truncate_graph_and_ir(
        graph, intent, max_nodes=6
    )

    assert len(truncated_graph.nodes) == 6
    assert validate_page_graph(truncated_graph) == []
    assert validate_intent_ir(truncated_intent, truncated_graph) == []
    assert {element.id for element in truncated_intent.elements} == {
        "e_2", "e_3", "e_4"
    }
    assert {group.id for group in truncated_intent.groups} == {"g_0", "g_1"}


def test_html_export_uses_token_variables_and_preserves_text():
    intent = build_weak_intent(make_page_graph())
    html = export_editable_html(intent)

    assert "<!doctype html>" in html
    assert "--token_" in html
    assert "var(--token_" in html
    assert "标题一" in html
    assert 'data-intent-id="g_0"' in html


def test_editability_checks_pass_for_weak_intent():
    checks = compute_editability_checks(build_weak_intent(make_page_graph()))
    assert checks["token_reference_integrity"] == 1.0
    assert checks["layout_order_consistency"] == 1.0


def test_visual_metrics_identical_images():
    image = Image.new("RGB", (32, 32), (120, 80, 40))
    metrics = compute_visual_metrics(image, image.copy())
    assert metrics["mse"] == 0.0
    assert metrics["mean_rgb_distance"] == 0.0
    assert abs(metrics["ssim"] - 1.0) < 1e-9


def test_core_supports_code_only_ablation():
    model = DesignIntentRecoveryCore(
        dim=32,
        text_dim=16,
        visual_dim=24,
        structured_dim=STRUCTURED_FEATURE_DIM,
        tag_vocab_size=len(TAG_VOCAB),
        use_visual=False,
    )
    outputs = model(
        text_features=torch.randn(1, 4, 16),
        visual_features=torch.randn(1, 196, 24),
        tag_ids=torch.zeros(1, 4, dtype=torch.long),
        structured_features=torch.randn(1, 4, STRUCTURED_FEATURE_DIM),
        dom_parents=torch.tensor([[-1, 0, 0, 1]]),
        boxes=torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 0.5, 0.5], [0.5, 0, 1, 0.5], [0, 0.5, 1, 1]]],
            dtype=torch.float32,
        ),
        node_mask=torch.ones(1, 4),
    )
    assert torch.isfinite(outputs["action_logits"]).all()
    assert torch.all(outputs["gates"][..., 0] == 1)
    assert torch.all(outputs["gates"][..., 1:] == 0)


def test_group_metrics_ignore_arbitrary_entity_ids():
    graph = make_page_graph()
    left = build_weak_intent(graph)
    right = deepcopy(left)
    element_renames = {
        element.id: f"human_element_{index}"
        for index, element in enumerate(right.elements)
    }
    group_renames = {
        group.id: f"human_group_{index}"
        for index, group in enumerate(right.groups)
    }
    for element in right.elements:
        element.id = element_renames[element.id]
    for group in right.groups:
        group.id = group_renames[group.id]
        group.source_element_ids = [
            element_renames[element_id]
            for element_id in group.source_element_ids
        ]
    for edge in right.tree:
        edge.child_id = element_renames.get(
            edge.child_id, group_renames.get(edge.child_id, edge.child_id)
        )
        edge.parent_id = group_renames.get(edge.parent_id, edge.parent_id)
    for layout in right.layouts:
        layout.target_id = group_renames[layout.target_id]
    for token in right.style_tokens:
        token.member_ids = [
            element_renames.get(
                member_id, group_renames.get(member_id, member_id)
            )
            for member_id in token.member_ids
        ]
    metrics = compute_intent_metrics(left, right, graph)
    assert metrics["group_pair_f1"] == 1.0
    assert metrics["group_bcubed_f1"] == 1.0
    assert metrics["token_bcubed_f1"] == 1.0
    assert cohen_kappa(left, right) == 1.0
