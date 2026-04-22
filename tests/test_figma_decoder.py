import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from src.models.decoder.figma_decoder import (
    _extract_color_from_html,
    _make_node_name,
    _build_tree,
    build_figma_json,
    FigmaStylePredictor,
)


# ── 颜色解析 ──────────────────────────────────────────────
class TestExtractColorFromHtml:
    def test_hex3(self):
        fills = _extract_color_from_html('<nav style="background:#333">')
        assert len(fills) == 1
        assert fills[0]["type"] == "SOLID"
        c = fills[0]["color"]
        assert abs(c["r"] - 0.2) < 0.01
        assert abs(c["g"] - 0.2) < 0.01
        assert abs(c["b"] - 0.2) < 0.01
        assert c["a"] == 1.0

    def test_hex6(self):
        fills = _extract_color_from_html('<div style="background-color:#ff0000">')
        assert len(fills) == 1
        c = fills[0]["color"]
        assert abs(c["r"] - 1.0) < 0.01
        assert c["g"] == pytest.approx(0.0, abs=0.01)
        assert c["b"] == pytest.approx(0.0, abs=0.01)

    def test_rgb(self):
        fills = _extract_color_from_html('<p style="color:rgb(0, 128, 255)">')
        assert len(fills) == 1
        c = fills[0]["color"]
        assert c["r"] == pytest.approx(0.0, abs=0.01)
        assert abs(c["g"] - 0.502) < 0.01
        assert abs(c["b"] - 1.0) < 0.01

    def test_no_color(self):
        fills = _extract_color_from_html('<div class="foo">')
        assert fills == []

    def test_empty(self):
        assert _extract_color_from_html("") == []


# ── 节点命名 ──────────────────────────────────────────────
class TestMakeNodeName:
    def test_tag_with_id(self):
        assert _make_node_name('<nav id="main-nav">', "FRAME") == "nav#main-nav"

    def test_tag_with_class(self):
        assert _make_node_name('<button class="btn-primary">', "INSTANCE") == "button.btn-primary"

    def test_tag_with_aria_label(self):
        assert _make_node_name('<div aria-label="hero section">', "FRAME") == "div[hero section]"

    def test_tag_only(self):
        assert _make_node_name('<section>', "FRAME") == "section"

    def test_no_tag_falls_back_to_type(self):
        assert _make_node_name("some plain text", "TEXT") == "TEXT"

    def test_multiple_attrs_id_wins(self):
        assert _make_node_name('<div id="root" class="container">', "FRAME") == "div#root"


# ── 树形结构 ──────────────────────────────────────────────
class TestBuildTree:
    def _make_nodes(self, boxes):
        """boxes: list of (x1, y1, x2, y2)"""
        return [
            {"id": f"node_{i}", "name": f"node_{i}", "type": "FRAME",
             "fills": [], "strokes": []}
            for i in range(len(boxes))
        ]

    def test_parent_contains_child(self):
        # node_0 包含 node_1
        boxes = torch.tensor([
            [0.0, 0.0, 100.0, 100.0],
            [10.0, 10.0, 50.0, 50.0],
        ])
        nodes = self._make_nodes(boxes)
        roots = _build_tree(nodes, boxes)
        assert len(roots) == 1
        assert roots[0]["id"] == "node_0"
        assert len(roots[0]["children"]) == 1
        assert roots[0]["children"][0]["id"] == "node_1"

    def test_two_independent_roots(self):
        # node_0 和 node_1 互不包含
        boxes = torch.tensor([
            [0.0, 0.0, 40.0, 40.0],
            [60.0, 60.0, 100.0, 100.0],
        ])
        nodes = self._make_nodes(boxes)
        roots = _build_tree(nodes, boxes)
        assert len(roots) == 2

    def test_nearest_ancestor(self):
        # node_0 包含 node_1 包含 node_2 → 树高 3
        boxes = torch.tensor([
            [0.0,  0.0,  100.0, 100.0],
            [10.0, 10.0,  80.0,  80.0],
            [20.0, 20.0,  60.0,  60.0],
        ])
        nodes = self._make_nodes(boxes)
        roots = _build_tree(nodes, boxes)
        assert len(roots) == 1
        assert roots[0]["children"][0]["id"] == "node_1"
        assert roots[0]["children"][0]["children"][0]["id"] == "node_2"

    def test_no_children_key_on_leaf(self):
        boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
        nodes = self._make_nodes(boxes)
        roots = _build_tree(nodes, boxes)
        assert "children" not in roots[0]


# ── 样式预测器激活 ─────────────────────────────────────────
class TestFigmaStylePredictorActivations:
    def setup_method(self):
        self.predictor = FigmaStylePredictor(dim=64)
        self.predictor.eval()

    def test_opacity_in_range(self):
        x = torch.randn(1, 5, 64) * 10  # 故意放大，触发越界
        out = self.predictor(x)
        assert out["opacity"].min() >= 0.0
        assert out["opacity"].max() <= 1.0

    def test_fill_opacity_in_range(self):
        x = torch.randn(1, 5, 64) * 10
        out = self.predictor(x)
        assert out["fill_opacity"].min() >= 0.0
        assert out["fill_opacity"].max() <= 1.0

    def test_corner_radius_nonneg(self):
        x = torch.randn(1, 5, 64) * 10
        out = self.predictor(x)
        assert out["corner_radius"].min() >= 0.0

    def test_rotation_bounded(self):
        x = torch.randn(1, 5, 64) * 10
        out = self.predictor(x)
        assert out["rotation"].min() > -180.0
        assert out["rotation"].max() < 180.0

    def test_visible_in_01(self):
        x = torch.randn(1, 5, 64) * 10
        out = self.predictor(x)
        assert out["visible"].min() >= 0.0
        assert out["visible"].max() <= 1.0
