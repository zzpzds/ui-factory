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
