import torch

from src.models.decoder.figma_decoder import (
    FigmaStylePredictor,
    _build_tree_from_parents,
    _make_node_name,
    build_figma_json,
)


class TestMakeNodeName:
    def test_tag_with_id(self):
        assert _make_node_name('<nav id="main-nav">', "FRAME") == "nav#main-nav"

    def test_tag_with_class(self):
        assert (
            _make_node_name('<button class="btn-primary">', "INSTANCE")
            == "button.btn-primary"
        )

    def test_tag_with_aria_label(self):
        assert (
            _make_node_name('<div aria-label="hero section">', "FRAME")
            == "div[hero section]"
        )

    def test_tag_only(self):
        assert _make_node_name("<section>", "FRAME") == "section"

    def test_no_tag_falls_back_to_type(self):
        assert _make_node_name("some plain text", "TEXT") == "TEXT"


class TestBuildTreeFromParents:
    @staticmethod
    def _nodes(count):
        return [
            {
                "id": f"node_{index}",
                "name": f"node_{index}",
                "type": "FRAME",
                "fills": [],
                "strokes": [],
            }
            for index in range(count)
        ]

    def test_nested_tree(self):
        roots = _build_tree_from_parents(self._nodes(3), [-1, 0, 1])
        assert len(roots) == 1
        assert roots[0]["id"] == "node_0"
        assert roots[0]["children"][0]["id"] == "node_1"
        assert roots[0]["children"][0]["children"][0]["id"] == "node_2"

    def test_independent_roots(self):
        roots = _build_tree_from_parents(self._nodes(2), [-1, -1])
        assert [node["id"] for node in roots] == ["node_0", "node_1"]

    def test_leaf_has_no_children_field(self):
        roots = _build_tree_from_parents(self._nodes(1), [-1])
        assert "children" not in roots[0]


class TestFigmaStylePredictor:
    def setup_method(self):
        self.predictor = FigmaStylePredictor(dim=64).eval()

    def test_output_shapes_and_ranges(self):
        output = self.predictor(torch.randn(2, 5, 64) * 10)
        assert output["reg"].shape == (2, 5, 5)
        assert output["color"].shape == (2, 5, 3, 4)
        assert output["cls_textAlign"].shape == (2, 5, 6)
        assert output["cls_display"].shape == (2, 5, 8)
        assert output["reg"].min() >= 0
        assert output["reg"].max() <= 1
        assert output["color"].min() >= 0
        assert output["color"].max() <= 1


class TestBuildFigmaJson:
    @staticmethod
    def _sample():
        type_indices = torch.tensor([0, 2, 2])
        parent_logits = torch.tensor(
            [
                [-1e4, -2.0, -2.0, 5.0],
                [5.0, -1e4, -2.0, -2.0],
                [5.0, -2.0, -1e4, -2.0],
            ]
        )
        candidate_mask = torch.tensor(
            [
                [False, True, True, True],
                [True, False, True, True],
                [True, True, False, True],
            ]
        )
        node_mask = torch.ones(3)
        styles = {
            "reg": torch.tensor(
                [
                    [0.08, 0.1, 0.8, 0.25, 0.5],
                    [0.0, 0.0, 0.9, 0.25, 0.5],
                    [0.0, 0.0, 0.7, 0.25, 0.5],
                ]
            ),
            "color": torch.tensor(
                [
                    [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0.1, 0.1, 0.1, 1.0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0.2, 0.2, 0.2, 1.0], [0, 0, 0, 0]],
                ]
            ),
            "cls_textAlign": torch.zeros(3, 6),
            "cls_display": torch.zeros(3, 8),
        }
        node_texts = [
            '<div id="container">',
            "<p>Hello</p>",
            '<span class="label">World</span>',
        ]
        node_boxes = torch.tensor(
            [
                [0.0, 0.0, 224.0, 224.0],
                [10.0, 10.0, 100.0, 30.0],
                [10.0, 40.0, 100.0, 60.0],
            ]
        )
        return {
            "type_indices": type_indices,
            "parent_logits": parent_logits,
            "candidate_mask": candidate_mask,
            "node_mask": node_mask,
            "styles": styles,
            "node_texts": node_texts,
            "node_boxes": node_boxes,
        }

    def test_returns_nested_tree(self):
        result = build_figma_json(**self._sample())
        assert len(result) == 1
        assert result[0]["id"] == "node_0"
        assert [child["id"] for child in result[0]["children"]] == [
            "node_1",
            "node_2",
        ]

    def test_attaches_current_style_schema(self):
        result = build_figma_json(**self._sample())
        root = result[0]
        assert root["name"] == "div#container"
        assert root["fills"][0]["type"] == "SOLID"
        assert 0.0 <= root["opacity"] <= 1.0
        assert root["cornerRadius"] >= 0.0
