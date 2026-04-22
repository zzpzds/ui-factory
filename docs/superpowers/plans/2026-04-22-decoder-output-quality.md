# 解码器输出质量修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `build_figma_json` 生成合法的 Figma JSON：消除无效样式值、`visible` 转 bool、颜色从 HTML 解析、用边界框包含关系建嵌套树。

**Architecture:** 两层修复。模型层：在 `FigmaStylePredictor.forward` 里给各 head 加激活函数，约束输出范围。后处理层：在 `build_figma_json` 里加三个辅助函数——颜色解析、节点命名、树形结构构建。

**Tech Stack:** Python 3.11, PyTorch 2.5, re（标准库）, pytest 9

---

## 文件结构

| 操作 | 路径 | 职责 |
|---|---|---|
| 新建 | `tests/test_figma_decoder.py` | 覆盖所有新逻辑的单元测试 |
| 修改 | `src/models/decoder/figma_decoder.py` | 激活函数 + 三个辅助函数 + build_figma_json 重构 |

---

### Task 1：搭建测试基础设施

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_figma_decoder.py`

- [ ] **Step 1: 创建 tests 目录和 __init__.py**

```bash
mkdir -p /Users/didi/code/ui-factory/tests
touch /Users/didi/code/ui-factory/tests/__init__.py
```

- [ ] **Step 2: 写颜色解析的失败测试**

在 `tests/test_figma_decoder.py` 写入：

```python
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
```

- [ ] **Step 3: 运行测试，确认失败**

```bash
cd /Users/didi/code/ui-factory && source .venv/bin/activate && python -m pytest tests/test_figma_decoder.py::TestExtractColorFromHtml -v 2>&1 | tail -15
```

期望：`ImportError: cannot import name '_extract_color_from_html'`

---

### Task 2：实现 `_extract_color_from_html`

**Files:**
- Modify: `src/models/decoder/figma_decoder.py`（在文件顶部 import 块后添加）

- [ ] **Step 1: 在 figma_decoder.py 顶部加 import re**

在 `import torch` 后面插入：

```python
import re
```

- [ ] **Step 2: 在 FigmaStylePredictor 类定义前添加辅助函数**

```python
def _hex_to_rgba(hex_str: str) -> dict:
    """将 #RGB 或 #RRGGBB 转为 Figma color dict。"""
    h = hex_str.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return {"r": r, "g": g, "b": b, "a": 1.0}


def _extract_color_from_html(html_text: str) -> list[dict]:
    """从 HTML 内联样式提取颜色，返回 Figma fills 格式。解析失败返回 []。"""
    hex_pattern = r'(?:background(?:-color)?|color)\s*:\s*(#[0-9a-fA-F]{3,6})'
    rgb_pattern = r'(?:background(?:-color)?|color)\s*:\s*rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)'

    m = re.search(hex_pattern, html_text)
    if m:
        try:
            color = _hex_to_rgba(m.group(1))
            return [{"type": "SOLID", "color": color}]
        except (ValueError, IndexError):
            pass

    m = re.search(rgb_pattern, html_text)
    if m:
        try:
            color = {
                "r": int(m.group(1)) / 255.0,
                "g": int(m.group(2)) / 255.0,
                "b": int(m.group(3)) / 255.0,
                "a": 1.0,
            }
            return [{"type": "SOLID", "color": color}]
        except (ValueError, IndexError):
            pass

    return []
```

- [ ] **Step 3: 运行颜色测试，确认通过**

```bash
cd /Users/didi/code/ui-factory && source .venv/bin/activate && python -m pytest tests/test_figma_decoder.py::TestExtractColorFromHtml -v 2>&1 | tail -10
```

期望：`4 passed`

- [ ] **Step 4: 提交**

```bash
cd /Users/didi/code/ui-factory && git add src/models/decoder/figma_decoder.py tests/
git commit -m "feat: add _extract_color_from_html with hex/rgb parsing"
```

---

### Task 3：实现 `_make_node_name`

**Files:**
- Modify: `src/models/decoder/figma_decoder.py`
- Modify: `tests/test_figma_decoder.py`

- [ ] **Step 1: 写失败测试，追加到 tests/test_figma_decoder.py**

```python
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
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd /Users/didi/code/ui-factory && source .venv/bin/activate && python -m pytest tests/test_figma_decoder.py::TestMakeNodeName -v 2>&1 | tail -10
```

期望：`ImportError` 或 `FAILED`

- [ ] **Step 3: 在 _extract_color_from_html 后面实现 _make_node_name**

```python
def _make_node_name(html_text: str, node_type: str) -> str:
    """从 HTML 文本提取语义化节点名。"""
    tag_m = re.match(r'<(\w+)', html_text)
    if not tag_m:
        return node_type
    tag = tag_m.group(1)

    id_m = re.search(r'\bid=["\']([^"\']+)["\']', html_text)
    if id_m:
        return f"{tag}#{id_m.group(1)}"

    class_m = re.search(r'\bclass=["\']([^"\']+)["\']', html_text)
    if class_m:
        first_class = class_m.group(1).split()[0]
        return f"{tag}.{first_class}"

    aria_m = re.search(r'\baria-label=["\']([^"\']+)["\']', html_text)
    if aria_m:
        return f"{tag}[{aria_m.group(1)}]"

    return tag
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
cd /Users/didi/code/ui-factory && source .venv/bin/activate && python -m pytest tests/test_figma_decoder.py::TestMakeNodeName -v 2>&1 | tail -10
```

期望：`6 passed`

- [ ] **Step 5: 提交**

```bash
cd /Users/didi/code/ui-factory && git add src/models/decoder/figma_decoder.py tests/test_figma_decoder.py
git commit -m "feat: add _make_node_name with id/class/aria-label extraction"
```

---

### Task 4：实现 `_build_tree`

**Files:**
- Modify: `src/models/decoder/figma_decoder.py`
- Modify: `tests/test_figma_decoder.py`

- [ ] **Step 1: 写失败测试，追加到 tests/test_figma_decoder.py**

```python
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
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd /Users/didi/code/ui-factory && source .venv/bin/activate && python -m pytest tests/test_figma_decoder.py::TestBuildTree -v 2>&1 | tail -10
```

期望：`ImportError` 或 `FAILED`

- [ ] **Step 3: 实现 _build_tree，追加到辅助函数区**

```python
def _build_tree(nodes: list[dict], node_boxes: torch.Tensor) -> list[dict]:
    """
    用边界框包含关系将平铺节点列表重建为嵌套树。
    node_boxes: [N, 4] float，格式 (x1, y1, x2, y2)
    返回根节点列表（每个根节点的 children 字段为嵌套子节点对象）。
    """
    N = len(nodes)
    if N == 0:
        return []

    boxes = node_boxes.float()
    # parent[j] = i 表示 i 是 j 的最近父节点，-1 表示根节点
    parent = [-1] * N

    for j in range(N):
        best_parent = -1
        best_area = float("inf")
        bj = boxes[j]
        for i in range(N):
            if i == j:
                continue
            bi = boxes[i]
            # 检查 i 是否包含 j
            if bi[0] <= bj[0] and bi[1] <= bj[1] and bi[2] >= bj[2] and bi[3] >= bj[3]:
                area = float((bi[2] - bi[0]) * (bi[3] - bi[1]))
                if area < best_area:
                    best_area = area
                    best_parent = i
        parent[j] = best_parent

    # 深拷贝节点（避免修改原始 list）
    import copy
    node_copies = [copy.copy(n) for n in nodes]
    # 移除旧 children 字段
    for n in node_copies:
        n.pop("children", None)

    # 构建树
    for j, p in enumerate(parent):
        if p >= 0:
            if "children" not in node_copies[p]:
                node_copies[p]["children"] = []
            node_copies[p]["children"].append(node_copies[j])

    # 只返回根节点
    return [node_copies[i] for i in range(N) if parent[i] == -1]
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
cd /Users/didi/code/ui-factory && source .venv/bin/activate && python -m pytest tests/test_figma_decoder.py::TestBuildTree -v 2>&1 | tail -10
```

期望：`4 passed`

- [ ] **Step 5: 提交**

```bash
cd /Users/didi/code/ui-factory && git add src/models/decoder/figma_decoder.py tests/test_figma_decoder.py
git commit -m "feat: add _build_tree using bounding box containment"
```

---

### Task 5：给 FigmaStylePredictor 加激活函数

**Files:**
- Modify: `src/models/decoder/figma_decoder.py`（`FigmaStylePredictor.forward`）
- Modify: `tests/test_figma_decoder.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_figma_decoder.py`：

```python
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
```

还需要在测试文件顶部的 import 里加 `FigmaStylePredictor`：

```python
from src.models.decoder.figma_decoder import (
    _extract_color_from_html,
    _make_node_name,
    _build_tree,
    build_figma_json,
    FigmaStylePredictor,
)
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd /Users/didi/code/ui-factory && source .venv/bin/activate && python -m pytest tests/test_figma_decoder.py::TestFigmaStylePredictorActivations -v 2>&1 | tail -10
```

期望：`FAILED`（opacity 超出 [0,1]）

- [ ] **Step 3: 修改 FigmaStylePredictor.forward**

将 `forward` 方法改为：

```python
def forward(self, fused_features: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    fused_features: [B, N, D]
    返回: 样式属性字典，所有值已约束到合法范围
    """
    import torch.nn.functional as F
    h = self.net(fused_features)  # [B, N, D//2]
    outputs = {}
    for name, head in self.style_heads.items():
        raw = head(h).squeeze(-1)  # [B, N]
        if name in ("opacity", "fill_opacity", "visible"):
            outputs[name] = torch.sigmoid(raw)
        elif name == "corner_radius":
            outputs[name] = F.softplus(raw)
        elif name == "rotation":
            outputs[name] = torch.tanh(raw) * 180.0
        else:
            outputs[name] = raw
    return outputs
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
cd /Users/didi/code/ui-factory && source .venv/bin/activate && python -m pytest tests/test_figma_decoder.py::TestFigmaStylePredictorActivations -v 2>&1 | tail -10
```

期望：`5 passed`

- [ ] **Step 5: 提交**

```bash
cd /Users/didi/code/ui-factory && git add src/models/decoder/figma_decoder.py tests/test_figma_decoder.py
git commit -m "feat: add activations to FigmaStylePredictor (sigmoid/softplus/tanh)"
```

---

### Task 6：重构 build_figma_json

**Files:**
- Modify: `src/models/decoder/figma_decoder.py`（`build_figma_json` 函数）
- Modify: `tests/test_figma_decoder.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_figma_decoder.py`：

```python
# ── build_figma_json 集成 ─────────────────────────────────
class TestBuildFigmaJson:
    def _make_sample(self):
        # 3 个节点，node_0 包含 node_1 和 node_2
        type_indices = torch.tensor([0, 2, 2])  # FRAME, TEXT, TEXT
        parent_child_logits = torch.zeros(3, 3)  # 不用，树由 box 建
        style_attrs = {
            "opacity":       torch.tensor([0.8, 0.9, 0.7]),
            "fill_opacity":  torch.tensor([0.6, 0.5, 0.4]),
            "corner_radius": torch.tensor([4.0, 0.0, 2.0]),
            "rotation":      torch.tensor([0.0, 5.0, -5.0]),
            "visible":       torch.tensor([0.9, 0.8, 0.3]),  # 第3个 < 0.5 → False
        }
        node_texts = [
            '<div id="container" style="background:#ffffff">',
            '<p style="color:rgb(0, 0, 0)">Hello</p>',
            '<span class="label">World</span>',
        ]
        node_boxes = torch.tensor([
            [0.0, 0.0, 224.0, 224.0],
            [10.0, 10.0, 100.0, 30.0],
            [10.0, 40.0, 100.0, 60.0],
        ])
        return type_indices, parent_child_logits, style_attrs, node_texts, node_boxes

    def test_returns_nested_tree(self):
        ti, pc, sa, nt, nb = self._make_sample()
        result = build_figma_json(ti, pc, sa, nt, nb)
        assert len(result) == 1  # 只有 1 个根节点
        assert result[0]["id"] == "node_0"
        assert len(result[0]["children"]) == 2

    def test_visible_is_bool(self):
        ti, pc, sa, nt, nb = self._make_sample()
        result = build_figma_json(ti, pc, sa, nt, nb)
        # 展开所有节点
        all_nodes = []
        def collect(nodes):
            for n in nodes:
                all_nodes.append(n)
                collect(n.get("children", []))
        collect(result)
        for n in all_nodes:
            assert isinstance(n["visible"], bool)

    def test_visible_threshold(self):
        ti, pc, sa, nt, nb = self._make_sample()
        result = build_figma_json(ti, pc, sa, nt, nb)
        all_nodes = []
        def collect(nodes):
            for n in nodes:
                all_nodes.append(n)
                collect(n.get("children", []))
        collect(result)
        visibles = [n["visible"] for n in all_nodes]
        # visible tensor: [0.9, 0.8, 0.3] → [True, True, False]
        assert visibles[0] == True
        assert visibles[2] == False

    def test_opacity_clamped(self):
        ti, pc, sa, nt, nb = self._make_sample()
        result = build_figma_json(ti, pc, sa, nt, nb)
        all_nodes = []
        def collect(nodes):
            for n in nodes:
                all_nodes.append(n)
                collect(n.get("children", []))
        collect(result)
        for n in all_nodes:
            assert 0.0 <= n["opacity"] <= 1.0

    def test_fills_parsed_from_html(self):
        ti, pc, sa, nt, nb = self._make_sample()
        result = build_figma_json(ti, pc, sa, nt, nb)
        all_nodes = []
        def collect(nodes):
            for n in nodes:
                all_nodes.append(n)
                collect(n.get("children", []))
        collect(result)
        # node_0 有 background:#ffffff → fills 非空
        root = all_nodes[0]
        assert len(root["fills"]) == 1
        assert root["fills"][0]["type"] == "SOLID"

    def test_node_name_semantic(self):
        ti, pc, sa, nt, nb = self._make_sample()
        result = build_figma_json(ti, pc, sa, nt, nb)
        all_nodes = []
        def collect(nodes):
            for n in nodes:
                all_nodes.append(n)
                collect(n.get("children", []))
        collect(result)
        assert all_nodes[0]["name"] == "div#container"
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd /Users/didi/code/ui-factory && source .venv/bin/activate && python -m pytest tests/test_figma_decoder.py::TestBuildFigmaJson -v 2>&1 | tail -15
```

期望：多个 `FAILED`

- [ ] **Step 3: 重写 build_figma_json**

将 `figma_decoder.py` 里的 `build_figma_json` 函数替换为：

```python
def build_figma_json(
    type_indices: torch.Tensor,
    parent_child_logits: torch.Tensor,
    style_attrs: Optional[dict[str, torch.Tensor]] = None,
    node_texts: Optional[list[str]] = None,
    node_boxes: Optional[torch.Tensor] = None,
) -> list[dict]:
    """
    将解码器输出转换为 Figma JSON 格式（嵌套树）。

    type_indices:        [N] int64
    parent_child_logits: [N, N] float（保留参数，树结构由 node_boxes 包含关系决定）
    style_attrs:         样式属性字典，值已经过激活函数约束
    node_texts:          [N] str
    node_boxes:          [N, 4] float (x1, y1, x2, y2)
    """
    _STYLE_CLAMP = {
        "opacity":       lambda v: max(0.0, min(1.0, v)),
        "fill_opacity":  lambda v: max(0.0, min(1.0, v)),
        "corner_radius": lambda v: max(0.0, v),
        "rotation":      lambda v: v,
        "visible":       lambda v: v > 0.5,
    }

    N = type_indices.shape[0]
    nodes = []

    for i in range(N):
        type_id = type_indices[i].item()
        node_type = IDX_TO_NODE_TYPE.get(type_id, "FRAME")
        html_text = node_texts[i] if node_texts and i < len(node_texts) else ""

        node = {
            "id": f"node_{i}",
            "name": _make_node_name(html_text, node_type),
            "type": node_type,
            "fills": _extract_color_from_html(html_text),
            "strokes": [],
        }

        # 边界框
        if node_boxes is not None and i < node_boxes.shape[0]:
            box = node_boxes[i]
            node["absoluteBoundingBox"] = {
                "x": float(box[0]),
                "y": float(box[1]),
                "width": float(box[2] - box[0]),
                "height": float(box[3] - box[1]),
            }

        # 样式属性
        if style_attrs is not None:
            for attr_name, attr_values in style_attrs.items():
                if i < attr_values.shape[0]:
                    raw = float(attr_values[i].item())
                    clamp_fn = _STYLE_CLAMP.get(attr_name)
                    node[attr_name] = clamp_fn(raw) if clamp_fn else raw

        nodes.append(node)

    # 建树（需要 node_boxes）
    if node_boxes is not None and len(nodes) > 0:
        return _build_tree(nodes, node_boxes)

    return nodes
```

- [ ] **Step 4: 运行全部测试，确认通过**

```bash
cd /Users/didi/code/ui-factory && source .venv/bin/activate && python -m pytest tests/test_figma_decoder.py -v 2>&1 | tail -20
```

期望：所有测试 `passed`

- [ ] **Step 5: 跑 demo 训练，肉眼验证输出**

```bash
cd /Users/didi/code/ui-factory && source .venv/bin/activate && python scripts/train.py --demo --epochs 1 2>&1 | tail -5
```

然后检查 JSON：

```bash
python3 -c "
import json
with open('outputs/figma_epoch1.json') as f:
    nodes = json.load(f)
def show(ns, indent=0):
    for n in ns:
        print(' '*indent + f'{n[\"name\"]} ({n[\"type\"]}) opacity={n.get(\"opacity\"):.3f} visible={n.get(\"visible\")} fills={len(n.get(\"fills\",[]))}')
        show(n.get('children', []), indent+2)
show(nodes)
"
```

期望：树形缩进输出，opacity 在 [0,1]，visible 为 True/False，有颜色的节点 fills 长度 > 0。

- [ ] **Step 6: 提交**

```bash
cd /Users/didi/code/ui-factory && git add src/models/decoder/figma_decoder.py tests/test_figma_decoder.py
git commit -m "feat: refactor build_figma_json with tree structure, color parsing, valid style values"
```

---

## 完整测试命令

```bash
cd /Users/didi/code/ui-factory && source .venv/bin/activate && python -m pytest tests/test_figma_decoder.py -v
```
