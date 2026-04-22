# 设计稿：解码器输出质量修复

**日期**：2026-04-22  
**范围**：`src/models/decoder/figma_decoder.py`  
**目标**：让 `build_figma_json` 生成合法的 Figma JSON，消除无效样式值，建立嵌套树形结构，提取真实颜色信息。

---

## 问题清单

| 问题 | 当前表现 | 根因 |
|---|---|---|
| 样式值非法 | `opacity: -0.069`，`corner_radius: 0.073` | `FigmaStylePredictor` 输出无激活函数 |
| visible 类型错误 | `visible: 0.043`（float） | 同上，应为 bool |
| 颜色缺失 | `fills: []` | 没有颜色预测/解析 |
| 结构平铺 | children 只存 ID 引用，非嵌套 | `build_figma_json` 未建树 |
| 节点名无意义 | `name: "<nav style=\"background:#333; h"` | 直接截取原始 HTML |

---

## 方案：模型层激活 + 后处理双保险（方案 B）

### 1. FigmaStylePredictor.forward（模型层）

在各 head 输出后插入激活函数，不增加参数：

| 属性 | 激活 | 输出范围 |
|---|---|---|
| `fill_opacity` | `torch.sigmoid` | (0, 1) |
| `opacity` | `torch.sigmoid` | (0, 1) |
| `corner_radius` | `torch.nn.functional.softplus` | (0, +∞) |
| `rotation` | `torch.tanh * 180` | (-180, 180) |
| `visible` | `torch.sigmoid` | (0, 1)，后处理转 bool |

### 2. build_figma_json 后处理

**① 样式值合法化**

```python
clamp_rules = {
    "opacity":       lambda v: max(0.0, min(1.0, v)),
    "fill_opacity":  lambda v: max(0.0, min(1.0, v)),
    "corner_radius": lambda v: max(0.0, v),
    "rotation":      lambda v: v,  # 激活已保证范围
    "visible":       lambda v: v > 0.5,  # 转 bool
}
```

**② 颜色解析**

新增 `_extract_color_from_html(text: str) -> list[dict]`：
- 正则匹配 `background(-color)?:\s*(#[0-9a-fA-F]{3,6}|rgb\(...\))` 等常见 CSS 颜色
- hex → (r, g, b) → 除以 255 → Figma SOLID fill 格式
- 解析失败返回 `[]`

```json
{"type": "SOLID", "color": {"r": 0.2, "g": 0.2, "b": 0.2, "a": 1.0}}
```

**③ 嵌套树形结构**

新增 `_build_tree(nodes: list[dict], node_boxes: Tensor) -> list[dict]`：
- 对每个节点 j，遍历所有候选父节点 i（i ≠ j），条件：
  `box_i.x1 ≤ box_j.x1 and box_i.y1 ≤ box_j.y1 and box_i.x2 ≥ box_j.x2 and box_i.y2 ≥ box_j.y2`
- 多个候选取面积最小的（最近祖先）
- 有父节点的节点放入父节点的 `children` 列表
- 无父节点的节点作为顶层根节点
- `children` 字段存嵌套节点对象（非 ID 引用）

**④ 节点名提取**

新增 `_make_node_name(html_text: str, node_type: str) -> str`：
- 正则从 HTML tag 提取 `id`、`class`、`aria-label` 属性
- 格式：`tag#id`、`tag.class`、`tag[aria-label]`，无则用 `node_type`

---

## 改动范围

| 文件 | 改动 |
|---|---|
| `src/models/decoder/figma_decoder.py` | `FigmaStylePredictor.forward` 加激活；`build_figma_json` 加后处理逻辑；新增3个辅助函数 |

不改动其他文件。

---

## 成功标准

运行 `python scripts/train.py --demo` 后：
- 生成的 JSON 里所有 `opacity`/`fill_opacity` ∈ [0, 1]
- `visible` 为 `true`/`false`（bool）
- `corner_radius` ≥ 0
- 有内联颜色的节点 `fills` 非空
- 顶层是数组，有内嵌 `children` 的嵌套结构，不是平铺列表
- 节点 `name` 有语义（如 `nav`、`button.btn`），不是 HTML 片段
