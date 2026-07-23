# 基于网页截图与页面代码表示的设计意图恢复模型设计

**日期**：2026-07-23  
**范围**：课题方向调整、模型架构、Design Intent IR、训练目标与评测指标  
**约束**：论文题目保持不变；代码作为必须输入；研究目标偏向生成结果适合设计师二次编辑，而不是单纯像素级复刻。

---

## 1. 研究问题调整

原课题“网页截图 + HTML 源码 → Figma JSON”的实用价值容易被质疑：既然已有源码，为什么还需要再生成设计稿。新的研究表述调整为：

> 前端页面代码能够驱动浏览器渲染界面，但它主要描述低级实现结构，并不等价于设计师可编辑的设计稿结构。本文研究如何结合网页截图与页面代码表示，从实现结构中恢复高层设计意图，包括组件聚合、设计图层树、布局约束和样式归纳，并生成适合二次编辑的 Figma 设计稿。

研究重点从“源码转换”转为“设计意图恢复”。页面代码仍然是必须输入，但模型不假设代码中天然存在 `Card`、`Nav`、`AutoLayout` 等设计概念，而是把这些概念作为预测目标。

---

## 2. 输入与输出

### 2.1 输入

模型输入为二元组：

```text
(I, P)
```

- `I`：网页截图，表示页面最终视觉结果。
- `P`：页面代码表示，表示网页实际展示内容的结构化代码信息。

`P` 统一包含节点结构、文本内容、节点属性、几何位置、样式属性与布局状态等信息。论文中不再把这些信息拆成额外子通道，避免输入定义过于散乱。

### 2.2 输出

模型输出不直接定义为普通 Figma JSON，而是：

```text
Design Intent IR → Editable Figma JSON
```

其中 Figma JSON 需要强调：

- 合理组件分组，而不是一比一照搬 DOM 节点；
- 更接近设计稿的图层树；
- 容器具有 AutoLayout-like 布局属性；
- 图层命名具有语义；
- 颜色、字号、圆角、间距等样式尽可能归纳为可复用 token。

---

## 3. 总体模型架构

模型暂定为：

> 双流融合设计意图恢复模型（Dual-Stream Design Intent Recovery Network）

整体流程：

```text
网页截图 I
页面代码表示 P
      │
      ▼
双流编码器
      │
      ▼
视觉-代码节点对齐
      │
      ▼
Design Intent IR 解码器
      │
      ▼
Figma JSON 生成器
```

### 3.1 视觉流

视觉流编码网页截图，输出视觉 patch token。可以继承现有 `VisualEncoder` 的 ViT-B/16 方案：

```text
I → ViT → visual_tokens [196, D]
```

视觉流用于提供最终视觉分组、空间层级、图像内容和实际观感。

### 3.2 代码流

代码流编码页面代码表示 `P`。每个页面节点对应一个 node token，其特征由两部分组成：

- 文本/结构特征：标签、文本、属性、DOM 深度、兄弟序号；
- 几何/样式特征：bbox、颜色、字号、display、position、flex/grid、margin/padding 等结构化向量。

实现上可以在现有 `CodeEncoder` 与 `NodeContextEncoder` 基础上扩展：

```text
P → PageCodeEncoder → page_node_tokens [N, D]
```

### 3.3 视觉-代码节点对齐

继续保留现有跨模态对齐思想，用节点框与视觉 patch 的重叠关系构造弱监督。区别是对齐目标不再只是服务于 Figma 节点类型预测，而是作为设计意图恢复的底层对齐信号。

```text
visual_tokens + page_node_tokens → fused_node_tokens [N, D]
```

损失可以沿用当前已修复的解耦式对齐：

- softmax attention 用于融合；
- 独立 bilinear/sigmoid 通道用于多 patch 对齐监督；
- InfoNCE 作为辅助判别信号。

---

## 4. Design Intent IR

IR 是新方向的核心。它承接模型预测结果，并约束 Figma JSON 生成。

```text
DesignIntentIR = {
  elements,
  groups,
  layouts,
  styles,
  tree
}
```

### 4.1 Element Layer：基础可编辑元素

表示最终设计稿中必须保留为独立编辑对象的内容。

```json
{
  "id": "e_12",
  "source_nodes": [12, 13],
  "element_type": "TEXT | IMAGE | ICON | SHAPE | INPUT | BUTTON_VISUAL",
  "bbox": [x, y, w, h],
  "text": "...",
  "style_ref": "style_3"
}
```

训练目标：

- `element_type`：分类损失；
- `source_nodes`：由聚合关系间接监督；
- `bbox`：优先由页面代码表示给出，必要时做轻量修正。

### 4.2 Component Group：组件聚合关系

表示哪些低级页面节点或基础元素应被合并为一个设计组件或设计容器。

```json
{
  "id": "g_4",
  "source_elements": ["e_1", "e_2", "e_3"],
  "role": "CARD | NAV | FORM_ITEM | BUTTON | LIST | MODAL | SECTION | TEXT_GROUP | IMAGE_BLOCK | UNKNOWN",
  "bbox": [x, y, w, h],
  "name": "card_product"
}
```

训练目标：

- `group_membership`：元素两两是否属于同一组件，可用 pairwise BCE；
- `role`：组件语义角色分类；
- `bbox`：由成员元素外接框得到；
- `name`：先使用规则生成，后续可扩展为文本生成。

### 4.3 Layout Constraint：布局约束

表示设计师二次编辑时最重要的布局关系。

```json
{
  "target": "g_4",
  "layout_mode": "HORIZONTAL | VERTICAL | GRID | FREE",
  "gap": 12,
  "padding": [16, 16, 16, 16],
  "primary_align": "START | CENTER | END | SPACE_BETWEEN",
  "cross_align": "START | CENTER | END | STRETCH",
  "resize_constraint": {
    "horizontal": "LEFT | RIGHT | CENTER | STRETCH | SCALE",
    "vertical": "TOP | BOTTOM | CENTER | STRETCH | SCALE"
  }
}
```

训练目标：

- `layout_mode`：分类；
- `gap/padding`：回归；
- `alignment`：分类；
- `resize_constraint`：分类。

这些标签可以由 CSS flex/grid 属性、子元素空间关系、bbox 间距和容器边距构造弱监督。

### 4.4 Style Token：样式归纳

表示颜色、字号、圆角、间距等样式是否被归纳成可复用 token。

```json
{
  "id": "style_3",
  "kind": "COLOR | TEXT | EFFECT | SPACING | RADIUS",
  "value": {
    "color": "#1677ff",
    "font_size": 14,
    "font_weight": 500,
    "radius": 8
  },
  "usage_count": 12
}
```

训练目标：

- 样式属性预测：沿用当前 style head；
- token assignment：判断元素样式是否归入某个 token，可先用聚类/规则构造弱标签；
- token coverage：主要作为评测指标。

### 4.5 Design Tree：设计图层树

表示最终 Figma 图层树，不等同于 DOM 树。

```json
{
  "node_id": "g_4",
  "parent": "section_1",
  "children": ["e_1", "e_2", "e_3"],
  "figma_type": "FRAME | GROUP | TEXT | RECTANGLE | INSTANCE"
}
```

训练目标：

- `parent`：父指针分类；
- `figma_type`：分类；
- `children_order`：优先使用空间顺序规则，不单独训练。

---

## 5. 解码器设计

论文叙事采用三层解码，而不是把所有 head 并列展开。

```text
L1 基础元素识别
L2 设计意图恢复：组件聚合 + 语义角色 + 布局约束 + 样式归纳
L3 设计图层树生成
```

对应实现 head：

| Head | 作用 |
| --- | --- |
| `ElementHead` | 基础元素类型预测 |
| `GroupingHead` | pairwise 同组关系预测 |
| `RoleHead` | 组件语义角色预测 |
| `LayoutHead` | layout mode、gap、padding、alignment、constraint |
| `StyleTokenHead` | 样式属性与 token assignment |
| `TreeHead` | 设计树父节点预测 |

总损失：

```text
L = L_elem
  + λ_g L_group
  + λ_r L_role
  + λ_l L_layout
  + λ_s L_style
  + λ_t L_tree
  + λ_a L_align
```

其中：

- `L_elem`：元素类型 CE；
- `L_group`：pairwise BCE 或 contrastive clustering loss；
- `L_role`：组件角色 CE；
- `L_layout`：分类 CE + gap/padding MAE；
- `L_style`：样式回归 + token assignment CE/BCE；
- `L_tree`：父指针 CE；
- `L_align`：视觉 patch 与页面代码节点的对齐损失。

---

## 6. 评测指标

主目标是可编辑性，视觉保真度作为约束项。

| 维度 | 指标 | 说明 | 来源 |
| --- | --- | --- | --- |
| 组件聚合质量 | `Group-F1` | 预测组件组与参考组件组匹配后计算 F1；匹配条件可设为 leaf Jaccard ≥ 0.5 且 bbox IoU ≥ 0.5 | 借鉴 Screen Parsing 的 UI 元素关系建模，以及 DesignCoder 的 UI grouping / Container Match 思路 |
| 图层树质量 | `Parent-F1` | 父子边预测准确性 | 继承现有结构预测指标 |
| 图层树质量 | `TreeBLEU` | 序列化设计树后的局部子结构相似度 | DesignCoder 使用 TreeBLEU 衡量 UI 树结构 |
| 图层树质量 | `Normalized Tree Edit Distance` | 预测树到参考树的编辑距离，按树规模归一化 | DesignCoder 使用 Tree Edit Distance |
| 布局约束质量 | `Layout Mode Accuracy` | `horizontal / vertical / grid / free` 分类准确率 | Figma Auto Layout 属性包含 horizontal、vertical、grid |
| 布局约束质量 | `Gap / Padding MAE` | gap 与 padding 的平均绝对误差 | Figma Auto Layout 将 spacing 和 padding 作为核心属性 |
| 布局约束质量 | `Alignment Accuracy` | 主轴/交叉轴对齐方式分类准确率 | Figma Auto Layout 对齐属性 |
| 布局约束质量 | `Constraint Accuracy` | resize constraint 分类准确率 | Figma constraints 定义图层随父容器缩放的行为 |
| 样式归纳质量 | `Token Coverage` | 颜色、字号、间距、圆角等样式中被归纳为 token 的比例 | 借鉴 Figma styles / variables 的设计系统能力 |
| 样式归纳质量 | `Style MAE / Color ΔE` | 连续样式误差与颜色误差 | 继承当前项目指标，并参考 Design2Code 的 Color 指标 |
| 结构简洁性 | `Layer Compression Ratio` | `1 - 生成图层数 / 可见页面节点数` | 本文自定义，用于衡量是否减少无意义 wrapper |
| 结构简洁性 | `Redundant Wrapper Ratio` | 无样式、无布局作用、单子节点容器比例 | 本文自定义，用于衡量结构可维护性 |
| 结构简洁性 | `Semantic Naming Rate` | 语义图层名占比 | 借鉴 Figma 对 styles/components/variables 命名与描述的设计系统实践 |
| 结构简洁性 | `Leaf Preservation Rate` | 文本、图片、输入框、按钮等关键叶子对象保留率 | 借鉴 UIED / Screen Parsing 对 UI 基础元素检测的要求 |
| 视觉约束 | `Block-Match / Position / Color / CLIP / SSIM` | 防止可编辑性优化导致视觉偏离 | Design2Code、DesignCoder 等 UI 生成工作常用视觉保真度指标 |

论文中应明确：`Group-F1`、`Token Coverage`、`Layer Compression Ratio`、`Redundant Wrapper Ratio` 等是本文结合可编辑 Figma 生成任务提出或改造的指标；视觉指标不是唯一目标，而是约束项。

---

## 7. 与现有代码的迁移关系

现有项目不需要推倒重来，可以按模块升级：

| 现有模块 | 保留 / 调整 |
| --- | --- |
| `render_pages.py` | 继续负责截图、节点框、文本、样式抽取；扩展输出页面代码表示 `P` 和布局弱标签 |
| `VisualEncoder` | 保留 ViT patch 编码 |
| `CodeEncoder` | 升级为 `PageCodeEncoder`，融合文本/结构/几何/样式向量 |
| `CrossModalAlignment` | 保留解耦式对齐损失，作为视觉-代码节点对齐模块 |
| `FigmaDecoder` | 重构为 Design Intent Decoder，新增 grouping/layout/token/tree heads |
| `build_figma_json` | 从 Design Intent IR 生成可编辑 Figma JSON |
| `metrics.py` | 扩展可编辑性指标 |

第一阶段实现不追求全量真实 GT，可通过规则和弱监督构造训练标签：

- 组件聚合：由 bbox 包含、CSS display/flex/grid、视觉间距和重复结构构造；
- 布局约束：由 CSS flex/grid 与子节点空间分布构造；
- 样式 token：由颜色/字号/圆角/间距聚类构造；
- 设计树：由聚合结果和容器关系构造。

---

## 8. 风险与边界

- 组件语义标签可能噪声较大，因此 `role` 不应成为唯一核心指标，应与 grouping/tree/layout 指标共同评价。
- AutoLayout-like 约束是 Figma 友好的近似表示，不要求完全等同浏览器 CSS 布局。
- 视觉保真度不应压过可编辑性目标；实验报告中需要同时给出视觉约束指标和可编辑性主指标。
- 页面代码表示是必需输入，但论文中应避免说“代码直接提供设计语义”，而应强调“代码提供实现证据，模型恢复设计意图”。

---

## 9. 通过标准

该方向设计成立的标准：

- 能明确回答“已有代码为什么还要生成设计稿”：代码不是设计稿，缺少可编辑设计意图；
- 模型输入简洁：网页截图 `I` + 页面代码表示 `P`；
- IR 字段与训练目标、评测指标一一对应；
- 可在现有 pipeline 上渐进改造，而不是完全重写；
- 论文评价从视觉复刻升级为可编辑性评价。
