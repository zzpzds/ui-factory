# 系统设计蓝图

## 总体设计

新主链路命名为 **Dual-Stream Design Intent Recovery Network**。实现上以 Design Intent IR 为中心，不继续向旧 `FigmaDecoder` 叠加互不关联的 head。

```text
网页截图 I ──► Visual Encoder ──► patch tokens
                                      │
页面实现图 P ─► Page Graph Encoder ─► node tokens
                                      │
                    BBox-guided Cross-modal Fusion
                                      │
                         Intent Entity Decoder
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
       Atomic/Drop     Group Affinity   Layout/Token
          └───────────────┼───────────────┘
                          ▼
                  Design Tree Decoder
                          ▼
                 Constraint-aware Solver
                          ▼
                   Design Intent IR
                          ▼
              Editable Figma JSON / HTML
```

## 关键设计决策

### 1. 不做自由长度 JSON 生成

模型预测结构化决策，IR 和 Figma JSON 由确定性构建器生成。这样可以：

- 保证 JSON 合法；
- 对每个字段建立清晰监督；
- 单独评测每类设计意图；
- 避免小数据下训练自回归 JSON 解码器。

### 2. 候选实体而非一比一 DOM 节点

候选集合包含：

- 可见 DOM 节点；
- 文本、图像、图标和控件原子候选；
- DOM 容器候选；
- 由空间包含、重复结构和对齐关系产生的额外组候选。

模型为候选预测 `ATOMIC`、`GROUP`、`MERGE` 或 `DROP`。组件组由候选组和成员关系共同决定，而不是强制每个 DOM wrapper 都成为图层。

### 3. 分组和 token 都按关系建模

组件分组使用元素对之间的同组概率；样式 token 使用样式对象之间的同 token 概率。推理时通过受约束聚类得到离散集合。这样不需要预设每页组件数或 token 数。

### 4. 设计树必须无环

父指针 head 只提供候选边分数。约束求解器负责：

- 过滤无效父节点；
- 禁止自环；
- 保证父节点包含或合理覆盖子节点；
- 消除环；
- 加入统一页面根节点；
- 按空间阅读顺序排序 children。

### 5. 视觉对齐使用 bbox 引导

每个页面节点先聚合其 bbox 覆盖的视觉 patch，再执行可学习跨模态注意力。全局注意力保留为补充，而不是要求模型从 196 个 patch 中重新发现已知位置。

## Design Intent IR v1

```json
{
  "schema_version": "1.0",
  "canvas": {
    "width": 1280,
    "height": 800
  },
  "elements": [],
  "groups": [],
  "tree": [],
  "layouts": [],
  "style_tokens": [],
  "provenance": {
    "source_sample_id": "sample-id",
    "model_version": "checkpoint-id"
  }
}
```

### Element

```json
{
  "id": "e_12",
  "source_node_ids": [12, 13],
  "type": "TEXT",
  "bbox": [120, 80, 260, 32],
  "name": "product_title",
  "text": "Example",
  "style_token_refs": ["text/body", "color/foreground"]
}
```

### Group

```json
{
  "id": "g_4",
  "source_element_ids": ["e_10", "e_11", "e_12"],
  "role": "CARD",
  "bbox": [96, 48, 320, 240],
  "name": "product_card"
}
```

`tree` 是 group 直接父子关系的唯一真值，直接子实体可以是 element 或
另一个 group。`source_element_ids` 不是“直接原子成员”，而是从 tree
自动展开得到的全部后代原子元素集合，用于分组监督、实体签名和一致性指标。
例如 `TABLE` 可只直接包含多个 `TABLE_ROW`，同时其
`source_element_ids` 自动覆盖所有行中的单元格原子元素。

### Layout

```json
{
  "target_id": "g_4",
  "mode": "VERTICAL",
  "gap": 12,
  "padding": [16, 16, 16, 16],
  "primary_align": "START",
  "cross_align": "STRETCH",
  "horizontal_resize": "STRETCH",
  "vertical_resize": "HUG"
}
```

### Tree Edge

```json
{
  "parent_id": "page_root",
  "child_id": "g_4",
  "order": 2
}
```

### Style Token

```json
{
  "id": "color/primary",
  "kind": "COLOR",
  "value": {"rgba": [0.086, 0.467, 1.0, 1.0]},
  "member_ids": ["e_3", "e_8", "e_15"]
}
```

## 数据产物

每个渲染样本统一为：

```text
sample/
├── page.html
├── screenshot.png
├── page_graph.json
├── weak_intent.json
├── gold_intent.json       # 仅金标准样本
└── meta.json
```

`page_graph.json` 取代散落的 `nodes.pt`、`parents.pt`、`node_texts.json` 和 `styles.json` 作为主格式。迁移期继续写旧文件，保证现有实验可复现。

## 弱标签教师

弱标签不是简单复制 DOM，而是由多个可解释规则投票：

### 原子元素

- 文本叶、图像、SVG、输入控件优先保留；
- 纯 wrapper 且无独立视觉或布局作用时标记为 `DROP/MERGE`；
- 伪元素、背景图和 canvas 进入低置信候选。

### 组件分组

综合以下证据：

- DOM 最近公共祖先；
- bbox 包含与间距稳定性；
- flex/grid 容器；
- 重复子树结构；
- 同类样式和视觉对齐；
- 语义标签与 ARIA role。

每条标签保存证据列表和置信度。规则冲突时不强行产生高置信标签。

### 布局

- CSS flex/grid 提供强证据；
- 子元素空间分布提供独立验证；
- 二者冲突时降低置信度；
- gap 和 padding 从计算样式与几何反推值交叉校验。

### 样式 token

- 先按规范化后的属性完全匹配；
- 再在预设容差内合并近似颜色、字号、圆角和间距；
- token assignment 以成对关系保存，避免任意 token 编号造成标签置换问题。

### 设计树

只在已选择的原子元素和组上构造，不直接沿用 DOM 父子边。父节点必须是最小合理设计容器；单子节点无作用 wrapper 不保留。

## 模型模块

建议新增模块，不直接破坏旧基线：

```text
src/design_intent/
├── schema.py
├── validation.py
├── weak_supervision.py
├── solver.py
└── figma_export.py

src/models/intent/
├── page_graph_encoder.py
├── multimodal_fusion.py
├── entity_decoder.py
├── relation_heads.py
└── model.py
```

### PageGraphEncoder

节点输入：

```text
text_embedding
+ tag/role embeddings
+ depth/sibling/child-count embeddings
+ bbox MLP
+ computed-style MLP
```

随后使用带 DOM 边和空间边 bias 的 Graph Transformer。CodeBERT 只负责文本/代码片段语义，不再承担几何和样式编码。

### MultimodalFusion

输入节点 token、patch token、节点框与 mask，输出：

- bbox 内 patch 的 pooled visual token；
- bbox 邻域 token；
- 全局页面 token；
- 跨模态融合 node token；
- 独立对齐 logits。

### IntentEntityDecoder

- `AtomicHead`：候选动作和元素类型；
- `GroupAffinityHead`：元素对同组概率；
- `RoleHead`：组语义角色；
- `LayoutHead`：布局分类与连续属性；
- `TokenAffinityHead`：同 token 概率；
- `TreeHead`：父候选分数。

### ConstraintAwareSolver

求解器是可复现的确定性后处理，不参与反向传播。训练评测同时报告：

- 原始 head 指标；
- 求解后的最终 IR 指标。

这样可以区分模型学习能力与规则修正能力。

## 训练入口

保留 `scripts/train.py` 作为旧基线入口，新主方法使用：

```text
scripts/build_page_graphs.py
scripts/build_weak_intent.py
scripts/validate_annotations.py
scripts/train_intent.py
scripts/evaluate_intent.py
scripts/export_figma.py
```

配置按实验分层：

```text
configs/intent/base.yaml
configs/intent/full.yaml
configs/intent/baselines/*.yaml
configs/intent/ablations/*.yaml
```

## 实现顺序

1. 冻结 IR schema 和校验器；
2. 扩展渲染器生成 `page_graph.json`；
3. 实现弱标签教师和可视化检查；
4. 完成 5-10 页试标注，修订标注协议；
5. 实现强启发式基线；
6. 实现 PageGraphEncoder 与主模型 heads；
7. 实现约束求解器和 Figma 导出；
8. 打通训练、评测和重渲染；
9. 扩大数据并执行主实验；
10. 完成消融、统计分析和论文写作。

## 与现有代码的关系

| 现有模块 | 处理 |
| --- | --- |
| `VisualEncoder` | 保留为视觉基线，增加可配置模型与投影层 |
| `CodeEncoder` | 保留预训练权重，封装进 PageGraphEncoder |
| `CrossModalAlignment` | 保留独立监督思想，改为 bbox 引导融合 |
| `NodeContextEncoder` | 由图结构编码器替代 |
| `FigmaDecoder` | 冻结为旧基线，不继续扩展 |
| `WebpageDataset` | 保留旧实验；新增 IntentDataset |
| `build_figma_json` | 保留旧基线；新增 IR 驱动导出器 |
| `metrics.py` | 保留旧指标；新增可编辑性与聚类指标 |

## 最小可行闭环

在扩大训练前，必须先完成以下闭环：

```text
1 个 demo HTML
→ page_graph.json
→ weak_intent.json
→ schema validation
→ heuristic intent prediction
→ Design Intent IR
→ editable JSON
→ 自动编辑任务
→ 指标报告
```

只有这个闭环通过后，才进入模型训练。这样可以先验证任务定义、标签和评测是否自洽，避免在错误目标上投入大规模算力。
