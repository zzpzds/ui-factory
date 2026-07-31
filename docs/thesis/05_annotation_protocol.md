# Design Intent 人工标注协议 v1

## 目标

人工标注用于构建独立于 DOM/CSS 规则教师的金标准数据。标注问题不是“页面代码如何组织”，而是：

> 如果该页面需要交给设计师继续编辑，哪些对象应独立保留，哪些对象应形成组件，组件应如何布局和复用样式？

标注者可查看截图、实际页面和 PageGraph，但第一轮不得查看 `weak_intent.json`。

## 标注流程

### 第一步：原子元素

标记需要独立选择和编辑的对象：

- `TEXT`：具有独立文本内容或文本样式；
- `IMAGE`：位图、视频封面或内容图像；
- `ICON`：图标、独立 SVG 图形；
- `SHAPE`：具有独立视觉意义的矩形、分隔线或装饰形状；
- `INPUT`：输入框、选择器、文本域；
- `BUTTON_VISUAL`：按钮或具有按钮视觉和操作语义的链接。

不标记：

- 仅用于实现的无样式 wrapper；
- 不可见或被完全遮挡的节点；
- 没有独立编辑价值的文本内联 span；
- 仅继承样式、没有内容或视觉作用的节点。

每个 element 必须填写：

- 唯一 `id`；
- 一个或多个 `source_node_ids`；
- `type`；
- `bbox`；
- 语义名称；
- 文本内容；
- `confidence`。

### 第二步：组件组

当一组设计实体满足任一条件时建立 group。设计实体可以是原子元素，也可以
是已经建立的子 group：

- 应作为整体移动、复制或删除；
- 共享明确的内部布局约束；
- 构成重复出现的 UI 模式；
- 在设计稿中应作为独立 frame 或 component；
- 对外具有清晰语义角色。

允许嵌套组。禁止：

- 只有一个直接原子元素的组；
- 只有一个子 group、且没有新增语义或布局作用的重复 wrapper；
- 与子组成员完全相同、没有新增语义或布局的重复组；
- 只因为 DOM 中存在 div 就创建组。

允许 group 不直接包含任何原子元素。例如 `TABLE` 的直接子实体可以全部是
`TABLE_ROW`。设计树记录直接子实体；`source_element_ids` 由系统自动展开
为全部后代原子元素，仅用于训练与评测，标注者不手工维护。

角色集合：

`CONTAINER`、`CARD`、`NAV`、`FORM`、`LIST`、`LIST_ITEM`、`TABLE`、`TABLE_ROW`、`SECTION`、`UNKNOWN`。

### 第三步：设计树

- 每个 element/group 必须且只能有一个直接父节点；
- 页面最外层使用 `page_root`；
- parent 应是最小的合理设计容器；
- tree 必须无环；
- children 按视觉阅读顺序编号；
- DOM 父子关系只能作为参考，不能直接复制。

### 第四步：布局约束

每个 group 标注：

- `mode`：`HORIZONTAL`、`VERTICAL`、`GRID`、`FREE`；
- `gap`；
- `padding`：上、右、下、左；
- `primary_align`；
- `cross_align`；
- 横向和纵向 resize 行为。

判定原则：

- 元素形成单行且顺序稳定时优先 `HORIZONTAL`；
- 元素形成单列时优先 `VERTICAL`；
- 同时存在稳定行列时使用 `GRID`；
- 明显重叠、绝对定位或无稳定规律时使用 `FREE`；
- 数值以设计上最合理的规则值为准，不要求机械复制浏览器小数。

### 第五步：样式 token

只标注存在复用关系的 token，至少有两个成员：

- `COLOR`：区分 background、foreground、border；
- `TEXT`：字号、字重及必要的行高；
- `RADIUS`；
- `SPACING`。

两个值相同不必然代表同一个 token。只有当它们在设计上应同步修改时才归为同一 token。

## 双标注与裁决

1. 标注者 A、B 独立完成，不交换标注结果；
2. 运行 schema 校验，先修复格式错误；
3. 计算元素、分组、树、布局和 token 一致性；
4. 对分歧逐条讨论；
5. 无法达成一致时由裁决者决定；
6. 保留 `annotator_a.json`、`annotator_b.json` 和 `gold_intent.json`。

## 本地标注工作台

从仓库根目录启动：

```bash
source .venv/bin/activate
python scripts/serve_intent_annotation.py \
  --package_dir data/annotations/intent_pilot_v1 \
  --port 8765
```

浏览器访问 `http://127.0.0.1:8765`。工作台只读取 assignment 中声明的
PageGraph 和截图，不加载 `weak_intent.json`，也不展示另一位标注者的结果。

推荐操作顺序：

1. 在“节点”页签中点击截图候选框或左侧列表，合并选择属于同一可编辑对象的实现节点；
2. 确认元素类型并创建原子元素；
3. 在“实体”页签中勾选至少两个设计实体，或至少一个已有子分组，创建分组；
4. 在右侧设置分组角色、父级、直接子实体、布局方向、间距、内边距和 resize 行为；
5. 勾选至少两个设计实体，在“Token”页签中建立共享样式关系；
6. 随时保存草稿；完成当前页面后点击“提交完成”。

工作台会自动：

- 按视觉位置重排同一父级下的 tree order；
- 回填 element 的 `style_token_refs`；
- 保证每个实体最多保留一条父边；
- 根据 tree 自动展开每个 group 的后代原子覆盖集合；
- 为新建 group 初始化一个 `FREE` 布局约束；
- 在提交前检查边界框、源节点唯一性、树、组成员、布局和 token；
- 原子写入对应标注者目录，提交成功后将 status 设为 `complete`。

草稿允许带校验错误保存，只有全部校验通过后才能提交。切换样本或标注者时，
未保存内容会先写入当前草稿。

## 一致性门槛

试标注通过条件：

| 对象 | 指标 | 最低门槛 |
| --- | --- | ---: |
| 原子元素 | F1 | 0.85 |
| 分组 | B-cubed F1 | 0.75 |
| 设计树 | Parent Edge F1 | 0.80 |
| 布局模式 | Cohen's kappa | 0.75 |
| gap/padding | 归一化 MAE | 0.10 |
| token | B-cubed F1 | 0.70 |

若未达到门槛，必须修订规则并重新试标，不能直接扩大标注。

## 标注质量检查

每个文件提交前运行：

```bash
python scripts/validate_intent_data.py \
  --data_dir data/gold \
  --annotation_name gold_intent.json
```

检查清单：

- 所有实体均有直接父节点；
- 没有环和悬空引用；
- 每个 group 至少有一个直接设计子实体；
- group 不能只包含一个直接原子元素，但可以只直接包含有独立语义的子 group；
- `source_element_ids` 与 tree 展开的后代原子元素一致；
- 每个 layout 都引用 group；
- token 至少有两个有效成员；
- element 的源节点存在；
- 没有查看或复制弱标签；
- `provenance.status` 已从 `draft` 改为 `complete`。

## 文件初始化

```bash
python scripts/init_gold_annotation.py \
  --page_graph data/gold/0001/page_graph.json \
  --output data/gold/0001/annotator_a.json \
  --annotator annotator_a
```

空模板刻意不预填弱标签，避免确认偏差。

Pilot 批量初始化：

```bash
python scripts/create_annotation_pilot.py \
  --data_dir data/processed \
  --pilot_manifest data/intent_pilot_success_manifest.json \
  --duplicate_report outputs/intent-pilot-500/near-duplicate-report.json \
  --output_dir data/annotations/intent_pilot_v1 \
  --num_samples 10
```

Pilot 中途发现未标注样本质量不合格时，必须使用安全替换脚本，不能直接修改
assignment 或覆盖已有标注：

```bash
python scripts/replace_annotation_sample.py \
  --old_sample_id 0733 \
  --new_sample_id 0860 \
  --reason "截图存在大量破图和严重布局失序"
```

脚本只允许替换两位标注者均未开始的空草稿，并检查规模分层、结构指纹、
PageGraph 和近重复冲突；旧模板会归档，替换原因写入
`assignment.json::replacement_history`。任何已经产生元素、分组或 complete
状态的样本都必须保留，不能因标注困难而事后替换。

两位标注者应通过工作台提交完成；若直接编辑 JSON，则需把各自文件的
`provenance.status` 改为 `complete`。之后运行：

```bash
python scripts/evaluate_annotation_agreement.py \
  --package_dir data/annotations/intent_pilot_v1 \
  --output outputs/intent-pilot-500/agreement.json
```

只要存在未完成文件、格式错误或任一一致性指标未达门槛，脚本都会返回非零状态。
