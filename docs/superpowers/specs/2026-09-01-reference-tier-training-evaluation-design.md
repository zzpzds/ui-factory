# Reference 分层训练与评测设计

**日期**：2026-09-01

**状态**：已实施并通过本地回归；正式多随机种子实验尚未运行

**范围**：Reference v1 数据加载、来源感知损失、训练配置、分层评测与统计防护

**依赖**：`data/annotations/intent_gold_v1`、`data/intent_pilot_split.json`、现有 Design Intent 模型与约束求解器

---

## 1. 背景与目标

当前 DesignIntent Reference v1 包含两类性质不同的参考标注：

- 10 页 `human_gold`：由一名人工标注者完成底稿，并结合 AI 差异提示定稿；
- 50 页 `ai_silver`：由截图、PageGraph、规则候选和同一 Codex 工作流生成，未经人工逐项复核。

固定切分为：

| Split | Human Gold | AI Silver | 合计 |
| --- | ---: | ---: | ---: |
| train | 7 | 23 | 30 |
| validation | 1 | 9 | 10 |
| test | 2 | 18 | 20 |

现有训练和评测入口不能安全使用该数据：

1. `IntentDataset` 只支持资源与标注同目录，而 Reference 的资源位于
   `data/processed/<sample_id>/`，标注位于 `intent_gold_v1/reference/`；
2. 当前损失在整个 batch 内直接归约，无法表达不同页面的来源权重；
3. Human Gold 没有 Token 标签，但当前目标构造会把空 Token 列表当作全部负例；
4. 当前评测把两类来源混成一个均值，并会把 Human Gold 空 Token 计算为伪完美值；
5. 现有 Intent 配置引用了不存在的数据目录或切分清单；
6. 无匹配布局时，当前 MAE 返回 `0`，会把不可计算误写成完美结果。

本设计的目标是建立一条可审计、可复现、默认防误用的 Reference 训练与评测链路，
而不是扩大 Human Gold 的证据强度。

## 2. 方法学边界

该链路允许支持以下结论：

- AI Silver 指标衡量模型与当前 AI 代理标签的一致性；
- Human Gold test 的 2 页结果只作为探索性逐页人类参考；
- AI Silver 可以作为训练辅助标签，但必须降权并进行无 Silver 消融。

该链路不能支持以下结论：

- 模型达到或超过设计师水平；
- 模型相对一般人类设计意图具有稳定准确率；
- AI Silver 等价于人工 Gold；
- 当前 10 页 Human Gold 能提供标注者间信度；
- Silver Token 能证明恢复了人类设计系统中的真实复用意图。

Human Gold 的单人、非独立、AI 辅助定稿属性，以及标注者资历、作者关系和利益冲突
未记录的限制，继续按现有论文协议披露。

## 3. 方案选择

采用 **Reference 专用 Dataset + 页级来源加权 + tier 分层评测**。

不采用以下替代方案：

- 不把 `reference/*.json` 复制或改名成 `gold_intent.json`；这会丢失来源语义并允许
  旧 loader 误把 AI Silver 当作 Gold；
- 不依赖 `batch_size=1` 后乘总损失；该做法在批量训练时没有正确语义；
- 不用重采样代替来源权重；7 页 Human Gold 被频繁重复会提高过拟合风险，且无法
  解决任务标签缺失问题；
- 不自动推测 Human Gold 人工分组的 `source_node_id`；无法唯一映射的人工分组不应
  被转换成看似确定的 DOM 锚点真值。

## 4. 数据架构

### 4.1 Dataset 边界

保留现有 `IntentDataset` 作为页面目录型弱监督数据集，新增：

```python
ReferenceIntentDataset(
    data_dir: str,
    package_dir: str,
    split: str,
    max_nodes: int,
    tiers: list[str] | None = None,
    source_split_manifest: str | None = None,
    verify_hashes: bool = True,
)
```

两个 Dataset 共用一个由显式资源路径、标注路径和监督策略构造样本的内部函数，避免
复制图像、PageGraph 和张量化逻辑。Reference Dataset 不提供写接口。

运行时只打开由构造参数确定的规范路径：输入资源为
`data_dir/<sample_id>/{page_graph.json,screenshot.png}`，标注为
`package_dir/reference/<sample_id>.json`。manifest 中记录的路径用于身份与哈希交叉
校验，不能把任意 manifest 路径直接当作可读取路径。所有规范路径解析后必须位于
对应的 `data_dir` 或 `package_dir` 内。assignment 记录的资源路径必须规范化为与上述
路径相同的仓库相对路径，否则初始化失败。

### 4.2 数据源

Reference Dataset 读取：

- `package_dir/assignment.json`：权威样本 ID、逐页 split 与 tier 集合；
- `package_dir/selection_manifest.json`：冻结切分、资源哈希和选择信息；
- `package_dir/ai_reference_manifest.json`：AI Silver 参考文件哈希和 Token proxy 策略；
- `package_dir/reference/<sample_id>.json`：实际 IR；
- `data_dir/<sample_id>/page_graph.json` 与 `screenshot.png`：模型输入；
- `source_split_manifest`：原弱监督数据切分，用于防止 Reference test 进入弱训练集。

哈希来源固定为：60 页截图和 PageGraph 对照
`selection_manifest.samples[].asset_sha256`；50 份 Silver reference 对照
`ai_reference_manifest.samples[].reference_sha256`；10 份 Human reference 对照
`ai_reference_manifest.human_gold_files[].reference_sha256`。不得从 annotation
provenance 是否碰巧含有 `source_assets` 来决定校验覆盖范围。

`assignment.json` 的逐页 `split` 是加载时的直接索引；`selection_manifest.json` 和原
Pilot split 只用于交叉校验，不能在运行时重新随机切分。

### 4.3 Fail-closed 校验

Dataset 初始化时必须验证：

1. sample ID 唯一，三个 split 无交叉；
2. assignment 与 selection manifest 的样本集合和 split 完全一致；
3. 对 60 个 Reference ID 投影后，assignment split 与原 Pilot split 逐页一致；不要求
   60 页集合等于原 Pilot 的全部 492 页；
4. 文件名、`provenance.source_sample_id` 和 assignment sample ID 一致；
5. `reference_tier` 只能是 `human_gold` 或 `ai_silver`，并与 assignment tier 集合一致；
6. `label_source` 与 tier 匹配；
7. 标注状态为 `complete`；
8. 截图、PageGraph 和 reference 文件存在；正式配置下 SHA-256 与 manifest 一致；
9. 请求的 split 和 tier 非空，且不能包含未知值。

任何冲突均抛出带 sample ID、字段名、期望值和实际值的异常，不静默跳过页面。
metadata-only 预检必须先校验所有记录哈希为 64 位小写 SHA-256；manifest 的 JSON 解析
与审计哈希必须来自同一字节快照，禁止解析后重新打开文件计算哈希。正式 Reference
test 不允许关闭 `verify_hashes`。

### 4.4 Batch 契约

Reference batch 在现有字段之外增加：

```text
reference_tier: list[str]
label_source: list[str]
token_supervision_available: BoolTensor[B]
token_supervision_scope: list[str]
negative_action_supervision_available: BoolTensor[B]
```

节点级 Action 是否参与训练由 `targets.action_mask[B,N]` 表达；页级
`negative_action_supervision_available` 只用于审计，不能替代节点掩码。

其中：

- Human Gold：Token 不可用，Token scope 为 `not_collected`，负 Action 标签不可用；
- AI Silver：Token 可用范围为 `ai_computed_style_proxy`，负 Action 标签按代理标注处理；
- 弱监督旧 Dataset 保持原行为，不伪造 Reference tier。

## 5. 监督可用性

来源权重和任务可用性是两个独立概念。权重描述标签来源的相对可靠性，掩码描述该
任务是否实际存在可审计标签。

### 5.1 各任务规则

| 任务 | Human Gold | AI Silver | 弱监督 |
| --- | --- | --- | --- |
| Atomic Action | 显式原子与显式锚定组正例；其余未知 | 监督显式原子、显式锚定组和代理 DROP | 保持现有语义 |
| Element Type | 显式原子 | 显式原子 | 显式原子 |
| Group Affinity | 人工树中的原子直接父组 | Silver 树中的原子直接父组 | 弱标签树 |
| Group Role | 仅有明确 `source_node_id` 的组 | 仅有明确 `source_node_id` 的组 | 明确锚定组 |
| Layout | 仅有明确 `source_node_id` 的组 | 仅有明确 `source_node_id` 的组 | 明确锚定组 |
| Tree Parent | 子实体有源映射，且父节点为 root 或有明确源映射 | 同左 | 同左 |
| Token | 完全屏蔽 | capped computed-style proxy | 弱 Token |
| Visual Alignment | PageGraph 几何监督 | PageGraph 几何监督 | PageGraph 几何监督 |

不把缺失的组锚点改写成 root 父节点。Tree 目标新增独立 `tree_mask`，只有父边可表示时
才参与损失。

Human Gold 的 `action_mask` 只覆盖：每个 element 的第一个 source node 对应的
`ATOMIC`，以及具有显式 `source_node_id` 的 group 对应的 `GROUP`。未选择节点、
多 source element 的其余 source node 均为未知，mask 为 `0`，不能作为 `DROP`。
`MERGE` 当前没有可解码的直接正标签，本阶段不为任何来源制造 MERGE 真值；这是现有
模型限制，留待后续模型设计处理。AI Silver 和弱监督的代理 DROP 仍按各自置信度参与。

### 5.2 Group Affinity 修正

两个原子元素只有在共享同一个 **非 `page_root` 的直接父组** 时才是正 pair。
直接挂在 `page_root` 下的多个原子不是一个隐式组件组，必须作为负 pair。该修正同时
应用于弱监督和 Reference 目标构造，并增加回归测试。

### 5.3 Token 缺失语义

Human Gold 的空 `style_tokens` 表示“未采集”，不是“确认不存在 Token”。因此：

- `token_pair_mask` 全零；
- Token loss 对模型参数的梯度严格为零；
- Token B-cubed、Coverage 和任何后续 Token 指标全部输出 `unavailable`；
- 不允许依据空列表推断标签可用性，必须读取 provenance 和 package 契约。

## 6. 来源加权损失

### 6.1 页级归约

先在每个页面内部按有效节点或 pair 计算任务损失，再进行页级加权。设：

- `l[i,k]`：页面 `i` 的任务 `k` masked mean；
- `a[i,k]`：该页该任务是否至少有一个有效标签；
- `w[i]`：tier 原始权重；
- `lambda[k]`：现有任务权重。

Annotation-derived 任务集合固定为 `action`、`element`、`group`、`role`、
`layout_mode`、`layout_reg`、`layout_align`、`token` 和 `tree`；`align` 是唯一不使用
tier 权重的现有任务。`a[i,k]` 必须在所有有效掩码生效后计算：pair 任务先去除对角线，
layout 任务先应用 confidence mask。无标签页的 `l[i,k]` 使用与对应 logits 计算图相连
的零，不能返回独立常量。

主配置：

```yaml
reference_tier_weights:
  human_gold: 1.0
  ai_silver: 0.5
```

激活 tier 的权重必须完整、有限且严格大于零。过滤 tier 只能通过 Dataset 的 tier
选择完成，不能使用权重 `0` 代替过滤。

在当前激活的 Reference train 子集上只计算一次：

```text
mean_w = mean(w[i])
normalized_w[i] = w[i] / mean_w
```

默认 7 Human + 23 Silver 时，`mean_w = 18.5 / 30`，Human 与 Silver 的相对贡献
保持 `2:1`，同时训练集期望权重保持为 `1`。权重不能按当前 mini-batch 重新归一化。

Annotation-derived loss 定义为：

```text
L_annotation = mean_i(
  normalized_w[i] * sum_k(lambda[k] * a[i,k] * l[i,k])
)
```

`L_annotation` 使用 batch 中实际页面数作为分母，而不是只使用当前任务有标签的页面
数；无标签任务规定 `a[i,k] * l[i,k] = 0`。因此它表示“每页 Reference 监督的期望
贡献”，不会因为某个来源缺失某项标签而把其他页面的该任务梯度放大。每个命名任务
日志也使用同一个页级公式：

```text
L_k = mean_i(normalized_w[i] * a[i,k] * l[i,k])
```

Alignment 单独记录为不带 tier 权重的页均值。

上述“无标签页面仍占页级分母”是有意选择：例如 Human Gold 不提供 Token 时，Token
任务在完整 Reference train 中只由 23 页 Silver 贡献，而不会被重新放大成 30 页都
有 Token 标签的规模。该语义必须同时用于反向传播和训练日志。

IR confidence 不扩散成新的通用权重。本阶段只保留现有 Action confidence 和 Layout
confidence 参与各自页面内 masked mean；其他任务只使用二值可用性。页内 confidence
先作用于 `l[i,k]`，再乘页级 tier 权重。

Alignment loss 不使用 tier 权重：

```text
L_total = L_annotation + lambda_align * mean_i(l[i,align])
```

过滤为 Human-only 或 Silver-only 时在该训练子集内重新计算 `mean_w`，因此单一 tier 的
权重归一化为 `1`。Silver-only 不重复做 `0.25/0.5/1.0`，因为这只会改变整体学习率。

### 6.2 权重使用范围

- tier 权重只用于 Reference **训练**；
- validation 和 test 不应用训练权重；
- Reference validation 必须按 tier 分别记录无权重损失；
- 所有 Reference 训练消融都使用同一组 9 页 AI Silver validation 的无权重总损失
  选择 checkpoint，包括 Human-only 和 Silver-only；
- 唯一的 Human Gold validation 页只作诊断，不能作为单独早停依据；
- test 在模型、阈值和排除规则冻结后运行，不能参与 checkpoint 选择。

### 6.3 梯度累积

每个 micro-batch 的页均值先乘实际页面数并反向传播；执行 optimizer step 前，再把
累积梯度除以当前窗口的实际页面数。这样最后不足 `grad_accum` 的窗口不会被固定分母
低估，batch 最后一批不足配置大小时也保持页级均值语义。梯度归一化后再执行 clipping。

## 7. 配置设计

弱监督配置统一修正为：

```yaml
data:
  dataset_kind: page_directory
  data_dir: data/processed
  annotation_name: weak_intent.json
  split_manifest: data/intent_pilot_split.json
```

主 Reference 微调配置：

```yaml
data:
  dataset_kind: reference
  data_dir: data/processed
  package_dir: data/annotations/intent_gold_v1
  source_split_manifest: data/intent_pilot_split.json
  train_tiers: [human_gold, ai_silver]
  validation_tiers: [human_gold, ai_silver]
  verify_hashes: true
  max_nodes: 192

training:
  init_checkpoint: outputs/intent-full-v2/best.pt
  reference_tier_weights:
    human_gold: 1.0
    ai_silver: 0.5
  model_selection_tier: ai_silver
```

`train_tiers` 控制训练数据；`validation_tiers` 固定包含两层，以便记录 Human 诊断并
使用相同 9 页 Silver 选择所有消融的 checkpoint。训练入口分别把这两个列表传给
`ReferenceIntentDataset(split="train", tiers=train_tiers)` 和
`ReferenceIntentDataset(split="validation", tiers=validation_tiers)`，不能在缺少
selection tier 时静默回退到 mixed validation。

因此 Human-only 和 Silver-only 只描述 **训练标签来源**，不表示模型选择过程没有使用
Silver validation。论文表格必须单独列出 train tiers 和 selection tier，披露该代理
validation 对所有 Reference 消融的共同影响。

配置文件语义如下：

- `gold_finetune.yaml`：只包含 `human_gold`，避免文件名把 Silver 暗示为 Gold；
- `reference_finetune.yaml`：Human + Silver，Silver 权重 `0.5`，主配置；
- `reference_ai_only.yaml`：只包含 AI Silver；
- `reference_finetune_w025.yaml`：Human + Silver，Silver 权重 `0.25`；
- `reference_finetune_w100.yaml`：Human + Silver，Silver 权重 `1.0`。

完整消融至少覆盖：weak-only、weak + Human-only、weak + Silver-only、完整 Reference 的
Silver 权重 `0.25/0.5/1.0`。所有学习模型变体使用相同 tier 权重。

Group Affinity 和 Tree mask 的目标语义升级会有意改变弱标签张量，因此旧 Dataset 只
保证加载/API 兼容，不保证目标逐字节兼容。所有新训练配置和输出目录使用 `v2` 名称，
不得覆盖已有 `intent-*-v1` 结果。

## 8. 分层评测

### 8.1 输出版本

Reference 模式输出 `intent-evaluation/v2`。顶层必填结构为：

```json
{
  "schema_version": "intent-evaluation/v2",
  "run": {
    "checkpoint": "outputs/example/best.pt",
    "split": "test",
    "reference_package": "data/annotations/intent_gold_v1",
    "statistics_seed": 42,
    "bootstrap_samples": 10000,
    "hashes": {
      "assignment": "sha256:<hex>",
      "selection_manifest": "sha256:<hex>",
      "ai_reference_manifest": "sha256:<hex>",
      "source_split": "sha256:<hex>",
      "checkpoint": "sha256:<hex>"
    }
  },
  "dataset": {
    "expected_ids": ["<sorted_sample_id>"],
    "counts_by_tier": {"human_gold": 2, "ai_silver": 18}
  },
  "strata": {
    "human_gold": {
      "scope": "exploratory_human_reference",
      "counts": {},
      "aggregate_metrics": {},
      "inference": {"status": "not_performed"}
    },
    "ai_silver": {
      "scope": "ai_proxy_consistency_only",
      "counts": {},
      "aggregate_metrics": {},
      "inference": {"status": "exploratory_low_power"}
    },
    "overall_mixed_descriptive": {
      "scope": "mixed_descriptive_only",
      "counts": {},
      "aggregate_metrics": {},
      "inference": {"status": "prohibited"}
    }
  },
  "per_sample": {
    "<sample_id>": {
      "reference_tier": "human_gold",
      "status": "valid",
      "failure": null,
      "metrics": {}
    }
  },
  "failures": []
}
```

`run.hashes` 必须包含 assignment、selection、AI reference manifest、source split 和
checkpoint 的 SHA-256。`dataset.expected_ids` 必须按 sample ID 排序。三个 stratum
对象都必须包含 `scope`、页面 `counts`、`aggregate_metrics` 和 `inference`；这些空对象
只表示上面的顶层结构，字段契约由下文完整定义。

根级不再提供含义模糊的裸 `metrics`。`METRIC_SPECS` 中的每个 key 必须始终出现在每个
逐页 `metrics` 和每个 stratum 的 `aggregate_metrics` 中，不能用字段缺失表达 N/A。
逐页 metric 对象固定为：

```json
{
  "status": "available",
  "value": 0.72,
  "scope": "ai_proxy_consistency_only",
  "reason_code": null
}
```

逐页 unavailable metric 的 `value` 为 `null`，并提供非空 `reason_code`。聚合 metric
对象固定为：

```json
{
  "status": "available",
  "scope": "ai_proxy_consistency_only",
  "counts": {
    "n_expected_pages": 18,
    "n_valid_predictions": 18,
    "n_failed_predictions": 0,
    "n_available": 17,
    "n_unavailable": 1,
    "unavailable_reason_counts": {"no_matched_layout": 1}
  },
  "mean": {"status": "available", "value": 0.72, "reason_code": null},
  "sample_std": {"status": "available", "value": 0.08, "reason_code": null},
  "bootstrap_mean_95_ci": {
    "status": "available",
    "value": [0.68, 0.76],
    "reason_code": null
  },
  "estimand": "mean_among_metric_available_valid_predictions",
  "reason_code": null
}
```

聚合对象的 `status` 枚举为 `available`、`unavailable` 或
`suppressed_by_protocol`。若 `n_available=0`，mean、SD 和 CI 均 unavailable；若
`n_available=1`，mean 可用而 SD 和 CI 为 `insufficient_metric_pages`。Human Gold 的每项
aggregate metric 都是 `suppressed_by_protocol`，reason 为
`human_gold_aggregate_prohibited`。mixed 的 Token 项为 unavailable，reason 为
`cross_tier_token_truth_spaces_incomparable`。

必须满足：`n_expected_pages = n_valid_predictions + n_failed_predictions`，且
`n_available + n_unavailable = n_valid_predictions`。禁止使用 `0`、`1`、`NaN`、
字符串 `"N/A"` 或缺失字段代替不可用值。最终 JSON 使用 `allow_nan=False` 序列化。

### 8.2 三个来源层

`human_gold`：

- 保留 2 页逐页非 Token 指标；
- 聚合字段存在，但全部标记为 `suppressed_by_protocol`，不提供均值、SD 或 CI；
- 逐页 Token 指标为 unavailable；Token aggregate 与其他 Human aggregate 一样标记为
  `suppressed_by_protocol`；
- scope 为 `exploratory_human_reference`。

`ai_silver`：

- 输出每页指标、页均值、样本标准差和页面级 bootstrap 95% CI；
- 固定 bootstrap seed，默认 10,000 次；
- 全部指标 scope 为 `ai_proxy_consistency_only`；Token family 另带
  `interpretation=ai_computed_style_proxy_only`；
- 推断状态标记为 `exploratory_low_power`。

`overall_mixed_descriptive`：

- 非 Token 指标使用不带训练 tier 权重的页面宏平均；
- SD 和 CI 字段存在但标为 `suppressed_by_protocol`；
- Token 指标字段存在但标为 unavailable；
- inference 状态固定为 `prohibited`；
- scope 固定为 `mixed_descriptive_only`；
- 不能作为相对人类准确率的主结果。

### 8.3 Metric 规格

新增集中式 `METRIC_SPECS`。本阶段完整 catalog 冻结为：

| Metric key | Family | Direction | Allowed tier | Primary | Holm family |
| --- | --- | --- | --- | --- | --- |
| `leaf_f1` | structure | higher | Human, Silver, mixed | 是 | `structure_primary` |
| `group_pair_f1` | structure | higher | Human, Silver, mixed | 是 | `structure_primary` |
| `group_bcubed_f1` | structure | higher | Human, Silver, mixed | 是 | `structure_primary` |
| `group_match_f1` | structure | higher | Human, Silver, mixed | 是 | `structure_primary` |
| `parent_f1` | structure | higher | Human, Silver, mixed | 是 | `structure_primary` |
| `tree_normalized_edit_distance` | structure | lower | Human, Silver, mixed | 是 | `structure_primary` |
| `layout_mode_macro_f1` | layout | higher | Human, Silver, mixed | 是 | `layout_primary` |
| `gap_normalized_mae` | layout | lower | Human, Silver, mixed | 是 | `layout_primary` |
| `padding_normalized_mae` | layout | lower | Human, Silver, mixed | 是 | `layout_primary` |
| `token_bcubed_f1` | token_proxy | higher | Silver | 是 | `token_proxy_primary` |
| `leaf_precision`, `leaf_recall` | structure | higher | Human, Silver, mixed | 否 | 无 |
| `group_pair_precision`, `group_pair_recall` | structure | higher | Human, Silver, mixed | 否 | 无 |
| `group_bcubed_precision`, `group_bcubed_recall` | structure | higher | Human, Silver, mixed | 否 | 无 |
| `group_match_precision`, `group_match_recall` | structure | higher | Human, Silver, mixed | 否 | 无 |
| `parent_precision`, `parent_recall` | structure | higher | Human, Silver, mixed | 否 | 无 |
| `layout_mode_accuracy` | layout | higher | Human, Silver, mixed | 否 | 无 |
| `layout_match_coverage` | layout | higher | Human, Silver, mixed | 否 | 无 |
| `token_bcubed_precision`, `token_bcubed_recall` | token_proxy | higher | Silver | 否 | 无 |
| `token_coverage` | token_proxy | higher | Silver | 否 | 无 |
| `semantic_naming_rate` | diagnostic | higher | Human, Silver, mixed | 否 | 无 |
| `layer_compression_ratio` | diagnostic | higher | Human, Silver, mixed | 否 | 无 |

表中 Human、Silver、mixed 分别对应 `human_gold`、`ai_silver` 和
`overall_mixed_descriptive`。Token Purity 和视觉指标尚未在当前评测链实现，不属于 v2
catalog；它们是后续阶段，不得在本阶段结果中临时加入。

Human Gold 调用核心指标函数时显式关闭 Token 指标，不能先计算伪值再删除，但报告器
仍依据 catalog 写出 unavailable 状态对象。

每页布局签名为 layout target group 的 source signature 集合。设参考集合为 `G`，预测
集合为 `P`，则：

```text
layout_match_coverage = |G intersect P| / |G|
```

当 `G` 为空时，coverage 自身 unavailable，reason 为
`reference_layout_labels_absent`；当 `G` 非空但交集为空时 coverage 为 `0`，Layout
Mode、Gap 和 Padding 为 unavailable，reason 为 `no_matched_layout`。条件布局指标先在
每页匹配组内计算，再按页面宏平均，不能跨页面把所有 group 微平均。

AI Silver 聚合的 sample SD 使用 `ddof=1`。bootstrap 以该 metric 的 available 页面为
重采样单位，计算页面均值的 percentile 95% CI，分位点为 `2.5%/97.5%`。每项指标的
随机种子为 `SHA-256("<statistics_seed>:<metric_key>")` 前 8 字节的大端无符号整数，
不能使用进程相关的 Python `hash()`。

### 8.4 失败与分母

每个预定 sample ID 都必须出现在 `per_sample`：

- 成功页：`status=valid` 并携带指标；
- 失败页：`status=failed`、阶段、错误类型和错误摘要，所有指标为 unavailable；
- 每个 stratum 同时报告 expected、valid、failed；
- 每个聚合 metric 使用自己的 available 分母，metric unavailable 不等于页面 failed；
- `per_sample` keys 必须等于 `dataset.expected_ids`，`failures` 必须与 failed 记录完全一致。

允许作为逐页失败捕获的 stage 只有 `decode` 和 `ir_validation`。数据、hash、checkpoint、
模型加载和运行环境预检失败时整次评测中止；未预期的 metric 异常也中止，不能降级成
普通 unavailable。标签未采集或没有可匹配布局属于 metric unavailable，不是异常。

数据预检失败时没有“每页都必须出现”的要求，因为运行尚未开始；命令必须返回非零。

### 8.5 推断比较

`compare_intent_results.py` 在 v2 Reference 结果上：

- 仅允许 `ai_silver` 执行 bootstrap、配对置换和效应量；
- Human Gold 和 mixed 的推断请求直接拒绝；
- 两种方法的 prediction-valid cohort 不一致时整次比较 fail closed；
- 某 metric 的 available ID 集合不一致时，仅该 metric 输出
  `unavailable/metric_cohort_mismatch`，不能静默取交集；
- 只对上表三个预注册 Holm family 分别校正；不可用假设以机械 `p=1` 占据预注册
  family 大小，但其报告状态仍为 unavailable；
- lower-is-better 指标统一转换为“正差值表示方法改进”；
- 输出 `intent-comparison/v2`，至少包含 expected IDs、双方 valid/failed IDs、每项指标
  paired IDs、`n_paired`、方向化 improvement、bootstrap CI、Cohen's dz、原始 p 值和
  Holm p 值。

比较器的低样本规则固定为：`n_paired=0` 时整项 unavailable，reason 为
`no_paired_metric_pages`；`n_paired=1` 时只提供描述性 mean improvement，CI、置换检验
和 dz 为 unavailable，reason 为 `insufficient_paired_metric_pages`；`n_paired>=2` 才
执行配对 bootstrap 和置换检验。若 paired differences 的样本标准差为 `0`，dz 为
unavailable，reason 为 `zero_paired_difference_variance`。这些不可用假设仍以机械
`p=1` 占据 Holm family，comparison JSON 同样使用 `allow_nan=False`。

未来如需给失败页计分，必须先单独预注册规则；本阶段不得以交集删除代替失败处理。

三随机种子结果不能把 `18 x 3` 当作 54 个独立页面。后续聚合时先对每页跨 seed
求均值，再以 18 页为抽样单位 bootstrap，并另报 seed 间波动。

## 9. 运行审计

训练 checkpoint 本体至少记录：

- Git commit；
- 完整生效配置；
- assignment、selection manifest、AI reference manifest 和原 split 的 SHA-256；
- split x tier 计数；
- 原始与归一化 tier 权重；
- 随机种子；
- Python、PyTorch、Transformers 版本和设备；
- 训练起止时间；

数据快照元信息在 Dataset 初始化后冻结并写入 checkpoint，不能只依赖运行结束时重新
读取磁盘文件。checkpoint 序列化完成后，训练器在 `run_manifest.json` sidecar 中记录
checkpoint 路径与 SHA-256；checkpoint 不能包含自身哈希。评测结果记录其实际读取的
checkpoint SHA-256。

评测结果另行记录预定页、失败页和失败阶段；这些字段不写入训练 checkpoint。

Reference test 首次进入模型训练/评测消费入口时，CLI 先完成配置、JSON、checkpoint、
模型加载，以及只检查 manifest 语义、记录哈希格式、规范路径、文件存在性和 manifest
hash 的元数据
预检；该预检不打开或哈希 active test 资源和 IR。预检成功后、完整 Dataset 打开 test
IR 或执行 test 推理之前，使用排他创建写入固定路径
`outputs/intent-reference-v1/test_unseal.json`。该文件独立于方法和单次结果目录，包含
时间、Git commit、assignment hash 和 checkpoint hash；已存在时读取并沿用首次时间，
禁止覆盖。完整 Dataset 本身必须拒绝未解封的正式 test，且预检与完整构造的 manifest
hash 必须一致，不能只依赖 CLI 调用顺序。

该门禁不是开发者盲法声明。Reference 构建、标注工作台和包完整性测试已经读取过标注，
且 AI Silver 由同一研究工作流生成；这些操作不产生模型指标，也不得用于模型、阈值或
排除规则选择。使用临时合成 fixture 的 test 不创建正式 unseal。边界定义与已知访问
事件见 `docs/thesis/19_reference_test_access_audit.md`。

## 10. 代码边界

预计修改或新增：

- `src/data/intent_dataset.py`：Reference Dataset、共享样本构造、监督掩码；
- `src/training/intent_losses.py`：页级损失与来源权重；
- `scripts/train_intent.py`：Dataset 路由、权重、分层 validation、审计和梯度累积；
- `src/design_intent/metrics.py`：Token 开关、布局不可用语义和覆盖率；
- `src/design_intent/evaluation.py`：v2 纯汇总逻辑；
- `scripts/evaluate_intent.py`：Reference CLI 与 v2 输出；
- `scripts/compare_intent_results.py`：tier 限制和 cohort 防护；
- `configs/intent/*.yaml`：有效路径和 Reference 消融配置；
- 对应 Dataset、loss、evaluation 和配置回归测试；
- `docs/thesis/06_experiment_protocol.md`、`07_implementation_status.md`：实现状态和运行命令。

本阶段不修改模型 head、约束求解器、标注文件、Figma 导出器，不接入视觉指标，也不
实现三 seed 聚合器或运行正式多 seed 训练。正式训练与跨 seed 聚合属于该链路通过
smoke 和完整测试后的下一阶段。

## 11. 测试与验收

### 11.1 Dataset

- 精确读取 `30/10/20`，tier 分布为 `7+23 / 1+9 / 2+18`；
- batch size 2 时仍保留 tier、来源和监督可用性；
- assignment、selection、Pilot split 或 provenance 任一冲突时失败；
- 路径越界、assignment 资源路径与规范路径不一致时失败；
- test ID 不能进入 train；
- 旧 `IntentDataset` 加载/API 保持兼容；目标语义升级由 v2 测试明确覆盖。

### 11.2 Loss

- Human Token loss 严格零参与，改变 Human Token logits 不改变总损失或梯度；
- 合成双页面证明 Human:Silver annotation gradient 相对贡献为 `1:0.5`；
- 全部 train 页的 normalized tier weight 均值为 `1`；
- tier 权重不影响 alignment loss；
- 全部任务无标签时有限值且无 NaN；
- 无标签任务返回与 logits 计算图相连的零；
- batch size 1 和 batch size 2 均满足相同页级定义；
- micro-batch 页面数 `[2,2,1]` 与单批 `[5]` 的确定性 mock 梯度一致；
- Human 未选择节点 Action 梯度为零，显式 ATOMIC/GROUP 正例梯度非零；
- root 直属元素不产生伪正 group pair；不可表示父边不产生伪 root 监督。

### 11.3 Evaluation

- Human Token 全部为 `null/unavailable`；
- Human aggregate 字段存在，但 mean、SD、CI 均为 `suppressed_by_protocol`；
- AI Silver Token 保留且具有 proxy scope；
- mixed 只有非 Token 描述均值，inference 状态为 `prohibited`；
- 固定 seed 的 bootstrap 完全确定；
- 失败页不从 per-sample 消失；
- 每项指标满足固定分母不变量；
- 参考布局非空但无共同布局时条件指标 unavailable 且 coverage 为 0；
- `json.dumps(result, allow_nan=False)` 成功；
- 比较器拒绝 Human/mixed 和不一致的有效 cohort；metric cohort 不一致只阻断该指标。

### 11.4 配置与闭环

- 所有静态数据、manifest 和模型资源路径存在；生成型 init checkpoint 作为运行前置
  条件，在实际训练启动时 fail closed；
- weak、Human-only、Silver-only 和三种完整 Reference 权重均能构造非空 train/val；
- 通过 override 注入临时 mock checkpoint，完成 1 epoch Reference smoke；
- 使用临时合成 Reference fixture 的 test split 生成符合 v2 schema 的报告，不读取真实
  20 页 Reference test；
- 全部现有测试和新增测试通过；
- 文档不把 AI Silver、单人 Human Gold 或 mixed test 写成人工总体真值。

## 12. 完成定义

只有同时满足以下条件，本阶段才可标记完成：

1. Reference 数据无需复制即可进入训练与评测；
2. Human Gold 缺失标签不会产生负监督或伪指标；
3. AI Silver 权重在 batch size 大于 1 时仍具有严格页级语义；
4. validation/test 不使用训练 tier 权重；
5. Human、Silver 和 mixed 输出物理分层且带清晰解释范围；
6. 失败页、不可用指标和布局覆盖均有机器可读表示；
7. 配置、数据哈希、权重、seed 和 test 解封时间可审计；
8. 测试通过后才进入正式多 seed 训练和论文结果生成。
