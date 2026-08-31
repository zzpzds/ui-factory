# AI 多视角银标生成与披露协议

## Material Passport

- `artifact_type`: AI reference annotation protocol
- `artifact_id`: `ai-multiview-silver-v1-20260831`
- `verification_status`: `PIPELINE_IMPLEMENTED_AND_SCHEMA_VALIDATED`
- `target_samples`: 50
- `generator`: `codex_multiview_reference_v1`
- `human_review_performed`: false
- `human_labels_viewed_for_targets`: false
- `ai_labels_treated_as_human`: false
- `reproducibility_scope`: frozen-artifact replay only

该状态只覆盖 pipeline 实现、schema、引用和冻结输入后的哈希稳定性，不覆盖 AI
Silver 相对人类设计意图的标签有效性。

## 目的

在人力不足的约束下，为 50 个待标页面生成可追溯、可校验的设计意图参考标签，供
训练、开发和来源分层评测使用。输出是 **AI Silver**，不是人工 Gold。

## 输入隔离

每个目标页允许读取：

- `screenshot.png`：全分辨率页面视觉；
- `page_graph.json`：可见节点、bbox、文本、DOM 关系和 computed style；
- 两条由 PageGraph 产生的结构候选。

目标页不读取任何人工标签。固定 `0.75/0.85` 规则曾在 8 页 train/validation
Human Gold 上做诊断性回放，但代码只计算指标，没有据此选择或更新阈值。2 页 test
Human Gold 未参与该回放；这不等于作者开发后续流程时对它们完全盲法，开发者接触
Human Gold 的状态记为 `not_guaranteed`。

## 生成阶段

1. **保守结构候选**：`codex_pagegraph_preannotation_v1` 保留高置信原子和明确语义
   容器，降低过分组；
2. **扩展结构候选**：`expanded_pagegraph_structure_candidate` 保留更多通用容器，
   为视觉上层级明显但保守候选遗漏的页面提供高召回路径；
3. **截图语义审阅**：AI 视觉角色检查页面质量、页面类型、可见区域、候选与截图的
   一致程度及层级风险，并选择 conservative 或 expanded；
4. **显式视觉修正**：需要补充的分组以 `create_group` 操作记录，执行器验证所有实体
   引用、重挂父边、生成 bbox 和布局，并重算后代原子覆盖；
5. **紧凑 Token proxy**：先应用视觉结构修正，再从最终结构的 computed style 和
   布局中提取至少 3 个成员的复用关系；每页最多 COLOR 4、TEXT 4、RADIUS 2、
   SPACING 2，共 12 个；该结果不是人类设计系统 token 真值；
6. **确定性门禁**：运行 schema、bbox、源节点唯一性、树、group、layout、Token
   引用与 SHA-256 校验，只有零错误页面写入 `reference/`。

当前 50 页输出共含 1,573 个原子元素、447 个分组和 245 个 Token；35 页选择保守
结构，15 页选择扩展结构。50 页 AI Silver 全部有 Token，49 页至少有一个 group。逐页数值以
`ai_reference_manifest.json` 为准。

10 页 Human Gold 均没有可审计的 Token 正例、显式负例或 `token_review` 历史。
因此 Human Gold 不报告 Token F1；AI Silver Token 只衡量与 capped computed-style
聚类规则的一致性，也不能用来排除模型直接利用 `X_style` 的 shortcut。

## 质量与风险记录

视觉审阅把层级风险记录为 low 10、medium 21、high 19；结构对齐记录为 strong 18、
moderate 29、weak 3。36 页以限制条件接受，14 页直接接受结构候选。这些判断是
流程内诊断，不是与人类真值比较得到的准确率。

历史固定规则在 8 页 Human Gold 上的诊断性回放为：`group_pair_f1=0.6234`、
`group_match_f1=0.5395`、`parent_f1=0.3952`。`token_f1=1.0` 不可解释，因为这些
Human Gold 没有可审计 token 标签且 coverage 为 0。结构回放数值表明 Silver 的
group/tree 质量存在实质风险，相关结果只能作为代理一致性或 pipeline 回归指标。

所有视觉角色、裁决角色和生成角色属于同一 Codex 工作流。角色分离用于减少遗漏和
保留审计链，但不提供模型独立性，不能计算或宣称人类标注者间信度、跨模型一致性或
独立多评审共识。

## Provenance 契约

每个 AI Silver 文件必须包含：

- `reference_tier=ai_silver`；
- `label_source=ai_multiview_silver`；
- `annotator_kind=ai_agent_pipeline`；
- `human_review_performed=false`；
- `human_labels_viewed_for_target=false`；
- `visual_input_used=true`、`page_graph_used=true`；
- 四类 `pipeline_inputs` 的相对路径与 SHA-256；
- 截图和 PageGraph 的路径与 SHA-256；
- 候选选择、视觉风险、限制和生成器版本。
- 固定规则诊断回放 artifact 的路径、SHA-256 与样本 ID；
- `generator_threshold_selection_used_human_gold=false` 和
  `developer_human_gold_blinding_status=not_guaranteed`。

AI 文件不得写入 `gold_intent.json`，不得使用 `human_annotation`、
`human_corrected_ai_preannotation` 或其他人工来源名。

## 允许的实验使用

- 作为来源感知训练集，建议对 AI Silver 降权并做无 Silver 消融；
- 用于开发阶段的结构错误分析和 pipeline 回归；
- 用于测量模型与当前 AI 参考标签的一致性；
- 与 10 页 Human Gold 分层报告，不合并解释。
- Token 仅作为 capped computed-style proxy 的工程回归诊断。

## 禁止的实验使用

- 作为“60 页人工测试集”报告模型准确率；
- 用 50 页 AI Silver 计算人工标注者一致性；
- 以同源 AI 评测支持“达到设计师水平”的结论；
- 把混合 test 的总指标作为相对人类设计判断的主结论；
- 把 8 页诊断性回放写成阈值校准，或把 2 页未回放样本写成全流程盲法 test；
- 在 Human Gold 上报告 Token F1，或用 Silver Token 证明恢复了人类复用意图；
- 对来源混合的 20 页 test 做单一显著性检验。

## 论文披露模板

> 新增 50 页由同一模型家族的多阶段 AI 标注流程基于截图与 PageGraph 生成，
> 未经过人工逐项复核，因此定义为 AI 银标准；10 页由单名人工标注者以人工底稿
> 结合 AI 差异提示后定稿，单独保留为探索性人类金标准。该人员的设计资历及与作者
> 关系未归档，流程也不是独立裁决。基于银标准的结果衡量模型与 AI 参考标签的一致性，不等同于
> 相对人类设计判断的准确性或人类标注者间信度。Human Gold 未采集可审计 Token
> 标签，因此 Token 结果仅限 AI computed-style proxy 工程诊断。

## 复现命令

```bash
source .venv/bin/activate
python scripts/prepare_ai_reference_gold.py
python -m pytest \
  tests/test_ai_reference_annotations.py \
  tests/test_formal_gold_package.py -q
```

上述命令必须得到相同的 assignment、manifest 和参考文件哈希，但这一保证只覆盖
冻结 `ai_visual_reviews/*.json` 之后的 artifact replay。原视觉会话未记录具体模型
版本、提示词/协议哈希、采样参数或 session ID，无法端到端重生成 50 页视觉判断；
这是当前 Reference v1 的明确可复现性限制。
