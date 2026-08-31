# DesignIntent Reference v1 采样与数据记录

## Material Passport

- `artifact_type`: reference-dataset sampling and execution record
- `artifact_id`: `design-intent-reference-v1-20260831`
- `verification_status`: `ARTIFACT_INTEGRITY_VERIFIED_WITH_LIMITATIONS`
- `source_pool`: 492 个成功渲染的 Intent Pilot 页面
- `output`: `data/annotations/intent_gold_v1`
- `canonical_annotations`: `data/annotations/intent_gold_v1/reference`
- `human_gold`: 10 页单人 AI 辅助定稿的人工参考
- `ai_silver`: 50 页 AI 多视角银标
- `ai_labels_treated_as_human`: false
- `weak_labels_used_as_truth`: false

该状态只表示清单、结构、哈希和冻结工件回放已校验，不表示 AI Silver 相对人类
设计判断的正确性已经验证，也不表示来源许可治理已经完成。

## 定位

本数据集的规范名称为 **DesignIntent Reference v1**。目录名 `intent_gold_v1`
仅为兼容既有脚本而保留，不表示其中 60 页都是人工金标准。

数据集包含两个不可合并的标签层级：

| 标签层级 | 数量 | 来源 | 可支持的结论 |
| --- | ---: | --- | --- |
| Human-reviewed Gold | 10 | 同一名人工标注者先做底稿，再结合 AI 差异提示定稿；资历及与作者关系未记录 | 探索性人类参考评测 |
| AI multiview Silver | 50 | PageGraph 双结构候选、截图语义审阅、自动 Token 与确定性校验 | 训练、开发和模型对 AI 参考标签的一致性评测 |

新增 50 页没有经过真人逐项复核，不能写成“人工标注”“设计师标注”或 Gold。
同一 Codex 模型家族承担的多个角色也不能写成独立标注员或标注者间一致性。

Human Gold 的技术 `label_source` 仍为历史名称 `human_ai_adjudicated_gold`，但该
过程不是独立裁决：`reviewer_independent_of_base_annotator=false`。公开 reference
使用稳定编号 `human_reviewer_01` 并移除姓名；设计资历、与作者关系、利益冲突和
公开姓名同意状态均未归档。7/10 审核说明只有“以人工为准”一类简短摘要，因此
现有证据只支持“提交审核声明并生成最终文件”，不支持逐差异决策均有审计记录。

## 固定切分

Reference v1 沿用 `data/intent_pilot_split.json` 的既有归属：

| split | 总数 | Human Gold | AI Silver | 用途 |
| --- | ---: | ---: | ---: | --- |
| train | 30 | 7 | 23 | 来源感知训练与消融 |
| validation | 10 | 1 | 9 | 阈值和停止条件选择；按来源分层 |
| test | 20 | 2 | 18 | 冻结评测；不得只报告混合总分 |

固定结构规则曾在 8 页 train/validation Human Gold 上做诊断性回放，但没有据此
选择或修改 `0.75/0.85` 阈值。2 页 Human Gold test 只能称为“未参与该回放”；作者
开发后续 expanded 与视觉流程时可能接触过 Human Gold，不能声称全流程盲法。因此
人类参考结果只能逐页描述，不能声称获得稳定的人类泛化性能估计。

该回放本身也暴露了结构代理风险：`group_pair_f1=0.6234`、
`group_match_f1=0.5395`、`parent_f1=0.3952`。这些数值是固定规则相对 8 页既有
Human Gold 的诊断，不是新模型结果，但说明 AI Silver 的 group/tree 只能作为代理
一致性和回归检测，不能独立证明结构恢复准确。

## 样本筛选

筛选器只使用成功渲染且具备截图和 PageGraph 的页面，并执行节点规模、截图对比度、
内容安全、破图启发式、截图近重复簇和结构指纹去重。页面类型覆盖表格、表单、电商、
列表、导航、内容和通用页面。样式富集只影响采样机会，不把重复 computed style
直接当作标签。

完整筛选规则和资源哈希位于：

- `selection_manifest.json`：60 页固定清单与 split；
- `visual_exclusions.json`：视觉排除记录；
- `quality_replacement_manifest.json`：质量替换审计；
- `ai_reference_manifest.json`：50 页 AI 银标的逐页来源和校验结果。

全分辨率视觉复核后完成三次同 split 替换：

| 被替换页 | 替代页 | split | 原因 |
| --- | --- | --- | --- |
| `0144` | `0433` | validation | 导航、正文和版权区域大面积错位重叠 |
| `0553` | `0584` | validation | 正文持续推广赌场、老虎机与博彩策略 |
| `1365` | `1530` | train | 导航重复错位、内容碰撞且缺少稳定主体区域 |

替换后共 60 页，规模分层为 small 20、medium 31、large 9；另保留 18 页备用样本。
被替换页保留在审计记录中，不进入当前训练或评测清单。

## 生成与复现

```bash
source .venv/bin/activate

# 从初始 Formal 包执行已审计的质量替换
python scripts/replace_formal_reference_samples.py

# 生成 10 页 Human Gold 引用和 50 页 AI Silver
python scripts/prepare_ai_reference_gold.py

# 校验来源、哈希、结构、Token 和只读状态
python -m pytest \
  tests/test_ai_reference_annotations.py \
  tests/test_formal_gold_package.py -q

# 只读查看
python scripts/serve_intent_annotation.py \
  --package_dir data/annotations/intent_gold_v1 \
  --port 8766
```

工作台显示 `人工复核 Gold` 或 `AI 多阶段银标`，全部 60 页均为只读。AI Silver
显示“自动生成”Token；Human Gold 因没有可审计的 Token 复核记录，显示“未采集/
不可评估”。界面不显示人工候选确认、复核计时或提交按钮。

## 使用边界

1. 训练时必须保留 `reference_tier`，并报告仅 AI Silver、混合 Reference、仅
   Human Gold 的消融；
2. validation/test 指标必须按 `human_ai_adjudicated_gold` 与
   `ai_multiview_silver` 分层，混合指标只能作为补充；
3. AI Silver 上的结果衡量模型与 AI 参考标签的一致性，不等同于相对人类设计判断
   的准确率；
4. 不报告 50 页的人工标注耗时、人工一致性或人机锚定效应，因为没有发生对应人工
   流程；
5. 论文必须把 Human Gold 样本量过小和 AI 标签同源偏差列为主要构念效度威胁；
6. 若后续取得具备可审计资历的人员资源，只能新增独立审核层，不能回写 provenance 把现有
   AI Silver 改名为人工标签。
7. Human Gold 不报告 Token F1；AI Silver Token 是每页最多 12 个的 capped
   computed-style proxy，只报告同源规则一致性，不与人工协议直接合并；
8. Human Gold `n=2` 只逐页描述，AI Silver `n=18` 的 CI/检验注明低功效；不得对
   来源混合的 20 页 test 做单一显著性检验；
9. 当前生成命令可逐字节复现冻结视觉审阅 JSON 之后的确定性组装，但不能重现当时
   的 AI 视觉判断会话，因为模型版本、提示词哈希、采样参数和 session ID 未记录；
10. 数据集版本、许可、获取时间和逐页来源治理字段仍须在主实验冻结前补齐。

AI 生成细节和披露模板见 `18_ai_multiview_silver_protocol.md`。
