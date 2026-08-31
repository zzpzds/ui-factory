# 实验协议 v1

## 实验目的

当前主实验比较完整双流模型、启发式转换、旧模型和单模态模型的工程表现及代理标签
一致性。除非补充独立真人测试集，否则它不能充分回答“相对人类设计意图是否更准确”。

## 数据冻结

### Pilot

- 随机请求 500 页，成功获得 492 个有效弱标注页面；
- 去重后有效独立页面估算为 410；
- 10 页由同一名人工标注者完成底稿并结合 AI 差异提示定稿；其设计资历、与作者
  关系及利益冲突状态未记录，因此只作为探索性 Human Gold；
- 目的：验证失败率、标注耗时、显存和指标行为；
- Pilot 结果不进入论文主结果表。

完整生产结果见 `11_pilot_data_report.md`。

### Main

- `D_weak_pilot`：492 个成功渲染页面，独立页面估算 410；
- `D_reference_train`：30 页（Human Gold 7 + AI Silver 23）；
- `D_reference_val`：10 页（Human Gold 1 + AI Silver 9）；
- `D_reference_test`：20 页（Human Gold 2 + AI Silver 18）；
- 测试集只在模型、阈值和排除规则冻结后运行。

`D_reference_test` 不是 20 页人工测试集。AI Silver 和 Human Gold 必须分别汇总；
Human Gold `n=2` 只逐页描述，不能支撑稳定的总体性能结论。AI Silver `n=18` 的
结构与布局指标只衡量代理一致性；禁止对混合 20 页做单一推断。

数据切分命令：

```bash
python scripts/split_intent_dataset.py \
  --data_dir data/processed \
  --output data/intent_split.json \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --seed 42
```

同站点优先作为一个分组；缺少站点信息时按页面结构指纹分组。最终主实验还需对截图近重复做一次人工抽查。

```bash
python scripts/detect_near_duplicates.py \
  --data_dir data/processed \
  --split_manifest data/intent_split.json \
  --output outputs/near-duplicate-report.json
```

## 比较方法

| ID | 方法 | 训练数据 |
| --- | --- | --- |
| B0 | DOM 全量直接映射 | 无 |
| B1 | `weak_supervision_v1` 强启发式 | 无 |
| B2 | 旧 `FigmaGenerationModel` | 弱标签 |
| B3 | 代码单模态 Intent 模型 | 弱 + tier-aware Reference train |
| B4 | 视觉单模态 Intent 模型 | 弱 + tier-aware Reference train |
| B5 | 双流、无显式对齐 | 弱 + tier-aware Reference train |
| M | 完整双流模型 | 弱 + tier-aware Reference train |

所有学习方法共享：

- 相同候选节点；
- 相同数据切分；
- 相同训练轮次或早停规则；
- 相同约束求解器；
- 相同 Reference train，并保持 Human Gold / AI Silver 样本权重一致。

## 主指标

每个页面计算后再汇总：

- Leaf Preservation F1；
- Group Pairwise F1；
- Group B-cubed F1；
- Group Match F1；
- Parent Edge F1；
- Normalized Tree Edit Distance；
- Layout Mode Macro-F1；
- Gap/Padding MAE；
- Token B-cubed F1（仅 AI Silver capped computed-style proxy 诊断）；
- 重渲染 SSIM。

主指标不合并成单一总分。综合可编辑性分数仅用于可视化，并在实验前冻结权重。
Human Gold 未获得可审计的 token 正例或显式负例，因此其 token 指标记为 N/A，
不把空 token 列表当作完美负例。

## 辅助诊断

- 预测 IR 格式失败率；
- candidate oracle recall；
- 对齐 AUPRC；
- layer compression；
- redundant wrapper ratio；
- 自动编辑任务通过率；
- 按页面类型、节点数和 DOM-视觉冲突程度分层。

## 随机性与统计

- 随机种子：`42`、`123`、`2026`；
- AI Silver `n=18` 单独报告均值、标准差、效应量和页面级 bootstrap 95% CI，明确
  低统计功效；M 与最强基线的双侧配对置换检验只作探索性分析；
- Human Gold `n=2` 仅逐页报告，不计算 CI 或 `p` 值；
- 禁止对来源混合的 20 页 test 做单一 bootstrap 或显著性检验；
- 多指标比较使用 Holm 校正；
- 同时报告效应量；
- `alpha = 0.05`。

## 失败处理

以下情况不得静默删除：

- 页面渲染失败；
- 页面图为空；
- 超过 `max_nodes`；
- IR 求解失败；
- Figma/HTML 导出失败；
- 重渲染失败。

每类失败单独报告数量和比例。截断样本仍参与主实验，但需报告 candidate oracle recall；严重截断页面另做敏感性分析。

## 消融矩阵

| 实验 | 视觉 | 代码文本 | 结构样式 | DOM/空间图 | 显式对齐 | 约束求解 |
| --- | --- | --- | --- | --- | --- | --- |
| M | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| A1 |  | ✓ | ✓ | ✓ |  | ✓ |
| A2 | ✓ | ✓ |  | ✓ | ✓ | ✓ |
| A3 | ✓ | ✓ | ✓ |  | ✓ | ✓ |
| A4 | ✓ | ✓ | ✓ | ✓ |  | ✓ |
| A5 | ✓ | ✓ | ✓ | ✓ | ✓ |  |

另做训练数据规模：500、1k、5k、10k；Reference 微调做 0 页、仅 7 页 Human
Gold、23 页 AI Silver、完整 30 页四档。AI Silver 权重至少比较 0.25、0.5、1.0。

## 运行顺序

```bash
# 1. 生成 PageGraph 与弱标签
python scripts/prepare_intent_data.py --data_dir data/sampled

# 2. 校验
python scripts/validate_intent_data.py \
  --data_dir data/sampled \
  --annotation_name weak_intent.json

# 3. 训练
python scripts/train_intent.py --config configs/intent/full.yaml

# 4. 当前先校验 Reference v1；接入训练 loader 后按 reference_tier 分层评测
python -m pytest tests/test_formal_gold_package.py -q

# 5. 后续先接入 reference_tier-aware loader，再运行基线和模型评测
```

现有评测入口默认按 `gold_intent.json` 扫描，尚不能安全消费双层 Reference v1。
在来源感知 loader 与分层汇总测试完成前，不得把 `reference/` 批量改名或复制成
`gold_intent.json` 来绕过该门槛。

## 结果冻结要求

主结果生成前必须记录：

- Git commit；
- 数据 manifest hash；
- 配置文件；
- checkpoint hash；
- Python、PyTorch、Transformers 版本；
- GPU 型号；
- 训练时长；
- 三个随机种子；
- 测试集解封时间；
- Human Gold 与 AI Silver 的独立指标文件；
- AI Silver 训练权重与无 Silver 消融结果。
