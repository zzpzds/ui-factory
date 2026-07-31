# 实验协议 v1

## 实验目的

主实验回答完整双流模型是否比启发式转换、旧模型和单模态模型更准确地恢复可编辑设计意图。

## 数据冻结

### Pilot

- 随机请求 500 页，成功获得 492 个有效弱标注页面；
- 去重后有效独立页面估算为 410；
- 已创建 10 页双人试标注包，当前等待人工完成；
- 目的：验证失败率、标注耗时、显存和指标行为；
- Pilot 结果不进入论文主结果表。

完整生产结果见 `11_pilot_data_report.md`。

### Main

- `D_weak`：去重后 5,000-10,000 个独立页面；
- `D_gold_train`：120 页；
- `D_gold_val`：40 页；
- `D_gold_test`：40 页；
- 测试集只在模型、阈值和排除规则冻结后运行。

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
| B3 | 代码单模态 Intent 模型 | 弱 + gold train |
| B4 | 视觉单模态 Intent 模型 | 弱 + gold train |
| B5 | 双流、无显式对齐 | 弱 + gold train |
| M | 完整双流模型 | 弱 + gold train |

所有学习方法共享：

- 相同候选节点；
- 相同数据切分；
- 相同训练轮次或早停规则；
- 相同约束求解器；
- 相同金标准微调集。

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
- Token B-cubed F1；
- 重渲染 SSIM。

主指标不合并成单一总分。综合可编辑性分数仅用于可视化，并在实验前冻结权重。

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
- 报告均值和标准差；
- 页面级 bootstrap 10,000 次计算 95% CI；
- M 与最强基线做双侧配对置换检验；
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

另做训练数据规模：500、1k、5k、10k；金标准微调：0、40、120。

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

# 4. 评测模型
python scripts/evaluate_intent.py \
  --checkpoint outputs/intent-full-v1/best.pt \
  --annotation_name gold_intent.json

# 5. 评测强启发式
python scripts/evaluate_intent_baseline.py \
  --data_dir data/gold \
  --gold_name gold_intent.json \
  --split_manifest data/gold_split.json \
  --split test

# 6. 页面级配对统计
python scripts/compare_intent_results.py \
  --method outputs/intent-evaluation.json \
  --baseline outputs/intent-baseline.json
```

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
- 测试集解封时间。
