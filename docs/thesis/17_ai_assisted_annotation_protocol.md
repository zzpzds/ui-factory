# AI 预标注与设计师校正协议

## Material Passport

- `artifact_type`: human-in-the-loop annotation protocol
- `artifact_id`: `formal-gold-ai-assist-20260814`
- `verification_status`: `IMPLEMENTED`
- `input_package`: `data/annotations/intent_gold_v1`
- `ai_prelabels`: 50
- `ai_assisted_condition`: 45
- `blind_control_condition`: 5
- `independent_existing_gold`: 10
- `ai_used_as_gold`: false

## 研究决策

新增 50 页不再全部采用从零人工标注，而改为 AI 预标注后由真实设计师逐项校正。
该方案降低候选发现和重复录入成本，但会产生锚定偏差，因此最终数据不能表述为
“50 页独立人工盲标”。

AI 初稿始终独立保存在：

```text
data/annotations/intent_gold_v1/annotator_ai_prelabel/<sample_id>.json
```

设计师只能修改 `annotator_a` 草稿，原始 AI 文件只读并使用 SHA-256 绑定。AI 初稿
不是第二位真人，也不能直接复制为 `gold_intent.json`。

## 条件分配

| 条件 | 数量 | 人工起点 | 用途 |
| --- | ---: | --- | --- |
| AI 辅助校正 | 45 | 高置信 AI 初稿 | 主体成本降低 |
| 盲标对照 | 5 | 空白草稿，AI 文件隐藏 | 估计时间差与锚定偏差 |
| 已有独立 Gold | 10 | 已冻结 | 历史基准与生成器回放 |

盲标对照按 split 分层为 train 3、validation 1、test 1，并优先覆盖不同页面类型与
尺寸。固定样本为 `0165`、`0304`、`0832`、`1035`、`1365`。这 5 页在人工提交
前不得解封 AI 初稿。

## AI 初稿范围

生成器 `codex_pagegraph_preannotation_v1` 只读取 PageGraph，不读取人工标签，也不把
截图像素输入生成算法。它执行：

1. 保留置信度不低于 0.75 的原子元素；
2. 保留置信度不低于 0.85 的语义或显式布局容器；
3. 删除只包装一个原子元素的无效分组；
4. 根据保留实体的 DOM 祖先关系重建初始树；
5. 预填分组布局，但不直接创建样式 Token；
6. 写入输入资源哈希、生成器版本、限制和 `human_labels_viewed=false`。

在 8 个既有 train/validation Gold 上的冻结回放为：原子元素 F1 0.836、分组
B-cubed F1 0.756、父边 F1 0.395、布局模式准确率 0.688。test Gold 不参与这一步
校准。父边结果明显偏低，因此层级必须被列为人工重点复核项，而不是默认接受项。

## 设计师 SOP

1. 确认页头状态为“AI 预标注”或“盲标对照”；
2. 对照截图检查原子元素的误检、漏检、拆分和合并；
3. 检查每个分组的语义角色、直接子项、父级和顺序；
4. 检查布局方向、间距、内边距、对齐和 resize；
5. 在 Token 页签保留或忽略候选，并点击“完成 Token 检查”；
6. 勾选“设计师已完成逐项复核”；任何后续编辑都会撤销该确认；
7. AI 辅助页提交为 `human_corrected_ai_preannotation`，盲标页提交为
   `human_annotation`；
8. 校验无错误后提交，禁止直接把 AI 初稿视为完成结果。

## 成本与偏差分析

平台在页面可见期间累计 `human_active_seconds`。最终冻结后比较：

- 两种条件的单页人工前台时长中位数和四分位数；
- 相对 AI 初稿的原子元素新增、删除、合并和类型修改数量；
- 分组、父边、布局和 Token 的修改数量；
- AI 辅助组与盲标组最终结果对隐藏 AI 初稿的相似度差异。

由于盲标对照只有 5 页，统计结果以描述性效应量和置信区间为主，不把不显著解释
为“没有锚定偏差”。页面规模、页面类型和实体数量同时作为分层描述变量。

## 解释边界

- 最终 Gold 的有效性来自设计师校正与确认，不来自 AI 初稿本身；
- AI 辅助组不能用于报告独立人工标注者间一致性；
- AI 与最终 Gold 的高相似度既可能表示预标注准确，也可能包含锚定效应；
- 盲标对照只用于标注流程分析，不用于模型或阈值选择；
- 所有论文表格必须区分 `existing_independent_gold`、`ai_assisted` 和
  `blind_control` 三种来源。

## 复现命令

```bash
source .venv/bin/activate
python scripts/prepare_ai_assisted_gold.py
python -m pytest tests/test_formal_gold_package.py -q
python scripts/serve_intent_annotation.py \
  --package_dir data/annotations/intent_gold_v1 \
  --port 8766
```
