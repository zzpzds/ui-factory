# AI 预标注与设计师校正协议（历史方案）

## 状态

- `artifact_id`: `formal-gold-ai-assist-20260814`
- `status`: `SUPERSEDED_BEFORE_HUMAN_COMPLETION`
- `superseded_by`: `18_ai_multiview_silver_protocol.md`
- `historical_manifest`: `data/annotations/intent_gold_v1/ai_assistance_manifest.json`

本文件记录曾计划执行、但因人力条件变化而在新增 50 页完成人工复核前终止的方案。
它不得作为论文中“已执行的方法”引用。

## 原计划

原计划把新增 50 页分为 45 页可见 AI 预标注和 5 页隐藏 AI 初稿的盲标对照，要求
真实设计师逐项校正原子元素、分组、父子层级、布局和 Token，记录前台复核时长，
最后提交为 `human_corrected_ai_preannotation` 或 `human_annotation`。

该计划拟比较两种条件的人工时长、修改量和对 AI 初稿的相似度，用于观察成本与
锚定效应。由于没有完成这 50 页的真实设计师逐项复核，上述比较没有产生可分析的
人类研究数据。

## 实际保留的历史产物

- `annotator_ai_prelabel/`：PageGraph 保守结构候选；
- `annotator_a/`：当时初始化的草稿，不是已完成人工标签；
- `ai_assistance_manifest.json`：原条件计划和固定规则诊断性回放，现已标记
  `superseded`，并显式记录人工复核与盲标完成数均为 0；
- 旧的 45/5 条件只用于审计，不再决定当前标签来源。

不得从草稿存在、旧 assignment 状态或 AI 文件可见性推断人工已完成。当前规范标签
只读取 `reference/` 和 `ai_reference_manifest.json`。

## 禁止表述

- “新增 50 页均由设计师校正完成”；
- “5 页盲标对照已经用于锚定效应实验”；
- “AI 预标注显著降低了人工时长”；
- “60 页均为人工 Gold”；
- “同一模型家族的多个审阅角色等同于多名标注者”。

## 与当前流程的关系

当前流程复用了保守结构候选作为一个输入，但新增了 expanded PageGraph 结构候选、
全分辨率截图语义审阅、显式视觉修正操作和紧凑 Token 生成。50 页输出定义为
`ai_multiview_silver`，没有人工逐项复核，也不改写成人工来源。
