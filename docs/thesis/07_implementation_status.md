# 新主链路实现状态

## 已完成

- Page Implementation Graph v1 数据契约；
- Design Intent IR v1 数据契约；
- Playwright 富结构页面抽取；
- 旧数据迁移；
- 带置信度的弱标签教师；
- IR 与页面图校验器；
- bbox 引导的双流融合；
- PageGraphEncoder；
- Atomic、Group、Role、Layout、Token、Tree heads；
- 多任务损失；
- 约束求解器；
- Figma-like JSON 导出；
- 设计意图核心指标；
- 训练与评测入口；
- 强启发式评测入口；
- 站点/结构隔离切分；
- 人工标注空模板与批量校验；
- 自动编辑行为检查；
- 可重渲染 HTML 与样式 token CSS variables；
- SSIM、MSE 与颜色距离视觉评测；
- 防泄漏 manifest 强制训练/测试；
- 近重复截图检测；
- 单模态、无结构和无对齐消融配置；
- gold fine-tuning 配置；
- bootstrap CI、配对置换检验与 Holm 校正；
- 真实样本端到端 smoke test。
- 500 页 Pilot 选择、渲染、弱标注和质量报告；
- 10 页去重、分层、双人盲标工作台与标注包（未完成第二位真人标签）；
- 双人标注一致性阻断门的代码与测试；
- 截图与 PageGraph 驱动的本地人工标注工作台；
- 草稿保存、提交校验和双标注者进度隔离；
- 可审计且禁止覆盖已标注结果的 Pilot 样本替换工具；
- element/group 混合直接子实体和后代原子覆盖自动归一化。
- 10 页单名人工标注者底稿与独立 AI 代理候选；
- 人工/AI 可视化差异复核工具、来源选择与审核锁定；
- 10 页单人 AI 辅助定稿的 Human Gold 导出与可复现汇总；
- 60 页 DesignIntent Reference v1 固定清单与同 split 质量替换；
- 10 页人工复核 Gold 与 50 页 AI 多视角 Silver 的来源分层；
- 保守/扩展 PageGraph 结构候选、全分辨率截图语义审阅和显式视觉修正；
- AI Silver 每页最多 12 个的自动紧凑样式 Token；Human Gold Token 标为未采集；
- 60 页只读参考标注工作台，以及 AI Silver / Human Gold 直观来源标识；
- 参考文件、结构候选、截图、PageGraph 和视觉审阅的 SHA-256 审计链；
- Reference v1 来源感知 Dataset、完整 `split x tier` 审计和 test IR 访问边界；
- Human/AI 任务可用性掩码、页级来源权重与无权重 AI Silver checkpoint 选择；
- `intent-evaluation/v2` 分层报告、失败分母、Human 聚合抑制和 AI Silver bootstrap；
- `intent-comparison/v2` cohort 防缩减、配对置换、效应量与 Holm 校正；
- 正式 test 固定路径排他解封，且 checkpoint/model 与 manifest 语义预检早于解封。

## 尚未完成

- 第二位独立真人设计师标注（当前 AI 代理轨道不能替代双人信度）；
- 原生 Figma 插件/API 导出；
- 外部图片资源本地缓存；
- 无约束求解消融；
- 正式弱监督 checkpoint、Reference 微调与主结果生成；
- 三随机种子批量实验；
- 系统性文献综述；
- 主实验完成后的论文章节初稿。

## 已验证

```text
pytest: 195 passed（2026-09-01 全量回归）
真实 Playwright 渲染: 1/1 成功
旧数据迁移: 1/1 成功
真实样本闭环: 170 nodes → 39 elements + 31 groups
manifest 驱动模型训练 smoke: 1 epoch 完成
模型评测 smoke: 1/1 IR 通过约束校验
真实样本 HTML 重渲染: SSIM 0.9358, MSE 0.00665
Pilot 渲染: 492/500 成功
Pilot 弱标签与校验: 492/492 成功
Pilot 有效独立页面估算: 410
标注工作台 HTTP 提交闭环: 1/1 成功（临时标注包）
层级分组规则回归: TABLE → TABLE_ROW → element 通过
现有人工标注 group 覆盖迁移: 4 files，迁移后 0 invalid
人工主标注: 10/10 完成
AI 代理独立候选: 10/10 完成（不计作第二位真人）
单名人工标注者提交审核声明并定稿: 10/10 完成（7/10 说明仅为简短摘要）
最终 gold_intent.json: 10/10 导出，0 invalid
DesignIntent Reference v1: 60/60 complete，10 Human Gold + 50 AI Silver
AI Silver: 1573 elements + 447 groups + 245 tokens，0 invalid
视觉结构候选: conservative 35 / expanded 15
视觉审阅复现边界: 冻结审阅 JSON 后的 artifact replay；不支持端到端重生成视觉判断
Reference train/validation 实际加载: 30/10，tier 为 7+23 / 1+9
Reference 完整审计计数: train 7+23 / validation 1+9 / test 2+18
Reference 1 epoch mock 训练与 v2 合成评测: 通过
正式模型评测: 未运行，固定 test_unseal.json 不存在
访问审计: 回归测试曾误构造 model-ready test Dataset；无推理/指标，已移除并加门禁
```

Smoke 指标不代表模型效果，只证明数据、训练、求解与评测链路连通。
