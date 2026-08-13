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
- 10 页去重、分层、盲双标注包；
- 双人标注一致性阻断门；
- 截图与 PageGraph 驱动的本地人工标注工作台；
- 草稿保存、提交校验和双标注者进度隔离；
- 可审计且禁止覆盖已标注结果的 Pilot 样本替换工具；
- element/group 混合直接子实体和后代原子覆盖自动归一化。
- 10 页真实设计师主标注与独立 AI 代理候选；
- 人工/AI 可视化差异仲裁、逐项来源选择与审核锁定；
- 10 页 AI 辅助、真实人员审核的最终金标准导出与可复现汇总。

## 尚未完成

- 第二位独立真人设计师标注（当前 AI 代理轨道不能替代双人信度）；
- 原生 Figma 插件/API 导出；
- 外部图片资源本地缓存；
- 无约束求解消融；
- 三随机种子批量实验；
- 系统性文献综述；
- 主实验完成后的论文章节初稿。

## 已验证

```text
pytest: 33 passed
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
真实人员逐页仲裁: 10/10 完成
最终 gold_intent.json: 10/10 导出，0 invalid
```

Smoke 指标不代表模型效果，只证明数据、训练、求解与评测链路连通。
