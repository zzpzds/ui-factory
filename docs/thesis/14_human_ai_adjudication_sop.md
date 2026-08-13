# 人工-AI 差异仲裁与金标准导出 SOP

## 1. 适用范围

本流程用于当前无法取得第二位独立设计师时的辅助复核。设计师 A 是人工主标注，
`annotator_ai` 是在冻结前未查看人工标签的 AI 代理。该流程产生的是
**AI 辅助、人工审核的金标准**，不能写成“双人标注”“标注者间一致性”或把 AI
计入真人标注员人数。

## 2. 文件角色

| 路径 | 作用 | 是否可直接训练 |
| --- | --- | --- |
| `annotator_a/<id>.json` | 人工原始标注，保留原貌 | 否 |
| `annotator_ai/<id>.json` | 冻结的 AI 代理标注，保留原貌 | 否 |
| `adjudication/<id>.json` | 人机指标、逐项差异和审核记录 | 否 |
| `gold_drafts/<id>.json` | 以人工标注为底稿的待审核版本 | 否 |
| `data/processed/<id>/gold_intent.json` | 通过导出门槛的最终版本 | 是 |

## 3. 生成待审核材料

在项目根目录执行：

```bash
source .venv/bin/activate
python scripts/prepare_human_ai_adjudication.py
```

脚本只处理人工和 AI 均为 `complete` 的样本。当前 10 对材料已全部生成，
`adjudication/index.json` 中 `ready_pairs=10`、`complete=true`。重复运行同一命令
不会覆盖已有记录。

若原始人工或 AI 文件在生成后发生变化，脚本会因 SHA-256 不一致而中止。此时
不要删除旧记录直接重跑，应先确认变更原因并保留变更审计记录。

## 4. 启动仲裁平台

在项目根目录执行：

```bash
source .venv/bin/activate
python scripts/serve_intent_annotation.py \
  --package_dir data/annotations/intent_pilot_v1 \
  --port 8765
```

浏览器打开 `http://127.0.0.1:8765`。平台默认进入“差异仲裁”模式，顶部进度表示
已由真实人员审核的样本数；样本编号前出现勾选表示该页已审核并锁定。不要在仲裁
阶段切换到“人工标注”模式修改原始的 `annotator_a` 文件。

## 5. 单页审核顺序

1. 在左侧“差异”页签查看一致性摘要；这些指标只用于定位争议，不能作为自动采用
   人工或 AI 版本的规则。
2. 依次选择“原子元素、语义分组、父子层级、布局约束、样式 Token”筛选器。点击
   任意差异项后，平台会选中对应源节点并将画布滚动到相关区域。
3. 每条元素、分组、层级和布局差异下方都会显示金标准当前状态，并提供“采用人工”
   和“采用 AI”按钮。点击后只修改金标准草稿，不修改原始人工或 AI 文件；涉及
   删除实体时必须再次确认。若两边都不正确，继续使用节点、实体和 Token 编辑器
   手工调整，状态会显示为“当前：已调整”。
4. 画布默认以青色框显示 AI 参考。按需打开琥珀色“人工参考”，或关闭其中一层以
   避免框线重叠。实线框表示原子元素，虚线框表示语义分组。
5. 在左侧“节点”页签和“实体”页签修改当前金标准草稿。草稿初始来自人工标注；
   只有经视觉与语义判断确认后，才采用 AI 方案或重新标注。原始人工与 AI 文件
   始终保持只读。
6. 元素差异重点检查合并/拆分、漏标、类型和 bbox；名称差异本身不代表错误。
7. 分组与层级重点检查独立选择、移动或复用意义。分组可以只包含子分组，也可以
   只有一个直接设计子实体，但最终必须覆盖至少一个后代原子元素。表格优先检查
   `TABLE -> TABLE_ROW -> elements`，列表优先检查 `LIST -> LIST_ITEM`。
8. 布局差异结合截图核对方向、gap、padding、对齐和 resize；Token 只合并应当
   联动修改的成员，不要仅因样式数值相同而强行合并。
9. 中途点击“保存仲裁草稿”。此操作只更新 `gold_drafts/<id>.json`，不会把样本
   标成已审核，也不会生成训练用金标准。

## 6. 完成人工审核

确认右侧校验错误为 0 后，在“人工审核”区域填写：

- `审核人标识`：真实审核人的姓名或稳定编号；不能填写 `annotator_ai`。
- `审核说明`：说明主要分歧、最终采用依据以及重新标注的内容，不能留空。

点击“完成人工审核”。服务端会重新校验 IR 结构、输入哈希、bbox、tree、layout
与 Token，并自动写入审核时间和 provenance。校验失败时，页面仍保持待审核，右侧
会列出原因；通过后该样本锁定，不能继续修改。

审核凭证由平台同时写入 `adjudication/<id>.json` 和 `gold_drafts/<id>.json`，
不要再手工编辑这两处状态字段。审核动作必须由真实人员完成。

## 7. 导出最终金标准

先按单样本导出，确认无误后再批量执行：

```bash
python scripts/finalize_human_ai_adjudication.py --sample_id 0001
python scripts/finalize_human_ai_adjudication.py
```

导出脚本会同时检查：

- 仲裁记录与草稿均已人工审核；
- 审核人、时间和说明完整且相互一致；
- 人工与 AI 原始输入哈希未变化；
- IR 结构、bbox、分组、布局和 Token 通过提交校验；
- 目标 `gold_intent.json` 不存在冲突版本。

通过后写入 `data/processed/<id>/gold_intent.json`，其来源固定为
`human_ai_adjudicated_gold`，并保留 `ai_assistance_disclosed=true`。任何门槛失败
都会拒绝导出，不会用待审核草稿覆盖训练金标准。

## 8. 论文披露

方法章节应明确说明：AI 代理在盲态下独立生成候选标注，用于发现分歧；最终标签
由真实人员结合截图、HTML/PageGraph 和差异报告逐项复核。人机指标属于跨来源
描述性比较，不是 inter-annotator reliability。条件允许时，应另请外部人员抽查
至少 2 页，并报告抽查范围和发现。
