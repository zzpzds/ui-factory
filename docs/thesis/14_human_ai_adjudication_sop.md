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

脚本只处理人工和 AI 均为 `complete` 的样本。当前会生成 7 对材料，并把
`1094`、`1429`、`1474` 标为 `incomplete_human`。设计师 A 完成剩余样本后，
再次运行同一命令即可；已有记录不会被覆盖，新增 3 个样本会被补齐。

若原始人工或 AI 文件在生成后发生变化，脚本会因 SHA-256 不一致而中止。此时
不要删除旧记录直接重跑，应先确认变更原因并保留变更审计记录。

## 4. 单页审核顺序

以 `0001` 为例，同时打开网页截图、
`adjudication/0001.json` 和 `gold_drafts/0001.json`，按以下顺序处理：

1. 查看 `differences.elements.only_human` 与 `only_ai`，判断是否为源节点选择、
   合并/拆分或漏标；在 gold draft 中保留正确方案或重新标注。
2. 查看 `matched_but_changed`，重点核对 `type_disagrees` 与低 `bbox_iou` 项；
   名称差异不是自动错误，类型和可编辑边界才是主要判断依据。
3. 查看 `differences.groups`，判断每个语义组是否具有独立选择、移动或复用意义；
   组可以只含子组，但必须至少有一个直接设计子实体。
4. 查看 `differences.tree`，确认每个元素/组应挂在哪个父组。表格优先检查
   `TABLE -> TABLE_ROW -> elements`，列表优先检查 `LIST -> LIST_ITEM`。
5. 查看 `differences.layouts`，结合截图核对方向、gap、padding、对齐和 resize。
6. 查看 `differences.tokens`，只把“应当联动修改”的成员放入同一 Token；数值
   相同但语义无关的样式不应强行合并。
7. 修改 gold draft 后，使用标注平台或校验代码确认 schema、bbox、tree、layout
   和 Token 全部合法。

`metrics` 只帮助定位分歧密集区域，不能代替逐项视觉判断，也不能作为自动采用
人工或 AI 版本的规则。

## 5. 写入审核凭证

完成一页后，修改 `adjudication/<id>.json`：

```json
{
  "status": "reviewed",
  "review": {
    "reviewed_by": "真实审核人标识",
    "reviewed_at": "2026-08-03T15:00:00+08:00",
    "notes": "说明主要分歧、采用依据和重新标注内容"
  }
}
```

同时修改 `gold_drafts/<id>.json` 的 `provenance`：

```json
{
  "status": "complete",
  "adjudication_status": "reviewed",
  "gold_finalized": false,
  "reviewed_by": "与仲裁记录相同的审核人标识",
  "reviewed_at": "与仲裁记录相同的时间",
  "review_notes": "非空审核说明"
}
```

不要把 `reviewed_by` 填为 `annotator_ai`。审核动作必须由真实人员完成。

## 6. 导出最终金标准

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

## 7. 论文披露

方法章节应明确说明：AI 代理在盲态下独立生成候选标注，用于发现分歧；最终标签
由真实人员结合截图、HTML/PageGraph 和差异报告逐项复核。人机指标属于跨来源
描述性比较，不是 inter-annotator reliability。条件允许时，应另请外部人员抽查
至少 2 页，并报告抽查范围和发现。
