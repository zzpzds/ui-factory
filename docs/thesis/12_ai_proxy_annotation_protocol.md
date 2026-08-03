# AI 代理标注协议 v1

## 定位与边界

AI 代理标注用于在无法招募第二位设计专家时，提供独立于人工标注和规则弱标签
的第二套结构判断。它服务于差异发现、人机跨来源一致性分析和人工复核，不得
冒充第二位真人设计师，也不得用于宣称人类标注者间信度。

论文中的规范名称为“AI 代理标注”或“人机跨来源比较”，禁止使用“双人标注”
“第二位人工标注员”或“人工 Cohen's kappa”等表述描述该轨道。

## 盲态要求

AI 代理在完成全部样本前只能访问：

- 原始首屏截图；
- 对应 PageGraph；
- `docs/thesis/05_annotation_protocol.md` 中的对象、分组和布局定义；
- Design Intent IR schema 与确定性校验规则。

AI 代理不得访问：

- `annotator_a/*.json`；
- `annotator_b/*.json` 中可能出现的后续真人结果；
- `weak_intent.json` 或弱监督生成器输出；
- 人工与弱标签的指标或分歧报告。

代码生成阶段只允许读取 assignment、截图和 PageGraph。每份输出记录截图与
PageGraph 的 SHA-256，并显式写入：

```json
{
  "label_source": "ai_proxy_annotation",
  "annotator": "annotator_ai",
  "annotator_kind": "ai_proxy",
  "weak_labels_viewed": false,
  "human_labels_viewed": false
}
```

## 判断流程

1. 先按原始分辨率查看完整截图，识别视觉区域、阅读顺序和重复模式。
2. 再查看 PageGraph 的文本、标签、边界框和 computed style，定位截图对象的
   稳定源节点。
3. 只保留具有独立选择或编辑价值的对象；不可见 wrapper、破损资源和纯实现
   节点不标记。
4. 按设计语义建立组件，不复制 DOM 层级。表格使用
   `TABLE → TABLE_ROW → elements`，列表优先显式建立重复 `LIST_ITEM`。
5. 根据截图中的实际排列标记布局；无法由稳定规则解释的重叠区域使用 `FREE`。
6. 只为明确应联动修改的重复样式建立 Token，不把所有数值相同的样式机械合并。
7. 每页输出后运行 schema、来源节点唯一性、树、group、layout 和 Token 校验。
8. 全部 AI 文件冻结后才允许读取人工标签并计算人机跨来源指标。

## 数据目录

```text
data/annotations/intent_pilot_v1/
├── annotator_a/       # 真人设计师 A
├── annotator_b/       # 预留真人设计师 B，不写入 AI 数据
└── annotator_ai/      # 独立 AI 代理标注
```

可复现生成命令：

```bash
python scripts/generate_ai_proxy_annotations.py \
  --package_dir data/annotations/intent_pilot_v1
```

## 结果解释

- AI 与人工一致只能说明两种来源对设计结构的判断接近，不能证明人工标注信度。
- AI 与人工都可能受到 PageGraph 缺失、截图裁剪和设计语义歧义影响。
- AI 输出不能直接成为共识金标准；最终 `gold_intent.json` 必须经过人工逐项复核。
- 同一位人工标注者复核自身与 AI 的分歧不等于独立第三方裁决。
- 条件允许时，应由导师或具备 UI 经验的外部人员抽查至少 20% 样本，并在论文
  中报告抽查范围。

## 论文披露模板

> 由于第二位设计专家招募受限，本研究由一名设计师独立完成人工标注，并使用
> 多模态大模型在不可见人工标签和规则弱标签的条件下完成代理标注。AI 结果仅
> 用于差异检测和人机跨来源一致性分析，最终金标准由人工复核确定。因此，本文
> 不将相关一致性指标解释为人类标注者间信度。
