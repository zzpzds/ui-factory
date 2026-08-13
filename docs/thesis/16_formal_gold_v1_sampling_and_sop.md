# Formal Gold v1 扩展设计与标注 SOP

## Material Passport

- `artifact_type`: human-study sampling and execution record
- `artifact_id`: `formal-gold-v1-20260813`
- `verification_status`: `VERIFIED`
- `source_pool`: 492 个成功渲染的 Intent Pilot 页面
- `output`: `data/annotations/intent_gold_v1`
- `human_labels`: 10 个已冻结 Gold + 50 个待人工标注样本
- `weak_labels_used_as_truth`: false

## 目标

把 10 页仲裁 Gold 扩展为 60 页正式 Gold，同时补足页面类型、页面规模和潜在
样式 Token 的覆盖。该数据集用于监督微调、验证和最终测试，不替代 492 页弱标
签池。

## 固定切分

正式 Gold 沿用 `data/intent_pilot_split.json` 的既有归属，不重新随机切分：

| split | 数量 | 用途 |
| --- | ---: | --- |
| train | 30 | Gold 监督微调和误差分析 |
| validation | 10 | 阈值、超参数和停止条件选择 |
| test | 20 | 最终一次主结果评估 |

现有 10 页 Gold 保持原 split 并设为只读。新增 50 页中，train、validation、
test 分别为 23、9、18 页。

## 筛选方法

运行：

```bash
source .venv/bin/activate
python scripts/create_formal_gold_package.py
```

筛选器执行以下约束：

1. 仅使用成功渲染且具备 `screenshot.png` 与 `page_graph.json` 的页面；
2. 页面节点数限定为 8 至 180，截图亮度标准差不低于 10；
3. 排除带有明确成人或赌博导流文本的页面，使样本边界保持为通用网页界面；
4. 当页面至少有 4 个图片节点且不少于 60% 退化为约 16x16 时，标记为疑似
   大规模破图并排除；
5. 使用 16x16 dHash、Hamming 距离不大于 12 构造近重复簇，每簇只取一页；
6. 正式清单中的结构指纹必须唯一；
7. 同时覆盖表格、表单、电商、列表、导航、内容和通用页面；
8. 依据 PageGraph 中重复 computed style 选择 Token 富集候选，不读取
   `weak_intent.json`；
9. 人工检查候选联系表与可疑原图，将资源失败、大片无效空白和明显布局失序
   写入 `visual_exclusions.json` 后重新筛选。
10. Pilot 阶段已被真实设计师判定为低质量并替换的页面自动进入永久禁选集合，
   不得因自动评分较高而重新进入正式 Gold。

后加入的内容与破图排除规则只约束新增 50 页，不反向删除已经由真人完成并冻结
的 10 页 Pilot Gold；既有页的规则触发情况仍写入清单，作为样本边界披露。

清单同时保存每张截图和 PageGraph 的 SHA-256，用于 Git 跨设备同步后的完整性
核验。

另保留 20 个不进入训练或评测的独立备用页：train 10、validation 5、test 5，
仅在待标样本经人工判定不合格时按原 split 替换。

当前 PageGraph 没有保存图片元素的 `complete` 与 `naturalWidth`，因此破损图片
无法只靠结构数据稳定识别，仍需以候选联系表和原始截图完成人工视觉 QA。这是
本轮采集器的已知限制，后续重新渲染数据时应补充该字段。

最终 60 页包含 small 20、medium 31、large 9。样式富集候选总数及分 split
数量以 `selection_manifest.json` 为准，并且新增页按 train/validation/test
分别不少于 12/4/8 页。

## 解释边界

样式富集的目的是让标注者有机会识别真实的设计 Token，不表示这些页面一定
存在语义 Token，也不能把 PageGraph 重复样式直接写成 Gold。由于样本经过质量
筛选和 Token 富集，最终指标表示“研究定义的高质量网页任务分布”表现，不能无
权重外推到全部网页。

## 标注启动

```bash
source .venv/bin/activate
python scripts/serve_intent_annotation.py \
  --package_dir data/annotations/intent_gold_v1 \
  --port 8766
```

访问 `http://127.0.0.1:8766`。工作台会：

- 自动进入“人工标注”，不显示尚不存在的人机仲裁模式；
- 把 50 个待标页面排在前面，把 10 个已冻结 Gold 放在末尾；
- 显示初始进度 `10 / 60`；
- 禁止在前端或 API 修改已冻结 Gold。

## 每页操作顺序

1. 先检查截图是否仍有未发现的破图、主体缺失或严重布局错乱；发现后停止标注，
   记录 sample id，不自行更改 assignment；
2. 在“节点”中创建原子元素，优先保证视觉对象完整，不复制 DOM wrapper；
3. 在“实体”中建立语义分组和设计树；父组允许只直接包含子分组；
4. 为每个 group 设置布局模式、gap、padding、对齐和 resize；
5. 在“Token”中只标注设计上应同步修改且至少有两个成员的共享样式；
6. 保存草稿后检查右侧校验；全部通过再点击“提交完成”；
7. 提交后确认样本前出现勾选，进度增加 1。

## 阶段门槛

1. 完成前 10 个新增样本后暂停，检查 Token 是否仍为 0；若为 0，先复核协议理解，
   不直接完成剩余 40 页；
2. 50 页全部完成后冻结 `annotator_a` 原始结果并生成独立 AI 代理轨道；
3. AI 只能用于差异定位，不能被报告为第二位真人；
4. 每页经真人差异复核后才能生成 `gold_intent.json`；
5. test 的 20 页在模型、阈值和排除规则冻结前不得用于调参。

## 可复现性检查

```bash
python -m pytest tests/test_formal_gold_package.py -q
```

测试固定检查样本数、split、重复簇、结构指纹、冻结状态、空白草稿、资源路径和
Token 富集最低配额。
