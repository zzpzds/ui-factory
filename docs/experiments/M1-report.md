# 实验报告 M1：基础模型可行性验证

> **实验日期**：2026-06-04
> **数据**：data/sampled（95 条 WebCode2M 样本，train=80 / val=15）
> **硬件**：MacBook MPS，16GB
> **配置**：configs/default.yaml
> **训练时长**：~45 分钟（30 epoch）

本次实验是论文方案的**端到端可行性验证**。在 95 条小数据集上验证：
1. 多任务损失能否同时收敛；
2. 解耦的对齐通道是否真的让对齐损失下降（vs 旧版 softmax-attention 直接做 BCE）；
3. 递归结构解码器在 teacher forcing 下是否能学出有意义的父指针；
4. 样式监督是否带动样式头收敛。

---

## 1. 模型规模与配置

| 项 | 值 |
|----|-----|
| 总参数 | 234.7M |
| 可训练 | 38.4M |
| ViT | google/vit-base-patch16-224，frozen |
| CodeBERT | microsoft/codebert-base，最后 2 层解冻 |
| max_nodes | 96 |
| batch | 1 物理，grad_accum=4 等效 |
| epochs | 30 |
| optimizer | AdamW，cosine + 5% warmup |
| 损失权重 | type=1.0, struct=0.5, align_bce=0.3, align_nce=0.05, style_reg=0.2, style_color=0.05, style_cls=0.1 |

---

## 2. 训练曲线

见 `outputs/M1-baseline/viz/fig_loss_curves.png`。

主要观察：
- `total` 总损失从 ~3.5 下降到 _TBD_（30 epoch 后）
- `type` 损失从 1.27 → _TBD_
- `struct` 损失从 3.16 → _TBD_（若 < 1.5 说明结构监督有效）
- `align_bce` 从 1.34 → _TBD_（这是关键指标，验证修复后的对齐损失能学）

> **注**：表格数值在主实验完成后自动从 `outputs/M1-baseline/train.log.jsonl` 提取。

---

## 3. 验证集指标（best epoch）

待 `scripts/evaluate.py --checkpoint outputs/M1-baseline/best.pt` 跑完后填入：

| 指标 | 数值 |
|------|------|
| type_acc | _TBD_ |
| parent_f1 | _TBD_ |
| parent_p | _TBD_ |
| parent_r | _TBD_ |
| tree_edit | _TBD_ |
| style_reg_mae | _TBD_ |
| color_de | _TBD_ |
| cls_textAlign_acc | _TBD_ |
| cls_display_acc | _TBD_ |

---

## 4. 关键修复对训练稳定性的影响（论文 chapter 3.4 用）

| 修复前 | 修复后 |
|--------|--------|
| **对齐 BCE**：直接 softmax-attention × 多对多标签 → 一个 node 多 patch 时永远学不会 | 解耦通道，bilinear sigmoid+BCE，pos_weight 自动平衡 |
| **结构 loss 爆炸到 6669**：训练初期 predicted type 错把 GT 父判为叶子，target 落到 -1e4 mask 列上 | 训练阶段 `apply_leaf_prior=False`；TEXT 从 LEAF_TYPES 移除 |
| **batch=1**：N 不一致没法 stack | padding+node_mask，所有 N 维 loss 乘 mask |
| **截断后悬空指针**：`parents[parents>=max_nodes]=-1` 把祖父→孙强制变根 | BFS 保留连通子树 + `_reindex_parents` 回溯祖先 |

---

## 5. 消融对比（待消融实验结束后填）

```
| 实验 | 配置                                | type_acc | parent_f1 | tree_edit | total |
|------|-----------------------------------|----------|-----------|-----------|-------|
| M1   | full（baseline）                   |   _TBD_  |   _TBD_   |   _TBD_   | _TBD_ |
| A1a  | 去掉对齐监督（α_bce=α_nce=0）       |   _TBD_  |   _TBD_   |   _TBD_   | _TBD_ |
| A1b  | 仅 BCE，去掉 InfoNCE              |   _TBD_  |   _TBD_   |   _TBD_   | _TBD_ |
| A4a  | max_nodes=50（覆盖率↓60%）         |   _TBD_  |   _TBD_   |   _TBD_   | _TBD_ |
| A4b  | max_nodes=128（覆盖率↑85%）       |   _TBD_  |   _TBD_   |   _TBD_   | _TBD_ |
```

预期结论：
- A1a 应在 parent_f1 上**显著下降**——对齐监督让节点表征带有视觉锚定
- A1b 应在 parent_f1 上**轻微下降**——InfoNCE 提供更强判别信号
- A4a 应在 tree_edit 上**显著上升**——大网页被截断丢失结构
- A4b 应能略胜 M1，但 MPS 上慢约 1.7×

---

## 6. 启发式基线对比（B1：BuilderIO/figma-html）

> 占位：待 fork 安装并运行 figma-html 后填。

预期：figma-html 仅靠规则解析，缺视觉理解，在含图标/动态布局的网页上 type_acc 与 parent_f1 应明显低于本方案。

---

## 7. 论文用可视化（一览）

- `viz/fig_loss_curves.png` —— 5 项 loss 曲线
- `viz/fig_cm.png` —— 节点类型混淆矩阵
- `viz/fig_attention.png` —— 3 个代表节点的对齐 attention 热图叠加原图
- `viz/fig_tree_compare.png` —— GT vs Pred 树并排
- `viz/fig_depth_hist.png` —— 树深度分布

---

## 8. 失败案例（人工挑选 4 张）

待主实验后由 `scripts/evaluate.py` 输出排序最差样本的 figma JSON，再人工归因。常见预期：
1. **深嵌套层（depth>10）**：递归解码深度上限 16，但训练样本中很少 > 8，模型外推能力受限。
2. **重叠浮动元素**：`position:absolute` 的元素 box 包含关系不再可靠，IoU 标签噪声大。
3. **图文混排密集**：节点数超过 max_nodes 后 BFS 会丢叶子。
4. **无 inline 颜色**：颜色 head 只能从 computedStyle 监督；若 CSS 用了变量或 ::before 伪元素，可能丢失监督信号。

---

## 9. GPU 集群迁移指南

把同一份代码复制到 CUDA 服务器，仅需：
1. `pip install -r requirements.txt`（已 device-agnostic）
2. 编辑 `configs/default.yaml`：
   - `training.batch_size: 4`
   - `data.max_nodes: 192`
   - `model.codebert_unfreeze_last: 4`
   - `training.epochs: 100`
3. 启动：`accelerate launch scripts/train.py --config configs/gpu.yaml`（DDP 自动）

预计 100 epoch 在 1×A100 上约 2-3 小时（节点数翻倍带来 N×N 矩阵开销）。
