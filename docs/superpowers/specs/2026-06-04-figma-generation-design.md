# 基于网页图像与源码的 Figma 设计稿生成 — 模型设计文档

**作者**：赵志鹏（学号 ZF2421123）
**导师**：胡峻林副教授（学院）/ 姜伟善高工（字节跳动）
**日期**：2026-06-04
**对应论文章节**：第 3 章 模型方案；第 4 章 实验设计

---

## 1. 研究问题与目标

**输入**：网页截图 $I \in \mathbb{R}^{H \times W \times 3}$ + 网页源码 HTML/CSS。
**输出**：Figma JSON 格式的结构化设计稿，包含节点类型、嵌套结构、几何与样式。

**核心挑战**：
1. **异构模态对齐**：视觉像素 vs 离散 DOM 节点，特征空间不同；
2. **多对多对应**：一个 DOM 节点对应多个视觉 patch；一个视觉区域可被多个嵌套节点共享；
3. **结构化输出**：Figma JSON 是树（容器/叶节点二元 + 父子约束 + 样式属性），不是平铺序列。

**论文创新点**（第 3.5 节）：
- (i) **以渲染区域为对齐监督**——比依赖隐式学习的方法更具可解释性；
- (ii) **三层解码架构**（节点类型 → 内部结构 → 样式属性）——模仿真实设计树的递归构建过程；
- (iii) **解耦的对齐通道**（融合 vs 监督）——数学上正确建模一对多对齐关系。

---

## 2. 整体架构

```
HTML  ─► parse_html_nodes ──┐
                            ├─► CodeEncoder (CodeBERT，最后2层解冻) ─► [B,N,768]
PNG   ─► PIL Image  ────────┼─► VisualEncoder (ViT-B/16，frozen)   ─► [B,196,768]
                            │
                       ┌────┴──────────────────────────────────────┐
                       │ CrossModalAlignment（解耦）                │
                       │  • 融合通道：MultiheadAttention(softmax)  │
                       │  • 监督通道：Bilinear sim → sigmoid + BCE │
                       │  • 辅助：InfoNCE（系数 0.05）             │
                       │  - fused [B,N,768]                        │
                       │  - sim_logits [B,N,196]                   │
                       └────┬──────────────────────────────────────┘
                            │
                       NodeContextEncoder
                       （2层self-attn + depth/sibling emb）─► ctx [B,N,768]
                            │
                ┌───────────┼───────────────────────────────┐
                ▼           ▼                               ▼
        TypeClassifier  RecursiveStructDecoder       StylePredictor
            (CE)        （指针网络 + 深度条件）        (回归+分类+颜色)
```

模型总参数量 234.7M，可训练 38.4M（ViT 全冻、CodeBERT 仅最后 2 层解冻）。

---

## 3. 模块详细设计

### 3.1 双模态编码器

**视觉编码器**：`google/vit-base-patch16-224`，输出 `[B, 196, 768]`（去 CLS 的 14×14 patch token）。
**代码编码器**：`microsoft/codebert-base`，每个 DOM 节点取 CLS token。
- 节点序列化：`<tag attr1="v1">inner_text</tag>`（不递归后代，inner 截 100 字符）
- 与 Playwright 渲染时的 JS 序列化协议保持一致，避免训/测分布漂移

### 3.2 跨模态对齐（关键模块，对应论文 3.2.2）

**关键修复**：原始实现用 softmax-attention（行和=1）和多对多 0/1 标签做 BCE，**数学错误**——
softmax 强制权重和为 1，模型永远学不到同时给多个 patch 高分。

**新设计**：融合通道 vs 监督通道完全解耦。
- 融合通道：`nn.MultiheadAttention` (softmax-normalized) → `fused [B, N, 768]`
- 监督通道：独立 `BilinearSimilarity`（投影到 256 维后 dot-product / √D）→ `sim_logits [B, N, 196]`
- 损失：
  - $L_{\text{align\_bce}} = \text{BCE}(\sigma(s_{ij}), y_{ij})$，pos_weight 自动平衡
  - $L_{\text{align\_nce}}$：InfoNCE，每节点取 IoU 最大 patch 为正样本

**对齐标签构造**：用 **Overlap Coefficient** 而不是 IoU
$\text{OC}(b_i^{\text{node}}, b_j^{\text{patch}}) = \frac{|b_i \cap b_j|}{\min(|b_i|, |b_j|)}$
节点框通常远大于 16×16 patch，IoU 会系统性低估。阈值 0.3。

### 3.3 NodeContextEncoder（结构感知）

**动机**：CodeBERT 单独编码每节点的 CLS，节点之间没有交互；交叉注意力虽然让节点看到了视觉，但兄弟/层级关系仍缺失。

**实现**：
- 加 `depth_emb`（BFS 距离根的层数，max=32）
- 加 `sibling_emb`（在父的 children 列表里的位置，max=32）
- 2 层 `nn.TransformerEncoderLayer(d=768, h=4, ff=1536, norm_first=True)`
- 使用 `src_key_padding_mask` 屏蔽 padding

输出 `ctx [B, N, 768]`，喂给三个解码器头。

### 3.4 三层解码器（对应论文 3.2.3）

#### 3.4.1 L1 节点类型分类
简单 MLP head，9 类（FRAME/GROUP/TEXT/RECTANGLE/...）。CE 损失，节点 mask 加权。

#### 3.4.2 L2 RecursiveStructDecoder（关键模块）

**输入**：`ctx [B, N, D]` + `target_types [B, N]` + `parents (训练时) [B, N]` + `depth (训练 teacher forcing) [B, N]`

**架构**：
```
h = ctx + depth_emb(d) + type_emb(t)
h = TransformerEncoder(h, padding_mask)            # 2 层
parent_logits[b, i, j] = c_proj(h_i)·p_proj(h_j) / √d_proj   # [B, N, N]
root_logits[b, i]      = c_proj(h_i)·root_query / √d_proj    # [B, N]
logits[b, i, :] = concat(parent_logits[b,i,:], root_logits[b,i])  # [B, N, N+1]
```

最后一列 N 作为 root 汇聚位（每个根节点在此列得分最高时归根）。

**candidate_mask** [B, N, N+1]：
- 列 j ∈ [0, N): 必须 valid + j != i
- 列 N: 永远 True
- 推理阶段额外屏蔽叶子类型作为父；**训练阶段不屏蔽 leaf**——HTML 中部分被映射为 RECTANGLE 的元素（如 `<input>`）仍可能含子，避免 GT 父被屏蔽导致 CE = ∞ 爆炸

**训练损失**：
$L_{\text{struct}} = \frac{1}{|\text{valid}|} \sum_i \text{CE}(\text{logits}_i, \text{parent}_i)$
其中 parent_i 落到 [0, N-1] 是常规父，落到 N 是 root。

**推理（递归展开，论文核心叙述）**：
- 第 0 步：把首选为 root 列的节点定为 depth=0
- 第 d 步：对每个未定父节点，在已定深度 d-1 的非叶节点里 argmax；落不下来回退为 root
- 最大深度 16

#### 3.4.3 L3 StylePredictor（多任务）

3 类输出头：
- **回归头**：5 个连续属性（borderRadius, borderWidth, opacity, fontSize, fontWeight），sigmoid 归一化到 [0,1]，训练时除以 STYLE_REG_NORMS
- **颜色头**：3 个 RGBA × 4 通道（backgroundColor, color, borderColor），sigmoid → [0,1]
- **分类头**：textAlign (6 类) + display (8 类)，CE

样式监督来自 Playwright `getComputedStyle`（每节点 9 字段，存 `styles.json`）。前 2 epoch 关闭样式损失，让对齐和类型先稳。

### 3.5 build_figma_json（推理时把 logits 转 Figma JSON）

**关键修复**：旧实现完全用边界框包含关系建树，**绕过模型预测的 parent_logits**——使消融实验失效。
新实现真正调用 `decode_parents_recursive` 用模型预测的指针建树，包含关系仅作 fallback。

---

## 4. 数据预处理

### 4.1 数据来源
- **WebCode2M**（百万级真实网页 HTML/截图）：本机采样 100 条做可行性验证
- **Material Design 3**（Figma 社区）：第二阶段微调用，本次提交不依赖

### 4.2 渲染管线
`scripts/render_pages.py`：Playwright 加载 HTML（保留 CSS/图像/字体）→ 截图 1280×800 → 通过 `document.querySelectorAll('*')` + `getBoundingClientRect()` + `getComputedStyle()` 取每节点：
- 序列化文本（与 `parse_html_nodes` 一致）
- 边界框 → 缩放到 224×224 坐标系
- 父索引（最近保留祖先）
- 9 个 computed style 字段

### 4.3 数据集对象
- `max_nodes=96`：覆盖 ~70% 数据；BFS 截断保留连通子树
- padding + node_mask：所有 N 维 tensor 已经填到固定形状，支持 batch>1
- `webpage_collate_fn`：真正支持多样本 batch（旧版 assert batch=1）
- `_encode_styles`：把 list[styles_dict] 转 `reg/reg_valid/color/color_valid/cls_textAlign/cls_display`

### 4.4 数据划分
本次 95 条有效样本 → train=80 / val=15。

---

## 5. 训练目标与超参

### 5.1 多任务总损失

$$L = w_t L_{\text{type}} + w_s L_{\text{struct}} + w_a L_{\text{align\_bce}} + w_n L_{\text{align\_nce}} + w_r L_{\text{style\_reg}} + w_c L_{\text{style\_color}} + w_l L_{\text{style\_cls}}$$

| 项 | 权重 | 说明 |
|----|------|------|
| $w_t$ (type) | 1.0 | CE，mask 加权 |
| $w_s$ (struct) | 0.5 | 父指针 CE |
| $w_a$ (align_bce) | 0.3 | 主对齐信号 |
| $w_n$ (align_nce) | 0.05 | 辅助判别信号 |
| $w_r$ (style_reg) | 0.2 | 5 项 MSE |
| $w_c$ (style_color) | 0.05 | RGB MSE |
| $w_l$ (style_cls) | 0.1 | 2 个枚举头 |

### 5.2 本机训练配置

| 项 | 值 |
|----|-----|
| device | MPS（Apple Silicon） |
| max_nodes | 96 |
| ViT | frozen |
| CodeBERT | 最后 2 层 + pooler 解冻 |
| batch_size | 1 物理，grad_accum=4 等效 |
| precision | fp32 |
| epochs | 30（小数据集冒烟版本） |
| lr | heads 2e-4 / codebert 2e-5（cosine + 5% warmup） |
| weight_decay | 0.01 |
| grad_clip | 1.0 |
| 内存监控 | psutil > 85% 时自动 break + checkpoint |
| 散热 | 每 50 step sleep 0.5s |

可移植性：device 通过 `get_device()` / `safe_empty_cache()` / `autocast_ctx()` 抽象，迁 GPU 集群只需把 yaml 里 `precision: bf16` 打开。

---

## 6. 实验设计

### 6.1 主实验 M1
M1：完整模型在 95 条样本上训 30 epoch（train=80, val=15）。
报告指标：
- 节点级：type accuracy / bbox IoU
- 结构级：parent F1 / tree edit distance
- 样式级：reg MAE / color ΔE / textAlign acc / display acc

### 6.2 消融 A1–A4
- **A1 对齐监督**：BCE only / BCE+InfoNCE / 完全去掉
- **A2 结构解码**：递归 Transformer（本方案）vs 平铺 N×N 矩阵（旧方案）
- **A3 NodeContextEncoder**：开/关
- **A4 max_nodes**：50 / 96 / 128

### 6.3 基线 B1
fork `BuilderIO/figma-html` 启发式工具，对同样 15 条 val 出 Figma JSON，对比 type acc / parent F1。

---

## 7. 已识别并修复的核心问题

| 编号 | 问题 | 修复位置 |
|------|------|----------|
| P1 | 跨模态对齐损失数学错误（softmax × 多对多标签） | `cross_modal_alignment.py` 解耦双通道 |
| P2 | batch=1 限制 | `dataset.py` padding+mask、`webpage_collate_fn` 真支持 batch |
| P3 | 样式监督完全缺失 | `render_pages.py` 收集 computedStyle、`dataset._encode_styles` |
| P4 | 解码器没按论文递归 | `RecursiveStructDecoder` 指针网络+深度迭代展开 |
| P5 | CodeBERT 节点间无交互 | `NodeContextEncoder`（depth+sibling emb + self-attn） |
| P6 | max_nodes=50 截断过严 | 提到 96，BFS 保连通子树 |
| P7 | 截断后父子链断裂污染监督 | `_bfs_truncate` + `_reindex_parents` |
| P8 | build_figma_json 绕过预测 | 真用 `decode_parents_recursive` |
| P9 | 训练入口分裂 | 删除 `trainer.train()`，单一入口 `scripts/train.py` |
| P10 | TEXT 在 LEAF_TYPES 导致 GT 父被屏蔽 → CE 爆炸 | LEAF_TYPES 移除 TEXT；训练阶段 `apply_leaf_prior=False` |

---

## 8. 关键文件指针

```
configs/default.yaml                                  # 全部超参
scripts/train.py                                      # 单一训练入口
scripts/evaluate.py                                   # 评测+可视化
scripts/render_pages.py                               # Playwright 渲染（含 styles）
src/data/dataset.py                                   # WebpageDataset + collate
src/data/preprocessing.py                             # parse_html_nodes / get_patch_boxes
src/models/encoders/visual_encoder.py                 # ViT
src/models/encoders/code_encoder.py                   # CodeBERT
src/models/encoders/node_context.py                   # NodeContextEncoder
src/models/alignment/cross_modal_alignment.py         # 解耦双通道
src/models/decoder/figma_decoder.py                   # 三层解码器 + build_figma_json
src/models/decoder/figma_types.py                     # 节点类型枚举
src/training/trainer.py                               # FigmaGenerationModel + compute_losses
src/eval/metrics.py                                   # 全部评测指标
src/viz/figures.py                                    # 论文图表
docs/experiments/M1-report.md                         # 实验结果报告
```

---

## 9. 可移植到 GPU 集群

1. **device 抽象**：`get_device()` 自动选 CUDA/MPS/CPU
2. **autocast 抽象**：CUDA 下启用 bf16 autocast，MPS/CPU fallback nullcontext
3. **配置驱动**：所有超参 YAML，命令行 `--override` 覆盖
4. **DDP 准备**：`make_param_groups` 已分组，加 `accelerate.Accelerator()` 包装即可
5. **依赖纯净**：除 transformers/torch/playwright，无 GPU 专用 lib
