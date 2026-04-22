# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

硕士论文实验代码：**从网页截图 + HTML 源码生成 Figma JSON 设计稿数据**，用于解决前端设计 AI 训练数据稀缺问题。整个仓库是一个研究性 pipeline，不是生产项目 —— 实验性代码、demo 路径、硬编码假设很常见。

## 常用命令

所有脚本都从项目根目录运行（脚本内自己做 `sys.path.insert`）：

```bash
# 激活环境
source .venv/bin/activate

# 1) 下载 WebCode2M HTML（流式）
python scripts/download_data.py --num_samples 1000

# 2) 用 Playwright 渲染 HTML → 截图 + 节点框 + 节点文本
python scripts/render_pages.py --data_dir data/processed

# 3) 从已渲染数据中随机采样
python scripts/sample_data.py --num 100

# 4) 端到端验证 pipeline（使用内置 demo 样本，无需外部数据）
python scripts/test_pipeline.py

# 5) 训练
python scripts/train.py --data_dir data/sampled --epochs 10 --batch_size 4
python scripts/train.py --demo                          # 单样本冒烟训练
```

## 核心架构

数据流（单条样本端到端）：

```
HTML 源码 ──► parse_html_nodes ──► list[节点文本串]
                                        │
网页截图 ──► PIL.Image                   │
                                        ▼
                    VisualEncoder (ViT, google/vit-base-patch16-224)
                        └─► [B, 196, 768]  （14×14 patch，去掉 CLS）
                    CodeEncoder (microsoft/codebert-base, 本地路径)
                        └─► [B, N, 768]    （每个 DOM 节点的 CLS token）
                                        │
                                        ▼
                    CrossModalAlignment (nn.MultiheadAttention)
                        query=code, key/value=visual
                        └─► fused [B, N, 768] + attn [B, N, 196]
                        监督信号：节点框 vs patch 框的 IoU → BCE 损失
                                        │
                                        ▼
                    FigmaDecoder（三层，全部共享 fused_features）
                        ├─ FigmaNodeTypeClassifier  → 节点类型 logits
                        ├─ FigmaStructurePlanner    → 父子关系矩阵 [B, N, N]
                        └─ FigmaStylePredictor      → 样式属性回归
                                        │
                                        ▼
                          build_figma_json → Figma JSON
```

**坐标系统一**：图像统一 resize 到 224×224，节点渲染框也映射到 224×224 坐标系（见 `render_pages.py` 的 `scale_box`），patch 框固定为 14×14=196 个 16×16 格子（`get_patch_boxes`）。IoU 标签和 attention 权重都在这个坐标系里对齐。

**训练目标的弱监督来源**（`trainer.build_targets`）：
- 节点类型 ← HTML 标签经 `TAG_TO_NODE_TYPE` 映射
- 父子关系 ← 边界框的**包含关系**（`is_containing`），不是 DOM 真实父子
- 样式标签目前没有自动构造

## 数据目录约定

```
data/
├── processed/       # download_data.py 写入；render_pages.py 读写
│   └── NNNN/
│       ├── page.html           # 下载脚本写入
│       ├── meta.json
│       ├── screenshot.png      # 渲染脚本写入
│       ├── nodes.pt            # [N, 4] 节点框（224 坐标系）
│       └── node_texts.json     # list[str]，与 nodes.pt 同序
├── sampled/         # sample_data.py 的产物，训练默认读这里
└── figma/raw/       # 预留给 Figma 社区数据
```

`WebpageDataset._scan` 只收集**同时有** `screenshot.png + nodes.pt + node_texts.json` 的目录 —— 没跑过 `render_pages.py` 的目录会被静默跳过。

## 环境与非显然的约束

- **离线 HF**：`scripts/train.py` 和 `test_pipeline.py` 开头强制 `TRANSFORMERS_OFFLINE=1` / `HF_HUB_OFFLINE=1`。首次使用需要先把模型下载到本地缓存，否则会报 `local_files_only=True` 找不到文件。
- **CodeBERT 硬编码路径**：`src/models/encoders/code_encoder.py` 写死 `/Users/didi/.cache/modelscope/microsoft/codebert-base`。换机器要改这里。
- **HF 镜像**：`download_data.py` 设置 `HF_ENDPOINT=https://hf-mirror.com`，`.env.example` 也提供了这项。
- **设备**：训练默认 `torch.device("mps" if available)`，训练循环里有 `torch.mps.empty_cache()` 调用 —— 这是 Mac 专用路径。CUDA/CPU 也能跑，但 MPS 是主路径。
- **batch_size 实际为 1**：`webpage_collate_fn` 里 `assert len(batch) == 1`。`--batch_size 4` 会直接 assert 失败。要想真正批量化，需要先处理 N（每张图节点数）不同带来的 padding。
- **max_nodes=50 截断**：`WebpageDataset` 超过 50 个节点的样本会被截断，避免 OOM。动架构参数前先留意这个上限。
- **Playwright 关闭外部资源**：`render_pages.py` 的 `page.route` 屏蔽了 image/stylesheet/font/media，截图是**无样式的骨架**。这是有意为之（网络超时 + 速度），但意味着截图质量不接近真实浏览效果。
- **结构规划器的 leaf mask**：`FigmaStructurePlanner` 里 `TEXT/RECTANGLE/ELLIPSE/LINE/VECTOR` 被硬编码为叶子节点，无法作为父节点。改节点类型枚举时要同步这里。

## 代码入口对照

| 想改的东西 | 入口文件 |
| --- | --- |
| 视觉编码（ViT 模型 / patch 输出） | `src/models/encoders/visual_encoder.py` |
| 代码编码（CodeBERT / 节点文本格式） | `src/models/encoders/code_encoder.py` + `src/data/preprocessing.py::parse_html_nodes` |
| 对齐损失 / IoU 阈值 | `src/models/alignment/cross_modal_alignment.py` |
| Figma 节点类型、HTML→Figma 映射 | `src/models/decoder/figma_types.py` |
| 解码器结构（三层 head） | `src/models/decoder/figma_decoder.py` |
| 损失权重、训练循环 | `src/training/trainer.py`（模块内 `train()`）+ `scripts/train.py`（CLI） |
| 数据加载 / 目标构建 | `src/data/dataset.py` + `src/training/trainer.py::build_targets` |

`scripts/train.py` 和 `src/training/trainer.py::train()` **是两条并行的训练入口**，`scripts/train.py` 更完整（tqdm、scheduler、eval、生成 Figma JSON）。修训练逻辑优先改脚本那条，别忘了两边可能会漂移。

## Notebook

`notebooks/` 下是研究笔记本（ViT、CodeBERT、cross-attention 的最小原型），不在主训练路径上。`learning/*.py` 同样是学习用脚本，不要在主 pipeline 里引用它们。

## 沟通约定

- 和用户用中文沟通；代码注释也是中文。
- 这是论文实验代码，优先保持可读性和可解释性，不追求生产工程强度（容错、日志级别、CI 等）。
