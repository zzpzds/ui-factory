# 项目进度与下一步任务

> **更新时间**：2026-06-05
> **当前阶段**：可行性验证完成（论文工作计划第三阶段中段）
> **下一步**：迁 GPU 集群继续训练 + 消融 + 数据扩充

---

## 一、已完成（本机 MPS 阶段）

### 1.1 数据预处理
- [x] **T0.1** 扩展 `scripts/render_pages.py`：通过 Playwright 收集 `getComputedStyle` 9 个字段写到 `styles.json`
- [x] **T0.2** 重渲已有 100 条样本，95 条成功生成 `styles.json`（5 条页面加载超时，可接受）

### 1.2 核心 Bug 修复（论文级）
- [x] **P1** `cross_modal_alignment.py` 解耦双通道：融合用 softmax MultiheadAttention，监督用独立 Bilinear sigmoid+BCE+InfoNCE。修复了原版"softmax 行和=1 × 多对多标签 BCE" 的数学错误
- [x] **P2** `dataset.py` 改 padding+node_mask，`webpage_collate_fn` 真支持 batch>1
- [x] **P3** 样式监督全链路打通（`render_pages` 写 styles.json → dataset 编码 → loss 计算）
- [x] **P4** 实现真正的 `RecursiveStructDecoder`（指针网络 + 深度位置嵌入 + transformer encoder）
- [x] **P5** 新增 `NodeContextEncoder`（depth/sibling 位置嵌入 + 2 层 self-attn）
- [x] **P6** `max_nodes` 从 50 提到 96
- [x] **P7** 截断改为 BFS 保留连通子树（`_bfs_truncate` + `_reindex_parents`）
- [x] **P8** `build_figma_json` 真用模型预测的 parent_logits 建树（旧版绕过模型）
- [x] **P9** 删除 `trainer.train()`，单一训练入口在 `scripts/train.py`
- [x] **P10** TEXT 从 LEAF_TYPES 移除；训练时 `apply_leaf_prior=False`，推理时启用——避免 GT 父被屏蔽 → CE 爆炸
- [x] **P11** `iterative_inference` 解决 train/inference 在 depth/parents 上的 mismatch
- [x] **P12** `style_reg` clamp 到 [0, 1]，避免 border-radius:9999px 极值导致 MSE 爆炸

### 1.3 训练系统
- [x] **T3.1** `scripts/train.py` device-agnostic + grad_accum + cosine warmup + psutil 内存监控 + 散热 sleep + autocast 抽象
- [x] **T5.1** YAML 配置系统 `configs/default.yaml`，支持 `--config` + `--override key=value`
- [x] **T5.2** `scripts/train_accelerate.py` 用 `Accelerator()` 包装，DDP/AMP-ready
- [x] **GPU 配置示例** `configs/gpu.yaml`（A100/H100 类）

### 1.4 评测与可视化
- [x] **T3.2** `src/eval/metrics.py`：type_acc / parent_f1 / tree_edit / style_reg_mae / color_de / cls_acc / 混淆矩阵
- [x] **T3.4** `src/viz/figures.py`：loss 曲线 / 混淆矩阵 / attention 热图 / 树并排 / 深度分布
- [x] **scripts/evaluate.py** 三种 structure_mode（model/heuristic/hybrid）+ heuristic-init 选项

### 1.5 主实验 M1
- [x] **T4.1** 30 epoch on 95 样本（train=80, val=15），约 45 分钟
  - val type_acc=**0.999**, total=2.75→1.29
  - val align_bce 1.21→0.76（核心修复有效）
  - 三种 structure_mode 都已评测
- [x] **T4.3** 启发式基线 B1（`scripts/baseline_heuristic.py`），val parent_f1=0.458, tree_edit=12.4

### 1.6 文档
- [x] 设计文档：`docs/superpowers/specs/2026-06-04-figma-generation-design.md`
- [x] 实验报告（自动生成）：`docs/experiments/M1-final.md`
- [x] 实验大纲（人写）：`docs/experiments/M1-report.md`

---

## 二、待做（GPU 集群阶段）

### 2.1 优先级 P0：扩数据 + 重训
- [ ] **D1** 扩样本：`python scripts/download_data.py --num_samples 10000`
- [ ] **D2** 渲染：`python scripts/render_pages.py --data_dir data/processed`（注意 GPU 集群上 chromium 依赖）
- [ ] **D3** 重训：`accelerate launch scripts/train_accelerate.py --config configs/gpu.yaml`
  - 期望 parent_f1 从 0.04 提到 0.40+（验证数据规模假说）
  - 100 epoch / max_nodes=192 / batch=4 / 解冻 codebert 后 4 层

### 2.2 优先级 P1：消融实验
- [ ] **A1** 对齐监督消融：`bash scripts/run_ablations.sh`（已写好，4 组 × 30 epoch 串行）
  - A1a 完全去掉对齐监督
  - A1b 只 BCE 不要 InfoNCE
  - A4a max_nodes=50
  - A4b max_nodes=128

### 2.3 优先级 P2：解码器架构改进（论文创新）
- [ ] **R1** 重新设计 `RecursiveStructDecoder` 解决 train/inference depth mismatch：
  - 选项 A：去掉 `depth_emb`，让模型直接靠 ctx 学结构
  - 选项 B：做 **scheduled sampling**——训练后期逐渐用预测 depth 代替 GT depth
  - 选项 C：让模型独立预测 depth（多任务）
- [ ] **R2** 引入更强的视觉感知：把 `boxes` 嵌入也加到 ctx 里（位置先验）
- [ ] **R3** 训练样式 head 时增加 perceptual loss（预测样式渲染回截图后的 SSIM）

### 2.4 优先级 P3：评测增强
- [ ] **E1** 渲染回比对：`Figma JSON → HTML/CSS → Playwright 截图 → 与原图 SSIM/DreamSim`（`src/eval/render_back.py` 占位）
- [ ] **E2** 真正接入 BuilderIO/figma-html npm 包做对比（不是 pure-Python 等价版）
- [ ] **E3** 与 GPT-4V few-shot 对比（API 成本）

### 2.5 优先级 P4：论文章节填充
- [ ] **W1** 第 4 章：把 M1 + 扩数据后的结果填入实验章节
- [ ] **W2** 第 4.5 节：写 hybrid 架构论述（model 类型/样式 + heuristic 结构）
- [ ] **W3** 第 5 章：Future work 写"数据规模假说曲线"
- [ ] **W4** 把 `outputs/M1-baseline/viz/*.png` 整合到论文图表

---

## 三、GPU 端启动手册

### 3.1 环境准备
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd ui-factory

# Python 3.10+ 推荐
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Playwright（用于渲染数据）
python -m playwright install chromium
# 集群无 GUI 环境若 chromium 报错，加：
# python -m playwright install-deps  # 需 sudo

# 准备 HF 模型缓存（首次需联网）
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE
python -c "from transformers import ViTModel, AutoModel; ViTModel.from_pretrained('google/vit-base-patch16-224'); AutoModel.from_pretrained('microsoft/codebert-base')"
```

### 3.2 修改硬编码路径
`src/models/encoders/code_encoder.py:5` 把 `CODEBERT_MODEL_PATH` 改成 GPU 端的实际缓存路径：
```python
CODEBERT_MODEL_PATH = "microsoft/codebert-base"  # 让 HF cache 自动找
# 或显式
# CODEBERT_MODEL_PATH = os.environ.get("CODEBERT_PATH", "microsoft/codebert-base")
```

### 3.3 一键复现 M1 结果（小数据冒烟）
```bash
# 已有 data/sampled/ 在仓库中（25MB，100 条样本）
# 直接训练（GPU 上 30 epoch 约 5 分钟）
accelerate config              # 第一次：选 No DDP / fp16 即可
accelerate launch scripts/train_accelerate.py --config configs/default.yaml
python scripts/evaluate.py --checkpoint outputs/M1-baseline/best.pt --config configs/default.yaml --split val
```

### 3.4 全规模训练
```bash
# 1) 扩数据到 1 万条
python scripts/download_data.py --num_samples 10000     # 流式下载 HTML
python scripts/render_pages.py --data_dir data/processed --sleep 0.0  # 渲染

# 2) 修改 configs/gpu.yaml 的 data_dir
sed -i 's|data/sampled|data/processed|' configs/gpu.yaml

# 3) DDP 训练（多卡）
accelerate config              # 选 multi-GPU + bf16
accelerate launch scripts/train_accelerate.py --config configs/gpu.yaml

# 4) 评测
python scripts/evaluate.py --checkpoint outputs/M1-gpu/best.pt --config configs/gpu.yaml --split val --structure_mode hybrid
```

### 3.5 消融实验（约 3 小时）
```bash
bash scripts/run_ablations.sh
python scripts/fill_report.py \
  --runs M1-baseline,A1a-no-align,A1b-bce-only,A4a-max50,A4b-max128,B1-heuristic \
  --output docs/experiments/M1-final.md
```

---

## 四、不会跟踪到 GitHub 的内容（已在 `.gitignore`）

| 类型 | 路径 | 大小 | 原因 |
|------|------|------|------|
| 虚拟环境 | `.venv/` | 1.6 GB | 在 GPU 端重建 |
| 模型权重 | `*.pt`（除 `data/sampled/*`）| ~3.5 GB | 重训而非传输 |
| 旧 / 临时 outputs | `outputs/_demo_smoke/`、`outputs/_real_smoke/`、`outputs/best_model.pt` | ~3 GB | 冗余 |
| 原始数据 dump | `data/raw/`、`data/processed/` | 211 MB | 由脚本重生成 |
| Playwright 缓存 | `.playwright-cli/` | 1.4 MB | 集群独立装 |
| 各种 cache | `__pycache__/`、`.pytest_cache/`、`.ipynb_checkpoints/` | 杂 | 自动 |
| 日志 / wandb | `*.log`、`wandb/` | 杂 | 不必要 |

预计上传到 GitHub 的总大小约 **27 MB**：
- 代码 + 配置 + 文档：~1 MB
- `data/sampled/`：~25 MB（100 条小样本，含 page.html + screenshot.png + 张量）
- `outputs/M1-baseline/` 小产物（loss 曲线 PNG、metrics JSON、figma_e*.json、train.log.jsonl）：~1.5 MB

---

## 五、关键命令速查

| 命令 | 用途 |
|------|------|
| `python scripts/test_pipeline.py` | demo 端到端冒烟 |
| `python scripts/train.py --config configs/default.yaml` | 本机训练 |
| `accelerate launch scripts/train_accelerate.py --config configs/gpu.yaml` | GPU 集群训练 |
| `python scripts/evaluate.py --checkpoint <path> --config <cfg> --split val --structure_mode hybrid` | 评测 + 可视化 |
| `python scripts/baseline_heuristic.py --split val` | 启发式基线 |
| `bash scripts/finalize_m1.sh` | M1 训完后的一键评测 + 报告 |
| `bash scripts/run_ablations.sh` | 跑全部消融 |
| `python scripts/fill_report.py --runs <run1>,<run2> --output <path>` | 自动生成对比表 |
