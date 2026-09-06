# 设计意图恢复模型训练总结（2026-09-06）

## 1. 运行结论

- 正式弱监督训练与 Reference 微调均正常完成，进程退出码均为 0。
- 完整测试套件：`195 passed in 16.98s`。
- 正式弱监督训练最佳模型出现在第 12/30 轮，验证总损失为 `1.916799`。
- Reference 微调最佳模型出现在第 4/15 轮，选模用 AI Silver 验证损失为 `2.693099`。
- Reference 微调前后采用相同的 10 条 validation 对照；AI Silver 损失由 `2.939962` 降至 `2.693099`，改善 `8.40%`；整体损失由 `3.056655` 降至 `2.611327`，改善 `14.57%`。
- 两个阶段的后期训练损失继续下降、验证损失反而回升，均出现过拟合；下游实验应使用 `best.pt`，不应使用 `last.pt`。
- Reference test 始终保持封存，本次训练和补充对照均未读取或评估 Reference test。

## 2. 可复现环境

| 项目 | 值 |
| --- | --- |
| Git 分支 | `annotation-pilot-20260731` |
| Git 提交 | `0f57e1d04e8ab973ff5f72da14d99b5b2e48606c` |
| GPU | NVIDIA GeForce RTX 4080，16 GB |
| Python | 3.10.11 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.16.1 |
| 设备 | CUDA |
| 随机种子 | 42 |
| 可训练参数 | 4,172,207 |
| 数据拆分清单 SHA-256 | `548ae94db9305408c9c468e9bfecd6270ee3ec65fb2667130a657b52cbcbd956` |

视觉编码器和文本编码器均冻结；模型维度为 256，同时启用视觉、文本、标签、结构化特征和 DOM 图。最大节点数为 192。

## 3. 阶段一：完整弱监督训练

配置：`configs/intent/full.yaml`

| 项目 | 值 |
| --- | --- |
| 训练/验证样本 | 394 / 51 |
| 未参与训练的固定测试样本 | 47 |
| Epochs | 30 |
| Batch size / 梯度累积 | 1 / 4 |
| 学习率 | 2e-4 |
| 权重衰减 | 0.01 |
| 训练时间 | 31 分 01 秒 |
| 最佳轮次 | 12 |
| 最佳验证总损失 | 1.916799 |
| 末轮训练/验证总损失 | 0.473694 / 2.440774 |

验证总损失从第 1 轮的 `3.831314` 降至最佳值 `1.916799`，下降 `49.97%`。末轮验证损失比最佳值回升 `27.34%`，而训练损失仍持续下降，说明第 12 轮之后出现明显过拟合。

### 最佳轮次验证损失分量

| 分量 | 损失 |
| --- | ---: |
| action | 0.158941 |
| element | 0.079528 |
| group | 0.242641 |
| role | 0.028508 |
| layout_mode | 0.594330 |
| token | 0.254648 |
| tree | 1.275833 |
| layout_reg | 0.000540 |
| layout_align | 0.096284 |
| align | 0.134982 |
| total | **1.916799** |

结构树 `tree` 仍是最大损失分量，后续若继续优化，应优先检查结构监督噪声、父子关系构造以及相应损失权重。

### 产物

- `outputs/intent-full-v2/best.pt`：第 12 轮，SHA-256 `afb98a5e62dffc205301e44d37a4340f698da15288913a2ae9ee883572c4d059`
- `outputs/intent-full-v2/last.pt`：第 30 轮，SHA-256 `07779548a2bda736eacc43ae9513b83d77c4411d987ad08828d9a9ef7c0c7690`
- `outputs/intent-full-v2/train.jsonl`：30 条逐轮记录
- `outputs/intent-full-v2/run_manifest.json`：运行时间与 checkpoint 哈希

## 4. 阶段二：Reference 微调

配置：`configs/intent/reference_finetune.yaml`，初始化自阶段一 `best.pt`。

| 项目 | 值 |
| --- | --- |
| 训练样本 | 30（7 Human Gold、23 AI Silver） |
| 验证样本 | 10（1 Human Gold、9 AI Silver） |
| Epochs | 15 |
| Batch size / 梯度累积 | 1 / 4 |
| 学习率 | 5e-5 |
| 原始来源权重 | Human Gold 1.0、AI Silver 0.5 |
| Checkpoint 选模层 | AI Silver validation |
| 训练时间 | 1 分 49 秒 |
| 最佳轮次 | 4 |
| 最佳 AI Silver 验证损失 | 2.693099 |
| 最佳整体验证损失 | 2.611327 |
| 最佳 Human Gold 验证损失 | 1.875378 |
| 末轮训练/AI Silver 验证损失 | 0.617336 / 2.947360 |

### 微调前后同条件对照

| Reference validation 层 | 微调前 | 微调后最佳 | 相对改善 |
| --- | ---: | ---: | ---: |
| 整体（10 条） | 3.056655 | 2.611327 | 14.57% |
| AI Silver（9 条，正式选模层） | 2.939962 | 2.693099 | 8.40% |
| Human Gold（1 条） | 4.106900 | 1.875378 | 54.34% |

Human Gold validation 只有 1 条，其改善幅度方差很大，不应单独作为稳健结论；AI Silver 的 9 条验证结果是配置规定的 checkpoint 选择依据。

### 产物

- `outputs/intent-reference-finetune-v2/best.pt`：第 4 轮，SHA-256 `4474b91363d818dd521e26e4254c293bf85168da003e58e3c7b8817a5e9b42ca`
- `outputs/intent-reference-finetune-v2/last.pt`：第 15 轮，SHA-256 `70c96f4ec6ac283e464704f81560e36296bcc21829e77f39acb6ddefc24ed781`
- `outputs/intent-reference-finetune-v2/train.jsonl`：15 条逐轮记录
- `outputs/intent-reference-finetune-v2/run_manifest.json`：运行时间与 checkpoint 哈希

## 5. 建议

1. 后续 Reference test 的最终评估只使用 `outputs/intent-reference-finetune-v2/best.pt`，并在所有实验方案、消融配置和评价脚本冻结后一次性执行。
2. 正式论文中将弱监督阶段的第 12 轮和 Reference 微调阶段的第 4 轮报告为验证集选出的 checkpoint，避免汇报末轮模型。
3. 若继续调参，优先处理 `tree` 结构损失；Reference 数据量较小，不建议仅凭当前 1 条 Human Gold validation 得出细粒度结论。
