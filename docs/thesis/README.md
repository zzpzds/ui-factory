# 论文研究与实验索引

## 文档顺序

1. `01_research_question_brief.md`：研究问题、假设与范围；
2. `02_methodology_blueprint.md`：研究设计、金标准与统计方案；
3. `03_system_design_blueprint.md`：模型、IR 与求解器设计；
4. `04_phase1_checkpoint.md`：Stage 1 冻结记录；
5. `05_annotation_protocol.md`：人工标注规范；
6. `06_experiment_protocol.md`：基线、消融和实验顺序；
7. `07_implementation_status.md`：当前代码完成度；
8. `08_verified_literature_map.md`：已核验文献与差异定位；
9. `09_thesis_outline.md`：论文写作结构与证据占位；
10. `10_phase2_checkpoint.md`：进入 pilot 前检查点；
11. `11_pilot_data_report.md`：500 页 Pilot 的生产与质量结论；
12. `12_ai_proxy_annotation_protocol.md`：AI 代理盲态、来源和披露协议；
13. `13_ai_proxy_pilot_report.md`：AI 代理标注与人机跨来源初步结果；
14. `14_human_ai_adjudication_sop.md`：逐项差异复核、审核凭证与金标准导出门槛。
15. `15_adjudicated_gold_report.md`：10 页最终 Gold 的数量、描述性指标与研究限制；
16. `16_formal_gold_v1_sampling_and_sop.md`：DesignIntent Reference v1 的双层来源、筛选与使用边界；
17. `17_ai_assisted_annotation_protocol.md`：未完成人工复核即终止的 AI 辅助历史方案；
18. `18_ai_multiview_silver_protocol.md`：50 页 AI 多视角银标的生成、审计与论文披露协议。
19. `19_reference_test_access_audit.md`：Reference test 的实际访问边界、已知事件与解封语义。

## 新主链路

```bash
source .venv/bin/activate

# 冻结 500 页 Pilot 并采集富 PageGraph
python scripts/select_pilot_samples.py \
  --data_dir data/processed \
  --num_samples 500 \
  --output data/intent_pilot_manifest.json
python scripts/render_pages.py \
  --data_dir data/processed \
  --manifest data/intent_pilot_manifest.json \
  --report outputs/intent-pilot-500/render-report.json
python scripts/finalize_pilot_manifest.py

# 生成并校验弱标签
python scripts/prepare_intent_data.py \
  --data_dir data/processed \
  --manifest data/intent_pilot_success_manifest.json
python scripts/validate_intent_data.py \
  --data_dir data/processed \
  --annotation_name weak_intent.json \
  --manifest data/intent_pilot_success_manifest.json

# 防泄漏切分与近重复检查
python scripts/split_intent_dataset.py \
  --data_dir data/processed \
  --sample_manifest data/intent_pilot_success_manifest.json \
  --output data/intent_pilot_split.json
python scripts/detect_near_duplicates.py \
  --data_dir data/processed \
  --split_manifest data/intent_pilot_split.json

# 历史双人盲标工作台（未完成第二位真人标注）
python scripts/serve_intent_annotation.py \
  --package_dir data/annotations/intent_pilot_v1 \
  --port 8765

# 无第二位真人时的独立 AI 代理轨道（不能解释为双人一致性）
python scripts/generate_ai_proxy_annotations.py \
  --package_dir data/annotations/intent_pilot_v1
python scripts/evaluate_annotation_agreement.py \
  --package_dir data/annotations/intent_pilot_v1 \
  --left-annotator annotator_a \
  --right-annotator annotator_ai \
  --comparison-kind human-ai \
  --allow-incomplete \
  --output outputs/intent-pilot-500/human-ai-agreement.json
python scripts/prepare_human_ai_adjudication.py

# 仅在单名人工标注者提交审核声明并定稿后执行；待审核草稿会被拒绝
python scripts/finalize_human_ai_adjudication.py

# 创建初始 60 页清单；后两步执行质量替换并生成当前 Reference v1
python scripts/create_formal_gold_package.py
python scripts/replace_formal_reference_samples.py
python scripts/prepare_ai_reference_gold.py
python scripts/serve_intent_annotation.py \
  --package_dir data/annotations/intent_gold_v1 \
  --port 8766

# Reference v1 完整性门禁
python -m pytest tests/test_formal_gold_package.py -q

# 弱监督训练与来源分层 Reference 微调
python scripts/train_intent.py --config configs/intent/full.yaml
python scripts/train_intent.py \
  --config configs/intent/reference_finetune.yaml

# 冻结前只允许 validation；正式 test 命令见 06_experiment_protocol.md
python scripts/evaluate_intent.py \
  --checkpoint outputs/intent-reference-finetune-v2/best.pt \
  --reference-package data/annotations/intent_gold_v1 \
  --split validation \
  --output outputs/intent-reference-v1/validation.json
```

## 关键约束

- 正式训练必须提供 `data.split_manifest`，禁止内部随机页面切分；
- Reference v1 必须区分 10 页 `human_ai_adjudicated_gold` 与 50 页
  `ai_multiview_silver`，不能把 60 页统称人工 Gold；
- Human Gold test 只有 2 页，相关结果只能作为探索性证据；AI Silver 指标只表示
  模型与 AI 参考标签的一致性；
- AI 代理只能用于差异检测和人工复核，不能冒充第二位真人标注员；
- 旧的 45 页 AI 辅助与 5 页盲标对照方案未完成人工执行，只作为历史审计记录；
- 混合 validation/test 指标必须按标签来源分层报告；
- Reference checkpoint 必须记录完整 `split x tier`、manifest hash、来源权重和运行环境；
- 正式 test 前必须先完成 checkpoint/model 与 manifest 语义预检，首次 test 会排他创建
  固定 unseal；正式 test 强制校验内容哈希，所有评测入口都必须显式指定 split 和对应
  split manifest；
- unseal 只审计模型训练/评测消费，不代表 Reference 构建过程具备开发者盲法；
- 三个随机种子固定为 `42`、`123`、`2026`；
- 测试集只在主模型、阈值和排除规则冻结后用于正式评测；这不表示 Reference v1
  构建过程对 2 页 Human Gold 保证了开发者盲法；
- smoke 指标只证明链路连通，不作为研究结果。
