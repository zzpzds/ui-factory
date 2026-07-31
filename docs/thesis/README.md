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
10. `10_phase2_checkpoint.md`：进入 pilot 前检查点。
11. `11_pilot_data_report.md`：500 页 Pilot 的生产与质量结论。

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

# 盲双人试标注
python scripts/serve_intent_annotation.py \
  --package_dir data/annotations/intent_pilot_v1 \
  --port 8765

# 训练与测试
python scripts/train_intent.py --config configs/intent/full.yaml
python scripts/evaluate_intent.py \
  --checkpoint outputs/intent-full-v1/best.pt \
  --annotation_name gold_intent.json \
  --split test
```

## 关键约束

- 正式训练必须提供 `data.split_manifest`，禁止内部随机页面切分；
- 主测试只使用人工 `gold_intent.json`，弱标签不能作为论文真值；
- 三个随机种子固定为 `42`、`123`、`2026`；
- 测试集在阈值、排除规则和模型选择冻结前不得解封；
- smoke 指标只证明链路连通，不作为研究结果。
