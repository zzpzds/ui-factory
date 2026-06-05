#!/usr/bin/env bash
# 主实验 M1 训完后一键生成评测 + 可视化 + 启发式基线 + 最终报告。
# 不依赖 ablation；只需 outputs/M1-baseline/best.pt 存在。
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

if [ ! -f outputs/M1-baseline/best.pt ]; then
  echo "❌ outputs/M1-baseline/best.pt 不存在，请先跑 scripts/train.py"
  exit 1
fi

echo "▶ 评测 M1 在 val..."
python scripts/evaluate.py \
  --checkpoint outputs/M1-baseline/best.pt \
  --config configs/default.yaml \
  --split val \
  --viz_dir outputs/M1-baseline/viz

echo "▶ 评测 M1 在 train（验证过拟合）..."
python scripts/evaluate.py \
  --checkpoint outputs/M1-baseline/best.pt \
  --config configs/default.yaml \
  --split train \
  --viz_dir outputs/M1-baseline/viz_train || true

echo "▶ 启发式基线 B1..."
python scripts/baseline_heuristic.py --split val --output_dir outputs/B1-heuristic

echo "▶ 生成最终报告..."
python scripts/fill_report.py \
  --runs M1-baseline,B1-heuristic \
  --output docs/experiments/M1-final.md

echo "✅ 完成。查看："
echo "   docs/experiments/M1-final.md"
echo "   outputs/M1-baseline/viz/*.png"
