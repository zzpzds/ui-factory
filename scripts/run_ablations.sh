#!/usr/bin/env bash
# 串行跑所有消融实验。用户要求"不要满载，留资源冗余" → 严禁并行。
# 跑前确认 outputs/M1-baseline 已经训完（或在跑时不要并发）。
set -e
cd "$(dirname "$0")/.."

source .venv/bin/activate

# 每条 = "config_path|out_dir"
EXPERIMENTS=(
  "configs/ablations/A1a_no_align.yaml|outputs/A1a-no-align"
  "configs/ablations/A1b_bce_only.yaml|outputs/A1b-bce-only"
  "configs/ablations/A4a_max50.yaml|outputs/A4a-max50"
  "configs/ablations/A4b_max128.yaml|outputs/A4b-max128"
)

for entry in "${EXPERIMENTS[@]}"; do
  cfg="${entry%%|*}"
  outd="${entry##*|}"
  name="$(basename "$outd")"
  echo "▶  跑消融: $name (config=$cfg, output=$outd)"
  python scripts/train.py --config "$cfg"
  if [ -f "$outd/best.pt" ]; then
    python scripts/evaluate.py \
      --checkpoint "$outd/best.pt" \
      --config "$cfg" \
      --split val \
      --viz_dir "$outd/viz"
  fi
  echo "✅ $name 完成"
  sleep 5
done

# 启发式基线
python scripts/baseline_heuristic.py --split val --output_dir outputs/B1-heuristic

# 汇总报告
python scripts/fill_report.py \
  --runs M1-baseline,A1a-no-align,A1b-bce-only,A4a-max50,A4b-max128,B1-heuristic \
  --output docs/experiments/M1-final.md

echo "全部消融完成"
