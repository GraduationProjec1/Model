#!/usr/bin/env bash
# Active learning (docs/al/PLAN.md, A6-A8): train each al_<selection> variant as the pilot, then score it on Val-B
# with thresholds chosen on Val-A-calib (tools/al_score.py). Needs the variants built on the data folder first:
#   python -m tools.al_select --release ../relearn_fix_v2
#   python -m tools.al_build_variant --release ../relearn_fix_v2 --selection <names...>
#
#   bash scripts/run_al.sh dot1               # đợt 1: B=500 (random x3, margin, margin_k, margin_bal) + oracle
#   bash scripts/run_al.sh dot1 --dry-run     # chỉ dựng dữ liệu (CPU), không train
#   bash scripts/run_al.sh al_margin_B250 ... # các biến thể chỉ định
#
# Bị ngắt thì chạy lại đúng lệnh: run đã có stages/<tên>.json được bỏ qua (update_model), điểm đã có cũng bỏ qua.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=$(cd "${RELEARN_DATA_ROOT:-$REPO/../relearn_fix_v2}" && pwd)
SETTINGS=${SETTINGS:-$REPO/configs/training/update_model_relearn_fix_v2.json}
CAMPAIGN=${CAMPAIGN:-$REPO/data/training/update-model/relearn-fix-v2}
DOT1=(al_random_B500_s0 al_random_B500_s1 al_random_B500_s2 al_margin_B500 al_margin_k_B500 al_margin_bal_B500
      al_oracle)

DRY=0; VARIANTS=()
for a in "$@"; do
    case $a in
        --dry-run) DRY=1 ;;
        dot1) VARIANTS+=("${DOT1[@]}") ;;
        al_*) VARIANTS+=("$a") ;;
        *) echo "không hiểu tham số: $a" >&2; exit 2 ;;
    esac
done
[[ ${#VARIANTS[@]} -gt 0 ]] || { echo "cần dot1 hoặc tên al_<selection>" >&2; exit 2; }
for v in "${VARIANTS[@]}"; do
    [[ -f $DATA/variants/$v/train_task_5_fix.json ]] || {
        echo "thiếu $DATA/variants/$v: python -m tools.al_build_variant --selection ${v#al_}" >&2; exit 1; }
done

cd "$REPO"
export RELEARN_DATA_ROOT=$DATA
mkdir -p data/training/update-model
LOG=data/training/update-model/run_al.log
for v in "${VARIANTS[@]}"; do
    if [[ $DRY == 1 ]]; then
        python -m autocheckout.update_model --mode improve --settings "$SETTINGS" --stage data --variant "$v" \
            2>&1 | tee -a "$LOG"
        continue
    fi
    echo "=== $(date -u +%FT%TZ) $v" | tee -a "$LOG"
    python -m autocheckout.update_model --mode improve --settings "$SETTINGS" --stage al --variant "$v" \
        2>&1 | tee -a "$LOG"
    PRED=$CAMPAIGN/runs/EXP-FB-FT-$v/task_5/pred_val.npz
    if [[ -f $PRED && ! -f $DATA/active/scored/$v.json ]]; then
        python -m tools.al_score --release "$DATA" --name "$v" --pred "$PRED" 2>&1 | tee -a "$LOG"
    fi
done
[[ $DRY == 1 ]] || python -m tools.al_report --release "$DATA" --campaign "$CAMPAIGN" 2>&1 | tee -a "$LOG"
