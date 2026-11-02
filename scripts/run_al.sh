#!/usr/bin/env bash
# Active learning (docs/al/RUNBOOK.md): train each al_<selection> variant from the round-1 model as the pilot, score
# it on Val-B with thresholds chosen on Val-A-calib (tools/al_score.py), then rebuild the report (tools/al_report.py).
#
#   bash scripts/run_al.sh dot1                 # B=500: random x3, margin, margin_k, margin_bal + oracle (~7 run)
#   bash scripts/run_al.sh dot2                 # B=250, 850 (random x3, margin, margin_k, margin_bal, entropy)
#   bash scripts/run_al.sh al_margin_k_B500_r2  # vòng 2: tự tạo tập chọn + gói từ model vòng 1, rồi train
#   bash scripts/run_al.sh dot1 --dry-run       # chỉ dựng dữ liệu (CPU), không train
#   bash scripts/run_al.sh test al_margin_k_B500 al_random_B500_s0 --confirm-test   # test2019, MỘT lần mỗi run
#
# Bị ngắt thì chạy lại đúng lệnh: run đã có stages/<tên>.json được bỏ qua, điểm đã có cũng bỏ qua.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=$(cd "${RELEARN_DATA_ROOT:-$REPO/../relearn_fix_v2}" && pwd)
SETTINGS=${SETTINGS:-$REPO/configs/training/update_model_relearn_fix_v2.json}
CAMPAIGN=${CAMPAIGN:-$REPO/data/training/update-model/relearn-fix-v2}
DOT1=(al_random_B500_s0 al_random_B500_s1 al_random_B500_s2 al_margin_B500 al_margin_k_B500 al_margin_bal_B500
      al_oracle)
DOT2=()
for b in 250 850; do
    DOT2+=(al_random_B${b}_s0 al_random_B${b}_s1 al_random_B${b}_s2 al_margin_B${b} al_margin_k_B${b}
           al_margin_bal_B${b} al_entropy_B${b})
done
DOT2+=(al_entropy_B500)

DRY=0; TEST=0; CONFIRM=0; VARIANTS=()
for a in "$@"; do
    case $a in
        --dry-run) DRY=1 ;;
        --confirm-test) CONFIRM=1 ;;
        test) TEST=1 ;;
        dot1) VARIANTS+=("${DOT1[@]}") ;;
        dot2) VARIANTS+=("${DOT2[@]}") ;;
        al_*) VARIANTS+=("$a") ;;
        *) echo "không hiểu tham số: $a" >&2; exit 2 ;;
    esac
done
[[ ${#VARIANTS[@]} -gt 0 ]] || { echo "cần dot1, dot2 hoặc tên al_<selection>" >&2; exit 2; }

cd "$REPO"
export RELEARN_DATA_ROOT=$DATA
mkdir -p data/training/update-model
LOG=data/training/update-model/run_al.log
um() { python -m autocheckout.update_model --mode improve --settings "$SETTINGS" "$@" 2>&1 | tee -a "$LOG"; }

if [[ $TEST == 1 ]]; then
    [[ $CONFIRM == 1 ]] || { echo "test2019 chỉ chấm MỘT lần: thêm --confirm-test khi nhóm đã chốt" >&2; exit 2; }
    for v in "${VARIANTS[@]}"; do
        um --stage final_test --variant "$v" --confirm-test
        python -m tools.al_score --release "$DATA" --name "$v" \
            --pred "$CAMPAIGN/runs/EXP-FB-FT-$v/task_5/pred_val.npz" \
            --test-pred "$CAMPAIGN/runs/EXP-FB-FT-$v/task_5/pred_test.npz" 2>&1 | tee -a "$LOG"
    done
    exit 0
fi

for v in "${VARIANTS[@]}"; do
    if [[ ! -f $DATA/variants/$v/train_task_5_fix.json ]]; then
        if [[ $v == *_r2 ]]; then
            python -m tools.al_round2 --release "$DATA" --campaign "$CAMPAIGN" --from "${v%_r2}" 2>&1 | tee -a "$LOG"
        else
            python -m tools.al_build_variant --release "$DATA" --selection "${v#al_}" 2>&1 | tee -a "$LOG"
        fi
    fi
    if [[ $DRY == 1 ]]; then
        um --stage data --variant "$v"
        continue
    fi
    echo "=== $(date -u +%FT%TZ) $v" | tee -a "$LOG"
    um --stage al --variant "$v"
    PRED=$CAMPAIGN/runs/EXP-FB-FT-$v/task_5/pred_val.npz
    if [[ -f $PRED && ! -f $DATA/active/scored/$v.json ]]; then
        python -m tools.al_score --release "$DATA" --name "$v" --pred "$PRED" 2>&1 | tee -a "$LOG"
    fi
done
[[ $DRY == 1 ]] || python -m tools.al_report --release "$DATA" --campaign "$CAMPAIGN" 2>&1 | tail -3 | tee -a "$LOG"
