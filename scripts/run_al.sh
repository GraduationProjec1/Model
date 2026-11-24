#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=$(cd "${AC_DATA:-$REPO/../autocheckout-data}" && pwd)
REL=$DATA/relearn
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
export AC_DATA=$DATA
mkdir -p data/training/update-model
LOG=data/training/update-model/run_al.log
um() { python -m autocheckout.update_model --mode improve --settings "$SETTINGS" "$@" 2>&1 | tee -a "$LOG"; }
FAILED=()
failed() { echo "[LỖI] $1 (xem $LOG)" | tee -a "$LOG" >&2; FAILED+=("$1"); }
finish() {
    [[ ${#FAILED[@]} -eq 0 ]] && return 0
    echo "[LỖI] ${#FAILED[@]} run lỗi: ${FAILED[*]}" | tee -a "$LOG" >&2
    exit 1
}

if [[ $TEST == 1 ]]; then
    [[ $CONFIRM == 1 ]] || { echo "test2019 chỉ chấm MỘT lần: thêm --confirm-test khi nhóm đã chốt" >&2; exit 2; }
    for v in "${VARIANTS[@]}"; do
        um --stage final_test --variant "$v" --confirm-test || { failed "$v test2019"; continue; }
        if [[ -f $REL/active/scored_test/$v.json ]]; then
            echo "$v: test2019 đã chấm ($REL/active/scored_test/$v.json), bỏ qua al_score" | tee -a "$LOG"
            continue
        fi
        python -m tools.al_score --name "$v" \
            --pred "$CAMPAIGN/runs/EXP-FB-FT-$v/task_5/pred_val.npz" \
            --test-pred "$CAMPAIGN/runs/EXP-FB-FT-$v/task_5/pred_test.npz" 2>&1 | tee -a "$LOG" || failed "$v al_score test"
    done
    finish
    exit 0
fi

r3_build() {
    local v=$1 tag=${1##*_r3_} sel=${1#al_}
    sel=${sel%_r3_*}_r2
    tag=${tag%_s[0-9]*}
    [[ $tag == freeze ]] && { tag=${R3_FREEZE_DATA:?"_r3_freeze: đặt R3_FREEZE_DATA=rows|train|real|all|prompt"}; }
    [[ -f $REL/active/selections/$sel.json ]] || { echo "$v: thiếu tập chọn $sel (chạy vòng 2 trước)" >&2; return 1; }
    local opts
    case $tag in
        rows|train) opts=(--rows configs/relearn/rows_r3.json --repeat 3) ;;
        real|prompt) opts=(--rows configs/relearn/rows_r3.json --repeat 8 --v2-keep 0.5) ;;
        all) opts=(--rows configs/relearn/rows_all.json --repeat 8 --v2-keep 0.5) ;;
        *) echo "$v: tag vòng 3 lạ: $tag" >&2; return 1 ;;
    esac
    python -m tools.al_build_variant --selection "$sel" --base e1 "${opts[@]}" --no-face-v2 600 --name "$v" \
        2>&1 | tee -a "$LOG"
}

build() {
    local v=$1
    if [[ $v == *_r3_* ]]; then
        r3_build "$v"
    elif [[ $v == *_r4 ]]; then
        python -m tools.al_round2 --campaign "$CAMPAIGN" --from "${v%_r4}_r3_real" --budget "${R4_BUDGET:-500}" \
            2>&1 | tee -a "$LOG"
    elif [[ $v == *_r2 ]]; then
        python -m tools.al_round2 --campaign "$CAMPAIGN" --from "${v%_r2}" 2>&1 | tee -a "$LOG"
    else
        python -m tools.al_build_variant --selection "${v#al_}" 2>&1 | tee -a "$LOG"
    fi
}

for v in "${VARIANTS[@]}"; do
    if [[ ! -f $REL/variants/$v/train_task_5_fix.json ]]; then
        build "$v" || { failed "$v dựng dữ liệu"; continue; }
    fi
    if [[ $DRY == 1 ]]; then
        um --stage data --variant "$v" || failed "$v --stage data"
        continue
    fi
    echo "=== $(date -u +%FT%TZ) $v" | tee -a "$LOG"
    um --stage al --variant "$v" || { failed "$v train"; continue; }
    PRED=$CAMPAIGN/runs/EXP-FB-FT-$v/task_5/pred_val.npz
    if [[ -f $CAMPAIGN/stages/$v.json && -f $PRED && ! -f $REL/active/scored/$v.json ]]; then
        python -m tools.al_score --name "$v" --pred "$PRED" 2>&1 | tee -a "$LOG" || failed "$v al_score"
    fi
done
if [[ $DRY != 1 ]]; then
    python -m tools.al_report --campaign "$CAMPAIGN" 2>&1 | tail -3 | tee -a "$LOG" || failed "al_report"
fi
finish
