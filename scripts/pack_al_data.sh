#!/usr/bin/env bash
# Đóng gói dữ liệu MỚI cho máy GPU (docs/al/RUNBOOK.md), chạy trên Mac, rồi upload 2 file .tar lên Kaggle.
#
#   bash scripts/pack_al_data.sh            # mặc định: ../relearn_fix_v2 và ../training/full-syn-r1
#
# 1. relearn_fix_v2_al_<ngày>.tar  : phần active learning thêm vào relearn_fix_v2 (active/, variants/al_*,
#    images/real). Giải nén ở THƯ MỤC CHỨA relearn_fix_v2 trên máy GPU (đè lên bản cũ, chỉ thêm/ghi đè file).
# 2. seed_full_syn_r1_b2_<ngày>.tar: FSA + task 1 của FULL-SYN r1 cho mốc fine-tune thẳng (B2). Giải nén ở gốc repo
#    (-> data/training/full-syn-r1/). Không cần nếu seed_full_syn_r1.tar trên Kaggle đã có đủ các file này.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=$(cd "${RELEARN_DATA_ROOT:-$REPO/../relearn_fix_v2}" && pwd)
SEED=$(cd "${SEED_RUN:-$REPO/../training/full-syn-r1}" && pwd)
OUT=${OUT:-$(dirname "$DATA")/packages_al}
DAY=$(date +%Y%m%d)
mkdir -p "$OUT"

AL=$OUT/relearn_fix_v2_al_$DAY.tar
(cd "$(dirname "$DATA")" && tar -cf "$AL" "$(basename "$DATA")/active" \
    $(cd "$(dirname "$DATA")" && ls -d "$(basename "$DATA")"/variants/al_*) "$(basename "$DATA")/images/real")
echo "$AL: $(du -h "$AL" | cut -f1)"

B2=$OUT/seed_full_syn_r1_b2_$DAY.tar
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
T=$STAGE/data/training/full-syn-r1
mkdir -p "$T/runs/EXP-FULL-SYN/task_1" "$T/runs/EXP-FULL-SYN-FSA/task_1" "$T/stages"
ln -s "$SEED/plan.json" "$T/plan.json"
for f in "$SEED"/stages/*.json; do ln -s "$f" "$T/stages/"; done
for f in calibration_count.json calibration_count_m2.json calibration_count_groups.json metrics_cl_val.json; do
    ln -s "$SEED/runs/EXP-FULL-SYN/$f" "$T/runs/EXP-FULL-SYN/$f"
done
for f in task_final.pth pred_val.npz run_info.json; do
    ln -s "$SEED/runs/EXP-FULL-SYN/task_1/$f" "$T/runs/EXP-FULL-SYN/task_1/$f"
done
ln -s "$SEED/runs/EXP-FULL-SYN-FSA/task_1/hf_model" "$T/runs/EXP-FULL-SYN-FSA/task_1/hf_model"
for f in run_info.json pred_val.npz; do
    [[ -e $SEED/runs/EXP-FULL-SYN-FSA/task_1/$f ]] && ln -s "$SEED/runs/EXP-FULL-SYN-FSA/task_1/$f" \
        "$T/runs/EXP-FULL-SYN-FSA/task_1/$f"
done
(cd "$STAGE" && tar -chf "$B2" data)
echo "$B2: $(du -h "$B2" | cut -f1)"

(cd "$OUT" && md5sum "$(basename "$AL")" "$(basename "$B2")" > "MD5SUMS_$DAY")
cat "$OUT/MD5SUMS_$DAY"
echo "Upload 2 file .tar + MD5SUMS_$DAY lên dataset Kaggle (Private), xem docs/al/RUNBOOK.md."
