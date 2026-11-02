#!/usr/bin/env bash
# Học lại (mode improve) với thư mục dữ liệu tự đủ (build_relearn_data.py --self-contained).
# Mặc định relearn_fix_v2 (ảnh đủ mặt + ảnh thật + v2 + replay). Bản v1 (không ảnh đủ mặt):
#   RELEARN_DATA_ROOT=../relearn_fix_v1 SETTINGS=configs/training/update_model_relearn_fix_v1.json bash scripts/run_relearn_fix.sh
#
#   bash scripts/run_relearn_fix.sh                 # mặc định: split -> baseline -> pilot
#   bash scripts/run_relearn_fix.sh data            # chỉ dựng dữ liệu pilot, in số (không train)
#   bash scripts/run_relearn_fix.sh e1              # sau khi pilot qua cổng (go)
#   bash scripts/run_relearn_fix.sh final_test --confirm-test   # chấm test2019 một lần, khi nhóm đã chốt
#
# Thư mục dữ liệu mặc định ở cạnh repo (../relearn_fix_v2); đặt nơi khác thì RELEARN_DATA_ROOT=/duong/dan.
# Bị ngắt thì chạy lại đúng lệnh: bước đã xong được bỏ qua, task dở chạy tiếp từ last.ckpt.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=$(cd "${RELEARN_DATA_ROOT:-$REPO/../relearn_fix_v2}" && pwd)
SETTINGS=${SETTINGS:-$REPO/configs/training/update_model_relearn_fix_v2.json}
STAGE=${1:-pilot}
shift || true

for f in eval/val_full.json eval/test_full.json variants/pilot/train_task_5_fix.json \
         seed/full-syn-r1/runs/EXP-FULL-SYN/task_5/task_final.pth seed/full-syn-r1/runs/EXP-FULL-SYN-FSA/task_1/hf_model; do
    [[ -e $DATA/$f ]] || { echo "thiếu $DATA/$f: thư mục dữ liệu chưa đủ (build_relearn_data.py --self-contained)" >&2; exit 1; }
done
if [[ ${CHECK:-1} == 1 && ! -f $DATA/.sha256_ok ]]; then
    echo "kiểm SHA256SUMS của $DATA (một lần, vài phút)..."
    (cd "$DATA" && sha256sum -c --quiet SHA256SUMS) && touch "$DATA/.sha256_ok"
fi

cd "$REPO"
export RELEARN_DATA_ROOT=$DATA
if [[ $STAGE == data ]]; then
    ARGS=(--stage data --variant "${VARIANT:-pilot}")
else
    ARGS=(--stage "$STAGE")
fi
mkdir -p data/training/update-model
python -m autocheckout.update_model --mode improve --settings "$SETTINGS" "${ARGS[@]}" "$@" \
    2>&1 | tee -a "data/training/update-model/relearn_fix_${STAGE}.log"
