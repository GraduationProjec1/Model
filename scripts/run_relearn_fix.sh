#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=$(cd "${AC_DATA:-${RELEARN_DATA_ROOT:-$REPO/../autocheckout-data}}" && pwd)
SETTINGS=${SETTINGS:-$REPO/configs/training/update_model_relearn_fix_v2.json}
STAGE=${1:-pilot}
shift || true

if [[ -d $DATA/relearn ]]; then
    NEED=(eval/val_full.json eval/test_full.json relearn/variants/pilot/train_task_5_fix.json relearn/SHA256SUMS
          models/full-syn-r1/runs/EXP-FULL-SYN/task_5/task_final.pth models/full-syn-r1/runs/EXP-FULL-SYN-FSA/task_1/hf_model)
    SUMS=$DATA/relearn
else
    NEED=(eval/val_full.json eval/test_full.json variants/pilot/train_task_5_fix.json
          seed/full-syn-r1/runs/EXP-FULL-SYN/task_5/task_final.pth seed/full-syn-r1/runs/EXP-FULL-SYN-FSA/task_1/hf_model)
    SUMS=$DATA
fi
for f in "${NEED[@]}"; do
    [[ -e $DATA/$f ]] || { echo "thiếu $DATA/$f: thư mục dữ liệu chưa đủ" >&2; exit 1; }
done
if [[ ${CHECK:-1} == 1 && ! -f $SUMS/.sha256_ok ]]; then
    echo "kiểm SHA256SUMS của $SUMS (một lần, vài phút)..."
    (cd "$SUMS" && sha256sum -c --quiet SHA256SUMS) && touch "$SUMS/.sha256_ok"
fi

cd "$REPO"
export AC_DATA=$DATA
if [[ $STAGE == data ]]; then
    ARGS=(--stage data --variant "${VARIANT:-pilot}")
else
    ARGS=(--stage "$STAGE")
fi
mkdir -p data/training/update-model
python -m autocheckout.update_model --mode improve --settings "$SETTINGS" "${ARGS[@]}" "$@" \
    2>&1 | tee -a "data/training/update-model/relearn_fix_${STAGE}.log"
