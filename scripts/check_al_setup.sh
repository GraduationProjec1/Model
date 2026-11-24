#!/usr/bin/env bash
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=${AC_DATA:-$REPO/../autocheckout-data}
cd "$REPO"
ok=1
need() { if [[ -e $DATA/$1 ]]; then echo "  có    $1"; else echo "  THIẾU $1"; ok=0; fi; }

echo "== thư mục dữ liệu: $DATA"
need eval/val_full.json
need rpc/val2019
need rpc/test2019
need pretrained/hf-cache
need models/full-syn-r1/runs/EXP-FULL-SYN/task_5/task_final.pth
need models/full-syn-r1/runs/EXP-FULL-SYN/task_1/task_final.pth
need models/full-syn-r1/runs/EXP-FULL-SYN-FSA/task_1/hf_model
need relearn/SHA256SUMS
need relearn/active/splits/val_a_split.json
need relearn/variants/al_margin_k_B500/train_task_5_fix.json
need synth_v2/images/task_5
echo "== md5 model r1 task 5 (cần fe510a4c4dc2c81a328299c4009d31ff)"
md5sum "$DATA/models/full-syn-r1/runs/EXP-FULL-SYN/task_5/task_final.pth" 2>/dev/null || ok=0
if [[ ${FULL:-0} == 1 ]]; then
    echo "== SHA256 toàn bộ dữ liệu"
    (cd "$DATA" && sha256sum -c --quiet SHA256SUMS && echo "  dữ liệu nguyên vẹn") || ok=0
fi
echo "== GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || ok=0
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || ok=0

echo "== test (có torch: gồm cả dry run train của update_model và B2, vài phút)"
python -m pytest -q -p no:cacheprovider tests/test_active_learning.py tests/test_al_build_variant.py \
    tests/test_forgetting_report.py tests/test_update_model.py tests/test_native_exp_configs.py \
    tests/test_full_syn.py -k "not m5" || ok=0

echo "== dựng dữ liệu 1 gói (CPU)"
AC_DATA=$DATA bash scripts/run_al.sh al_margin_k_B500 --dry-run | tail -1 || ok=0

[[ $ok == 1 ]] && echo "SẴN SÀNG" || { echo "CHƯA SẴN SÀNG: xem các dòng THIẾU / lỗi ở trên"; exit 1; }
