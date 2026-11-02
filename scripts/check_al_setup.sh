#!/usr/bin/env bash
# Kiểm tra máy GPU trước khi chạy active learning / B2 (docs/al/RUNBOOK.md). Không train, vài phút.
#   bash scripts/check_al_setup.sh
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=${RELEARN_DATA_ROOT:-$REPO/../relearn_fix_v2}
cd "$REPO"
ok=1
need() { if [[ -e $1 ]]; then echo "  có  $1"; else echo "  THIẾU $1  ($2)"; ok=0; fi; }

echo "== dữ liệu active learning ($DATA)"
need "$DATA/eval/val_full.json" "relearn_fix_v2.zip"
need "$DATA/seed/full-syn-r1/runs/EXP-FULL-SYN/task_5/task_final.pth" "relearn_fix_v2.zip"
need "$DATA/active/splits/val_a_split.json" "relearn_fix_v2_al_*.tar"
need "$DATA/variants/al_margin_k_B500/train_task_5_fix.json" "relearn_fix_v2_al_*.tar"
echo "== dữ liệu B2 (fine-tune thẳng)"
need data/processed/rpc_synth_v2_v1/images/task_5 "full_syn_rpc_synth_v2_v1_*.tar"
need data/processed/rpc_eval_official_v1/val_full.json "full_syn_rpc_synth_v2_v1_*.tar"
need data/archive/val2019 "full_syn_rpc_synth_v2_v1_*.tar"
need data/pretrained/hf-cache "full_syn_rpc_synth_v2_v1_*.tar"
need data/training/full-syn-r1/runs/EXP-FULL-SYN/task_1/task_final.pth "seed_full_syn_r1_b2_*.tar"
need data/training/full-syn-r1/runs/EXP-FULL-SYN-FSA/task_1/hf_model "seed_full_syn_r1_b2_*.tar"
need data/training/full-syn-r1/runs/EXP-FULL-SYN/metrics_cl_val.json "seed_full_syn_r1_b2_*.tar"

echo "== md5 model r1 task 5 (cần fe510a4c4dc2c81a328299c4009d31ff)"
md5sum "$DATA/seed/full-syn-r1/runs/EXP-FULL-SYN/task_5/task_final.pth" 2>/dev/null || ok=0
echo "== GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || ok=0
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || ok=0

echo "== test (có torch: gồm cả dry run train của update_model và B2, vài phút)"
python -m pytest -q -p no:cacheprovider tests/test_active_learning.py tests/test_al_build_variant.py \
    tests/test_forgetting_report.py tests/test_update_model.py tests/test_native_exp_configs.py \
    tests/test_full_syn.py -k "not m5" || ok=0

echo "== dựng dữ liệu 1 gói (CPU)"
RELEARN_DATA_ROOT=$DATA bash scripts/run_al.sh al_margin_k_B500 --dry-run | tail -1 || ok=0

[[ $ok == 1 ]] && echo "SẴN SÀNG" || { echo "CHƯA SẴN SÀNG: xem các dòng THIẾU / lỗi ở trên"; exit 1; }
