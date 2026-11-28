#!/usr/bin/env bash
# Script trọn gói chạy lại các thí nghiệm đã sửa (N2 Logit KD đã fix bug teacher + robust-calib,
# E1 baseline cho nhánh prompt) cùng M6.2 RoI-Refine trên 1 máy Vast.ai.
#
# Chuỗi mặc định (~6–7 giờ trên 1x RTX 4090, ~$2,5–3,0):
#   1. n0_dump : ~5 phút  — xuất logits_val.npz của r4 (kiểm khớp 68,28%) làm đầu vào cho m6_roi
#   2. m6_roi  : ~15 phút — M6.2 RoI-Refine (ConvNeXt-Tiny 224px trên crop thật Val-A + AMP)
#   3. n2_e    : ~3,5 giờ — N2 Logit KD (đã sửa bug --pseudo none không tạo teacher + fail-fast)
#   4. attn_e1 : ~2,2 giờ — E1 (chỉ A = per_query_prompt, tách baseline so với attn_m1)
#
# Cách dùng trên máy Vast (trong tmux):
#   cd /workspace/AutoCheckout-CL
#   source /workspace/venv-pdp/bin/activate
#   export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9
#   export RCLONE_REMOTE="gdrive:AutoCheckout/ket_qua_gpu"
#   export VAST_API_KEY="<key>"        # tuỳ chọn: tự huỷ máy SAU KHI TẤT CẢ TASK ĐỀU QUA
#   export AUTO_DESTROY=1              # mặc định 0; đặt 1 để tự huỷ sau khi hoàn tất
#   bash scripts/vast_rerun_n2_roi.sh
#
# Tuỳ chọn:
#   --setup-only                       # chỉ tải dữ liệu, kiểm md5 và dry-run, chưa chạy GPU
#   --run-only                         # bỏ qua bước tải (khi máy đã có sẵn dữ liệu)
#   TASKS="n0_dump m6_roi n2_e attn_e1"  # có thể thêm n3_e nếu muốn chạy lại N3
#   BRANCH=feat/n1-n5                  # nhánh git chứa bản sửa N2 + M6.2 RoI-Refine
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
export TASKS="${TASKS:-n0_dump m6_roi n2_e attn_e1}"
export BRANCH="${BRANCH:-feat/n1-n5}"

exec bash "$REPO/scripts/vast_setup_and_run_m_n.sh" "$@"
