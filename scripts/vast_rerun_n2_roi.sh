#!/usr/bin/env bash
# Script trọn gói chạy MỘT LƯỢT trên Vast.ai toàn bộ gói thí nghiệm sửa lỗi + nâng cấp (không phải tải đi tải lại dữ liệu):
#
# Chuỗi mặc định (7 task, ~16–18 giờ trên 1x RTX 4090, ~$7, tự sync Drive sau MỖI task):
#   1. n0_dump   : ~5 phút  — xuất logits_val.npz của r4 (kiểm khớp 68,28%) làm đầu vào cho m6_roi
#   2. m6_roi    : ~15 phút — M6.2 RoI-Refine (ConvNeXt-Tiny 224px trên crop thật Val-A + AMP, 0% quên)
#   3. n2_e      : ~3,5 giờ — N2 Logit KD (đã sửa bug --pseudo none không tạo teacher + fail-fast + robust-calib)
#   4. attn_e1   : ~2,2 giờ — E1 (chỉ A = per_query_prompt, tách baseline cho M1)
#   5. r4_all_g3 : ~2,2 giờ — Mở toàn bộ 200 hàng class_embed + N3 (w=3) + robust-calib
#   6. r4_res800 : ~4,5 giờ — Fine-tune 800x800 px từ checkpoint 640 px (--allow_resolution_change 1, batch=4)
#   7. r4_lora   : ~3,5 giờ — Decoder LoRA rank-8 (khởi tạo 0) + N2 Logit KD (giữ lớp cũ)
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
#   TASKS="n0_dump m6_roi n2_e attn_e1 r4_all_g3 r4_res800 r4_lora"
#   BRANCH=feat/n1-n5                  # nhánh git chứa toàn bộ bản sửa + nâng cấp
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
export TASKS="${TASKS:-n0_dump m6_roi n2_e attn_e1 r4_all_g3 r4_res800 r4_lora}"
export BRANCH="${BRANCH:-feat/n1-n5}"

exec bash "$REPO/scripts/vast_setup_and_run_m_n.sh" "$@"
