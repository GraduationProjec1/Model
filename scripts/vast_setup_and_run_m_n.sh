#!/usr/bin/env bash
# Tải đủ dữ liệu (32 GB tar + gói r3 + r3_real + r4) và chạy một lượt các task M1/M6 và N0–N3 trên Vast.ai.
#
# Cách dùng trên máy Vast (trong tmux):
#   cd /workspace/AutoCheckout-CL
#   source /workspace/venv-pdp/bin/activate
#   export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9
#   export RCLONE_REMOTE="gdrive:AutoCheckout/ket_qua_gpu"
#   export VAST_API_KEY="<key>"        # tuỳ chọn: chỉ dùng để tự huỷ máy SAU KHI CHẠY XONG CẢ CHUỖI
#   export AUTO_DESTROY=1              # mặc định 0; đặt 1 nếu muốn tự huỷ sau khi tất cả task đều qua
#   bash scripts/vast_setup_and_run_m_n.sh
#
# Tuỳ chọn:
#   --setup-only                       # chỉ tải dữ liệu, kiểm md5 và dry-run, chưa chạy GPU
#   --run-only                         # bỏ qua bước tải (dùng khi đã cài xong dữ liệu trước đó)
#   TASKS="n0_dump m6_res attn_m1 n3_e n2_e"   # có thể thêm attn_e1..attn_e5 nếu muốn
#   BRANCH=feat/n1-n5                  # nhánh git (mặc định feat/n1-n5, đã gộp sẵn feat/m1-m6)
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA="${AC_DATA:-/workspace/autocheckout-data}"
BRANCH="${BRANCH:-feat/n1-n5}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive:AutoCheckout/ket_qua_gpu}"
R4_SRC="${R4_SRC:-gdrive:AutoCheckout/ket_qua_gpu/al_r4_20261114_165406}"
TASKS="${TASKS:-n0_dump m6_res attn_m1 n3_e n2_e}"

AUTO_DESTROY="${AUTO_DESTROY:-0}"
AUTO_STOP="${AUTO_STOP:-0}"
SYNC_EVERY_MIN="${SYNC_EVERY_MIN:-0}"
VAST_API_KEY="${VAST_API_KEY:-}"
INSTANCE_ID="${INSTANCE_ID:-}"

SETUP_ONLY=0
RUN_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --setup-only) SETUP_ONLY=1 ;;
        --run-only)   RUN_ONLY=1 ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *) echo "[LỖI] Không hiểu tham số: $arg" >&2; exit 2 ;;
    esac
done

export AC_DATA="$DATA"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

CAMP="$REPO/data/training/update-model/relearn-fix-v2"
R4_RUN="$CAMP/runs/EXP-FB-FT-al_margin_k_B500_r4/task_5"
SCORED="$DATA/relearn/active/scored"

TAR_MD5="5a1efb48209d998012b01dd1dfa9674c"
SEED_MD5="fe510a4c4dc2c81a328299c4009d31ff"
R4_SCORED_MD5="034e1702f014ea90342900dc94920a5b"

fail() { echo "[LỖI] $*" >&2; exit 1; }

if [[ "$RUN_ONLY" != "1" ]]; then
    echo "=========================================================="
    echo " BƯỚC 1/5: KIỂM TRA MÔI TRƯỜNG"
    echo "=========================================================="
    command -v rclone >/dev/null || fail "Chưa cài rclone"
    rclone lsd gdrive:AutoCheckout >/dev/null || fail "rclone chưa đọc được gdrive:AutoCheckout"
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('PyTorch:', torch.__version__, torch.cuda.get_device_name(0))" \
        || fail "Môi trường Python/CUDA chưa sẵn sàng (nhớ source /workspace/venv-pdp/bin/activate)"

    echo "=========================================================="
    echo " BƯỚC 2/5: DỮ LIỆU GỐC 32 GB (autocheckout-data.tar)"
    echo "=========================================================="
    if [[ -f "$DATA/eval/val_full.json" && -d "$DATA/rpc/val2019" ]]; then
        echo "   Đã có $DATA -> bỏ qua tải tar."
    else
        mkdir -p "$(dirname "$DATA")"
        cd "$(dirname "$DATA")"
        rclone copy gdrive:AutoCheckout/autocheckout-data.tar . --progress \
            --multi-thread-streams 16 --multi-thread-cutoff 50M --drive-chunk-size 128M --buffer-size 64M \
            || fail "Tải autocheckout-data.tar thất bại"
        [[ "$(md5sum autocheckout-data.tar | cut -d' ' -f1)" == "$TAR_MD5" ]] \
            || fail "md5 autocheckout-data.tar sai (kỳ vọng $TAR_MD5)"
        tar --warning=no-unknown-keyword -xf autocheckout-data.tar -C "$(dirname "$DATA")" \
            || fail "Giải nén autocheckout-data.tar thất bại"
        rm -f autocheckout-data.tar
        (cd "$DATA" && sha256sum -c --quiet --ignore-missing SHA256SUMS) || fail "SHA256SUMS không khớp"
        [[ "$(md5sum "$DATA/models/full-syn-r1/runs/EXP-FULL-SYN/task_5/task_final.pth" | cut -d' ' -f1)" == "$SEED_MD5" ]] \
            || fail "md5 seed task_final.pth sai"
    fi

    echo "=========================================================="
    echo " BƯỚC 3/5: GÓI VÒNG 3 (gpu_bundle_r3.zip)"
    echo "=========================================================="
    if [[ -f "$CAMP/plan.json" && -f "$CAMP/stages/al_margin_k_B500_r2.json" \
          && -f "$DATA/relearn/variants/al_margin_k_B500_r3_real/train_task_5_fix.json" ]]; then
        echo "   Đã cài gói r3 -> bỏ qua."
    else
        WORK_PARENT="$(dirname "$REPO")"
        cd "$WORK_PARENT"
        rclone copy gdrive:AutoCheckout/gpu_bundle_r3.zip . --progress --checksum \
            || fail "Tải gpu_bundle_r3.zip thất bại"
        command -v unzip >/dev/null || { apt-get update -qq && apt-get install -y -qq unzip >/dev/null; }
        rm -rf "$WORK_PARENT/_zip" && mkdir -p "$WORK_PARENT/_zip"
        unzip -q gpu_bundle_r3.zip -d "$WORK_PARENT/_zip" || fail "Giải nén gpu_bundle_r3.zip thất bại"
        B="$(dirname "$(find "$WORK_PARENT/_zip" -name install.sh -not -path '*__MACOSX*' | head -1)")"
        [[ -n "$B" && -f "$B/install.sh" ]] || fail "Không thấy install.sh trong gpu_bundle_r3.zip"
        rm -rf "$WORK_PARENT/gpu_bundle_r3" && mv "$B" "$WORK_PARENT/gpu_bundle_r3"
        rm -rf "$WORK_PARENT/_zip" "$WORK_PARENT/gpu_bundle_r3.zip"
        bash "$WORK_PARENT/gpu_bundle_r3/install.sh" || fail "gpu_bundle_r3/install.sh lỗi"
    fi

    echo "=========================================================="
    echo " BƯỚC 4/5: GÓI VÒNG 4 + CHECKOUT NHÁNH $BRANCH"
    echo "=========================================================="
    cd "$REPO"
    # install_r4.sh tự fetch + checkout + reset --hard origin/$BRANCH và tải r3_real, weights_md5, selection r3
    BRANCH="$BRANCH" REPO="$REPO" AC_DATA="$DATA" bash "$REPO/scripts/install_r4.sh" \
        || fail "scripts/install_r4.sh lỗi"

    echo "=========================================================="
    echo " BƯỚC 5/5: TẢI CHECKPOINT & ĐIỂM r4 (CHO n0_dump VÀ m6_res)"
    echo "=========================================================="
    mkdir -p "$R4_RUN" "$SCORED"
    rclone copy "$R4_SRC/update-model/relearn-fix-v2/runs/EXP-FB-FT-al_margin_k_B500_r4/task_5" "$R4_RUN" \
        --include "task_final.pth" --include "run_info.json" --include "pred_val.npz" --progress \
        || fail "Tải r4 task_5 từ $R4_SRC thất bại"
    rclone copy "$R4_SRC/relearn_active/scored" "$SCORED" \
        --include "al_margin_k_B500_r4.json" --progress \
        || fail "Tải điểm al_margin_k_B500_r4.json thất bại"

    for f in "$R4_RUN/task_final.pth" "$R4_RUN/run_info.json" "$SCORED/al_margin_k_B500_r4.json" \
             "$DATA/eval/val_full.json" "$DATA/eval/val_task_5.json"; do
        [[ -f "$f" ]] || fail "Thiếu file bắt buộc sau khi tải: $f"
    done
    if [[ "$(md5sum "$SCORED/al_margin_k_B500_r4.json" | cut -d' ' -f1)" != "$R4_SCORED_MD5" ]]; then
        echo "[CẢNH BÁO] md5 al_margin_k_B500_r4.json khác bản tham chiếu ($R4_SCORED_MD5); vẫn tiếp tục."
    fi

    echo ">> Dry-run dựng dữ liệu _r4 và kiểm đầu vào..."
    cd "$REPO"
    bash scripts/run_al.sh al_margin_k_B500_r4_pqm --dry-run || fail "dry-run thất bại"
    echo ">> CÀI ĐẶT VÀ KIỂM TRA ĐẦU VÀO HOÀN TẤT!"
fi

if [[ "$SETUP_ONLY" == "1" ]]; then
    echo "Đã dừng theo --setup-only. Khi sẵn sàng, chạy: bash scripts/vast_setup_and_run_m_n.sh --run-only"
    exit 0
fi

echo "=========================================================="
echo " CHẠY CHUỖI TASK TRÊN NHÁNH $(git -C "$REPO" branch --show-current)"
echo " Danh sách: $TASKS"
echo " RCLONE_REMOTE: $RCLONE_REMOTE"
echo " AUTO_DESTROY sau khi xong cả chuỗi: $AUTO_DESTROY"
echo "=========================================================="

cd "$REPO"
PASSED_TASKS=()
FAILED_TASKS=()

for t in $TASKS; do
    echo ""
    echo "##########################################################"
    echo " [$(date '+%F %T')] BẮT ĐẦU TASK: $t"
    echo "##########################################################"
    # Tắt AUTO_DESTROY/AUTO_STOP và ẩn VAST_API_KEY cho từng task con để máy không tắt giữa chuỗi;
    # SYNC_EVERY_MIN=0 (mặc định): chỉ đẩy kết quả lên Drive sau khi xong từng thí nghiệm
    if AUTO_DESTROY=0 AUTO_STOP=0 VAST_API_KEY="" SYNC_EVERY_MIN="$SYNC_EVERY_MIN" RCLONE_REMOTE="$RCLONE_REMOTE" \
       bash "$REPO/scripts/vast_train_and_sync.sh" "$t"; then
        PASSED_TASKS+=("$t")
    else
        echo "[CẢNH BÁO] Task $t kết thúc với lỗi; vẫn chạy tiếp task sau."
        FAILED_TASKS+=("$t")
    fi
done

echo ""
echo "=========================================================="
echo " TỔNG KẾT CHUỖI TASK ($(date '+%F %T'))"
echo "   Qua (${#PASSED_TASKS[@]}): ${PASSED_TASKS[*]:-không có}"
echo "   Lỗi (${#FAILED_TASKS[@]}): ${FAILED_TASKS[*]:-không có}"
echo "=========================================================="

if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
    echo "[LƯU Ý] Có task lỗi -> KHÔNG tự destroy; chuyển sang STOP (nếu có VAST_API_KEY)."
    AUTO_DESTROY=0
    [[ -n "$VAST_API_KEY" ]] && AUTO_STOP=1
fi

if [[ "$AUTO_DESTROY" == "1" || "$AUTO_STOP" == "1" ]]; then
    if [[ -n "$VAST_API_KEY" ]]; then
        pip install --quiet --upgrade vastai || true
        vastai set api-key "$VAST_API_KEY" || true
        if [[ -z "$INSTANCE_ID" ]]; then
            INSTANCE_ID="$(vastai show instances --raw 2>/dev/null | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, list) and data:
        print(data[0]['id'])
except Exception:
    pass
" || echo "")"
        fi
        if [[ -n "$INSTANCE_ID" ]]; then
            if [[ "$AUTO_DESTROY" == "1" ]]; then
                echo ">> ĐANG DESTROY INSTANCE $INSTANCE_ID..."
                vastai destroy instance "$INSTANCE_ID"
            elif [[ "$AUTO_STOP" == "1" ]]; then
                echo ">> ĐANG STOP INSTANCE $INSTANCE_ID..."
                vastai stop instance "$INSTANCE_ID"
            fi
        else
            echo "[CẢNH BÁO] Không tìm thấy INSTANCE_ID; vui lòng stop/destroy thủ công trên Vast.ai!"
        fi
    else
        echo "[LƯU Ý] Chưa đặt VAST_API_KEY; vui lòng destroy máy thủ công trên Vast.ai!"
    fi
fi

[[ ${#FAILED_TASKS[@]} -eq 0 ]]
