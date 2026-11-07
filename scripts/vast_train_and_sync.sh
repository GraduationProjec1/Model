#!/usr/bin/env bash
# ==============================================================================
# Script tự động hoá trên Vast.ai:
# 1. Ghi lại cấu hình hệ thống, phần cứng, GPU và Git provenance
# 2. Chạy huấn luyện (Active Learning hoặc Relearn Fix)
# 3. Ghi lại toàn bộ log, thông số train, metrics
# 4. Tự động đồng bộ kết quả (checkpoint, logs, stages, report) lên Google Drive bằng rclone
# 5. Tự động Huỷ (Destroy) hoặc Dừng (Stop) instance Vast.ai để tiết kiệm chi phí
# ==============================================================================
set -uo pipefail

# ----------------- CẤU HÌNH MẶC ĐỊNH (CÓ THỂ TRUYỀN TỪ MÔI TRƯỜNG) -----------------
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

DATA_DIR="${AC_DATA:-$(cd "$REPO_DIR/../autocheckout-data" 2>/dev/null && pwd || echo "/workspace/autocheckout-data")}"
export AC_DATA="$DATA_DIR"

# Remote rclone cho Google Drive (tên remote và thư mục trên Drive)
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive:AutoCheckout_Runs}"

# Chế độ tự huỷ / tắt máy sau khi đẩy kết quả lên Drive (1 = bật, 0 = tắt)
AUTO_DESTROY="${AUTO_DESTROY:-1}"
AUTO_STOP="${AUTO_STOP:-0}"

# API key của Vast.ai để ra lệnh destroy/stop từ bên trong máy
VAST_API_KEY="${VAST_API_KEY:-}"
INSTANCE_ID="${INSTANCE_ID:-}"

# Task muốn chạy:
#   - "al_dot1": Active Learning đợt 1 (7 run B=500: random x3, margin, margin_k, margin_bal, oracle)
#   - "al_dot2": Active Learning đợt 2 (B=250, 850 + entropy)
#   - "al_next": vòng 2 từ margin_k (+ random_s0 so sánh) rồi đợt 2; cần khôi phục campaign dot1 từ Drive trước
#   - "relearn_pilot": Chạy học lại pilot (17 SKU)
#   - "relearn_e1": Chạy học lại E1 (63 SKU)
#   - "naive": Chạy fine-tune thẳng (baseline B2)
TASK="${1:-${TASK:-al_dot1}}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_ID="${TASK}_${TIMESTAMP}"
LOG_DIR="$REPO_DIR/data/training/update-model/vast_logs_${RUN_ID}"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/master_execution.log"

echo "=========================================================="
echo " KHỞI ĐỘNG AUTO-RUN VAST.AI: $RUN_ID"
echo " Task: $TASK"
echo " Repo: $REPO_DIR"
echo " Data: $DATA_DIR"
echo " Google Drive Remote: $RCLONE_REMOTE"
echo " Tự động Destroy: $AUTO_DESTROY (Instance ID: ${INSTANCE_ID:-Chưa xác định})"
echo "=========================================================="

# ----------------- HÀM DỌN DẸP, SYNC DRIVE VÀ DESTROY INSTANCE -----------------
finish_and_cleanup() {
    local EXIT_CODE=$?
    if [[ -n "${SYNC_PID:-}" ]]; then pkill -P "$SYNC_PID" 2>/dev/null || true; kill "$SYNC_PID" 2>/dev/null || true; fi
    echo ""
    echo "=========================================================="
    echo " QUÁ TRÌNH HUẤN LUYỆN KẾT THÚC VỚI MÃ: $EXIT_CODE"
    echo " BẮT ĐẦU ĐỒNG BỘ KẾT QUẢ VỀ GOOGLE DRIVE..."
    echo "=========================================================="

    # 1. Lưu lại metadata kết thúc
    echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_DIR/run_metadata.txt"
    echo "Exit Code: $EXIT_CODE" >> "$LOG_DIR/run_metadata.txt"

    # 1b. Bản dự phòng báo cáo AL (đi cùng update-model) + md5 mọi weights để kiểm tra sau khi tải về
    if [[ -d "$DATA_DIR/relearn/active" ]]; then
        mkdir -p "$REPO_DIR/data/training/update-model/relearn_active_backup"
        cp -r "$DATA_DIR/relearn/active/." "$REPO_DIR/data/training/update-model/relearn_active_backup/" || true
    fi
    (cd "$REPO_DIR/data/training/update-model" 2>/dev/null && find . -name "*.pth" -exec md5sum {} \; | sort -k2) \
        > "$LOG_DIR/weights_md5.txt" 2>/dev/null || true

    # 2. Đồng bộ kết quả qua rclone
    if command -v rclone &>/dev/null; then
        echo "[RCLONE] Đang upload log và stages..."
        rclone copy "$LOG_DIR" "$RCLONE_REMOTE/$RUN_ID/logs" --verbose || true

        echo "[RCLONE] Đang upload data/training/update-model..."
        if [[ -d "$REPO_DIR/data/training/update-model" ]]; then
            # Sync toàn bộ logs, configs, checkpoints, npz và json
            rclone copy "$REPO_DIR/data/training/update-model" "$RCLONE_REMOTE/$RUN_ID/update-model" \
                --exclude "vast_logs_*" --verbose || true
        fi

        # Chốt an toàn: đếm .pth trên máy và trên Drive; thiếu thì KHÔNG destroy (chỉ stop) để còn lấy lại.
        local N_LOCAL N_REMOTE
        N_LOCAL=$(find "$REPO_DIR/data/training/update-model" -type f -name "*.pth" 2>/dev/null | wc -l)  # not the symlinks in scored/
        N_REMOTE=$(rclone ls "$RCLONE_REMOTE/$RUN_ID/update-model" --include "*.pth" 2>/dev/null | wc -l)
        echo "[KIỂM TRA] Weights .pth: trên máy $N_LOCAL, trên Drive $N_REMOTE" | tee -a "$LOG_DIR/run_metadata.txt"
        if [[ "$N_REMOTE" -lt "$N_LOCAL" ]]; then
            echo "[CẢNH BÁO] Drive thiếu weights -> KHÔNG destroy, chỉ STOP máy. Vào lại máy để upload tay." | tee -a "$LOG_DIR/run_metadata.txt"
            rclone copy "$LOG_DIR" "$RCLONE_REMOTE/$RUN_ID/logs" || true
            AUTO_DESTROY=0; AUTO_STOP=1
        fi

        echo "[RCLONE] Đang upload báo cáo active learning..."
        if [[ -d "$DATA_DIR/relearn/active" ]]; then
            rclone copy "$DATA_DIR/relearn/active" "$RCLONE_REMOTE/$RUN_ID/relearn_active" --verbose || true
        fi
        echo "[RCLONE] Đã hoàn thành đồng bộ Google Drive thành công!"
    else
        echo "[CẢNH BÁO] Rclone chưa được cài hoặc chưa cấu hình! Bỏ qua bước sync."
    fi

    # 3. Tự huỷ hoặc tắt instance Vast.ai
    if [[ "$AUTO_DESTROY" == "1" || "$AUTO_STOP" == "1" ]]; then
        if [[ -n "$VAST_API_KEY" ]]; then
            pip install --quiet --upgrade vastai || true
            vastai set api-key "$VAST_API_KEY" || true

            # Tự dò Instance ID nếu chưa có
            if [[ -z "$INSTANCE_ID" ]]; then
                INSTANCE_ID=$(vastai show instances --raw 2>/dev/null | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, list) and len(data) > 0:
        print(data[0]['id'])
except Exception:
    pass
" || echo "")
            fi

            if [[ -n "$INSTANCE_ID" ]]; then
                if [[ "$AUTO_DESTROY" == "1" ]]; then
                    echo ">> ĐANG RA LỆNH XOÁ (DESTROY) INSTANCE $INSTANCE_ID ĐỂ TRÁNH TỐN PHÍ..."
                    vastai destroy instance "$INSTANCE_ID"
                elif [[ "$AUTO_STOP" == "1" ]]; then
                    echo ">> ĐANG RA LỆNH DỪNG (STOP) INSTANCE $INSTANCE_ID..."
                    vastai stop instance "$INSTANCE_ID"
                fi
            else
                echo "[CẢNH BÁO] Không tìm thấy Instance ID, vui lòng stop/destroy thủ công trên web vast.ai!"
            fi
        else
            echo "[LƯU Ý] Không có VAST_API_KEY, máy sẽ không tự huỷ. Vui lòng destroy trên web Vast.ai!"
        fi
    fi
}

trap finish_and_cleanup EXIT

# ----------------- BƯỚC 1: GHI LẠI TOÀN BỘ CẤU HÌNH HỆ THỐNG VÀ THÔNG SỐ TRAIN -----------------
{
    echo "=== RUN METADATA & SYSTEM AUDIT ==="
    echo "Run ID: $RUN_ID"
    echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Hostname: $(hostname)"
    echo "OS Kernel: $(uname -a)"
    echo "Git Branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
    echo "Git Commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    echo ""
    echo "=== GPU INFORMATION ==="
    nvidia-smi || true
    echo ""
    echo "=== CPU & RAM ==="
    lscpu 2>/dev/null | head -n 20 || true
    free -h || true
    echo ""
    echo "=== PYTHON & CUDA ENVIRONMENT ==="
    python --version || true
    python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" || true
    nvcc --version 2>/dev/null || true
    echo ""
    echo "=== PIP FREEZE ==="
    pip list || true
} > "$LOG_DIR/run_metadata.txt"

# ----------------- BƯỚC 1b: ẢNH CHỤP CẤU HÌNH + ĐỒNG BỘ ĐỊNH KỲ -----------------
mkdir -p "$LOG_DIR/config_snapshot"
cp -r configs/training configs/relearn "$LOG_DIR/config_snapshot/" 2>/dev/null || true
git status --porcelain > "$LOG_DIR/config_snapshot/git_status.txt" 2>&1 || true
git diff > "$LOG_DIR/config_snapshot/git_diff.patch" 2>&1 || true
{
    echo "TASK=$TASK"; echo "AC_DATA=$AC_DATA"; echo "CUDA_HOME=${CUDA_HOME:-}"
    echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-}"; echo "RCLONE_REMOTE=$RCLONE_REMOTE"
    echo "AUTO_DESTROY=$AUTO_DESTROY AUTO_STOP=$AUTO_STOP"
    [[ -f "$DATA_DIR/README.md" ]] && echo "DATA_README_MD5=$(md5sum "$DATA_DIR/README.md" | cut -d' ' -f1)"
} > "$LOG_DIR/config_snapshot/env.txt"

# Đồng bộ lên Drive mỗi SYNC_EVERY_MIN phút (mặc định 30; 0 = tắt): máy chết giữa chừng vẫn còn các run đã xong.
SYNC_EVERY_MIN="${SYNC_EVERY_MIN:-30}"
if command -v rclone &>/dev/null && [[ "$SYNC_EVERY_MIN" -gt 0 ]]; then
    (
        while sleep $((SYNC_EVERY_MIN * 60)); do
            rclone copy "$REPO_DIR/data/training/update-model" "$RCLONE_REMOTE/$RUN_ID/update-model" \
                --exclude "vast_logs_*" --exclude "**/cache/**" --exclude "**/last.ckpt" -q || true
            if [[ -d "$DATA_DIR/relearn/active" ]]; then
                rclone copy "$DATA_DIR/relearn/active" "$RCLONE_REMOTE/$RUN_ID/relearn_active" -q || true
            fi
            rclone copy "$LOG_DIR" "$RCLONE_REMOTE/$RUN_ID/logs" -q || true
            echo "[SYNC] $(date '+%F %T') đã đồng bộ giữa chừng" >> "$LOG_DIR/sync.log"
        done
    ) &
    SYNC_PID=$!
    echo "Đồng bộ định kỳ mỗi $SYNC_EVERY_MIN phút (PID $SYNC_PID)" | tee -a "$MAIN_LOG"
fi

# ----------------- BƯỚC 2: CHẠY HUẤN LUYỆN THEO TASK -----------------
echo "Bắt đầu chạy task: $TASK..." | tee -a "$MAIN_LOG"

case "$TASK" in
    al_dot1)
        echo ">> Chạy Active Learning đợt 1 (7 run B=500)..." | tee -a "$MAIN_LOG"
        bash scripts/run_al.sh dot1 2>&1 | tee -a "$MAIN_LOG"
        ;;
    al_next)
        # Vòng 2 từ model tốt nhất (margin_k) + random so sánh, rồi đợt 2 (B=250, 850). Cần khôi phục campaign dot1 trước.
        CAMP="$REPO_DIR/data/training/update-model/relearn-fix-v2"
        for v in al_margin_k_B500 al_random_B500_s0; do
            for f in "runs/EXP-FB-FT-$v/task_5/task_final.pth" "runs/EXP-FB-FT-$v/task_5/pred_val.npz" "stages/$v.json" "scored/$v/calibration_count_m2.json"; do
                [[ -f "$CAMP/$f" ]] || { echo "[THIẾU] $CAMP/$f -> khôi phục campaign từ Drive trước (bước B5)" | tee -a "$MAIN_LOG"; exit 3; }
            done
        done
        [[ -f "$CAMP/stages/baseline.json" && -f "$CAMP/plan.json" ]] || { echo "[THIẾU] stages/baseline.json hoặc plan.json" | tee -a "$MAIN_LOG"; exit 3; }
        echo ">> Vòng 2 (margin_k_r2, random_s0_r2) rồi đợt 2 (dot2)..." | tee -a "$MAIN_LOG"
        bash scripts/run_al.sh al_margin_k_B500_r2 al_random_B500_s0_r2 dot2 2>&1 | tee -a "$MAIN_LOG"
        ;;
    al_dot2)
        echo ">> Chạy Active Learning đợt 2 (B=250, 850 + entropy)..." | tee -a "$MAIN_LOG"
        bash scripts/run_al.sh dot2 2>&1 | tee -a "$MAIN_LOG"
        ;;
    relearn_pilot)
        echo ">> Chạy Học lại Pilot (17 SKU)..." | tee -a "$MAIN_LOG"
        bash scripts/run_relearn_fix.sh pilot 2>&1 | tee -a "$MAIN_LOG"
        ;;
    relearn_e1)
        echo ">> Chạy Học lại E1 (63 SKU)..." | tee -a "$MAIN_LOG"
        bash scripts/run_relearn_fix.sh e1 2>&1 | tee -a "$MAIN_LOG"
        ;;
    naive)
        echo ">> Chạy Baseline Naive Finetuning..." | tee -a "$MAIN_LOG"
        python -m tools.run_naive --settings configs/training/full_syn_naive.json --last 3 2>&1 | tee -a "$MAIN_LOG"
        ;;
    *)
        echo ">> Chạy lệnh tùy biến: $TASK..." | tee -a "$MAIN_LOG"
        eval "$TASK" 2>&1 | tee -a "$MAIN_LOG"
        ;;
esac

echo "Task $TASK đã hoàn thành tốt đẹp!" | tee -a "$MAIN_LOG"
