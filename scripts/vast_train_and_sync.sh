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
    echo ""
    echo "=========================================================="
    echo " QUÁ TRÌNH HUẤN LUYỆN KẾT THÚC VỚI MÃ: $EXIT_CODE"
    echo " BẮT ĐẦU ĐỒNG BỘ KẾT QUẢ VỀ GOOGLE DRIVE..."
    echo "=========================================================="

    # 1. Lưu lại metadata kết thúc
    echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_DIR/run_metadata.txt"
    echo "Exit Code: $EXIT_CODE" >> "$LOG_DIR/run_metadata.txt"

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

# ----------------- BƯỚC 2: CHẠY HUẤN LUYỆN THEO TASK -----------------
echo "Bắt đầu chạy task: $TASK..." | tee -a "$MAIN_LOG"

case "$TASK" in
    al_dot1)
        echo ">> Chạy Active Learning đợt 1 (7 run B=500)..." | tee -a "$MAIN_LOG"
        bash scripts/run_al.sh dot1 2>&1 | tee -a "$MAIN_LOG"
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
