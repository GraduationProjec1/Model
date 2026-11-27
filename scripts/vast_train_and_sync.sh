#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

DATA_DIR="${AC_DATA:-$(cd "$REPO_DIR/../autocheckout-data" 2>/dev/null && pwd || echo "/workspace/autocheckout-data")}"
export AC_DATA="$DATA_DIR"

RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive:AutoCheckout_Runs}"

AUTO_DESTROY="${AUTO_DESTROY:-1}"
AUTO_STOP="${AUTO_STOP:-0}"

VAST_API_KEY="${VAST_API_KEY:-}"
INSTANCE_ID="${INSTANCE_ID:-}"

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

finish_and_cleanup() {
    local EXIT_CODE=$?
    if [[ -n "${SYNC_PID:-}" ]]; then pkill -P "$SYNC_PID" 2>/dev/null || true; kill "$SYNC_PID" 2>/dev/null || true; fi
    echo ""
    echo "=========================================================="
    echo " QUÁ TRÌNH HUẤN LUYỆN KẾT THÚC VỚI MÃ: $EXIT_CODE"
    echo " BẮT ĐẦU ĐỒNG BỘ KẾT QUẢ VỀ GOOGLE DRIVE..."
    echo "=========================================================="

    echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_DIR/run_metadata.txt"
    echo "Exit Code: $EXIT_CODE" >> "$LOG_DIR/run_metadata.txt"

    if [[ -d "$DATA_DIR/relearn/active" ]]; then
        mkdir -p "$REPO_DIR/data/training/update-model/relearn_active_backup"
        cp -r "$DATA_DIR/relearn/active/." "$REPO_DIR/data/training/update-model/relearn_active_backup/" || true
    fi
    (cd "$REPO_DIR/data/training/update-model" 2>/dev/null && find . -name "*.pth" -exec md5sum {} \; | sort -k2) \
        > "$LOG_DIR/weights_md5.txt" 2>/dev/null || true

    if command -v rclone &>/dev/null; then
        echo "[RCLONE] Đang upload log và stages..."
        rclone copy "$LOG_DIR" "$RCLONE_REMOTE/$RUN_ID/logs" --verbose || true

        echo "[RCLONE] Đang upload data/training/update-model..."
        if [[ -d "$REPO_DIR/data/training/update-model" ]]; then
            rclone copy "$REPO_DIR/data/training/update-model" "$RCLONE_REMOTE/$RUN_ID/update-model" \
                --exclude "vast_logs_*" --verbose || true
        fi

        local N_LOCAL N_REMOTE
        N_LOCAL=$(find "$REPO_DIR/data/training/update-model" -type f -name "*.pth" 2>/dev/null | wc -l)
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

    if [[ "$AUTO_DESTROY" == "1" || "$AUTO_STOP" == "1" ]]; then
        if [[ -n "$VAST_API_KEY" ]]; then
            pip install --quiet --upgrade vastai || true
            vastai set api-key "$VAST_API_KEY" || true

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

echo "Bắt đầu chạy task: $TASK..." | tee -a "$MAIN_LOG"

FAILED_STEPS=()
step() {
    local name=$1; shift
    "$@" 2>&1 | tee -a "$MAIN_LOG"
    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        echo "[LỖI] bước $name thoát mã $rc (xem $MAIN_LOG)" | tee -a "$MAIN_LOG"
        FAILED_STEPS+=("$name")
    fi
}

case "$TASK" in
    al_dot1)
        echo ">> Chạy Active Learning đợt 1 (7 run B=500)..." | tee -a "$MAIN_LOG"
        bash scripts/run_al.sh dot1 2>&1 | tee -a "$MAIN_LOG"
        ;;
    al_next)
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
    al_r3)
        CAMP="$REPO_DIR/data/training/update-model/relearn-fix-v2"
        R3_RUNS=(al_margin_k_B500_r3_rows al_margin_k_B500_r3_train al_margin_k_B500_r3_real al_margin_k_B500_r3_all
                 al_margin_k_B500_r3_prompt)
        NEED=("$CAMP/plan.json" "$CAMP/stages/baseline.json" "$CAMP/stages/al_margin_k_B500_r2.json"
              "$CAMP/runs/EXP-FB-FT-al_margin_k_B500_r2/task_5/task_final.pth"
              "$CAMP/runs/EXP-FB-FT-al_margin_k_B500_r2/task_5/pred_val.npz"
              "$CAMP/scored/al_margin_k_B500_r2/calibration_count_m2.json"
              "$DATA_DIR/relearn/active/selections/margin_k_B500_r2.json"
              "$DATA_DIR/relearn/active/splits/val_a_split.json")
        for v in "${R3_RUNS[@]}"; do
            NEED+=("$DATA_DIR/relearn/variants/$v/train_task_5_fix.json" "$DATA_DIR/relearn/variants/$v/train_rows.json")
        done
        for f in "${NEED[@]}"; do
            [[ -f "$f" ]] || { echo "[THIẾU] $f -> chạy gpu_bundle_r3/install.sh trước" | tee -a "$MAIN_LOG"; exit 3; }
        done
        echo ">> Vòng 3: rows, train, real, all, prompt (~10 giờ 4090)..." | tee -a "$MAIN_LOG"
        bash scripts/run_al.sh "${R3_RUNS[@]}" 2>&1 | tee -a "$MAIN_LOG"
        ;;
    al_r4)
        CAMP="$REPO_DIR/data/training/update-model/relearn-fix-v2"
        SEL="$DATA_DIR/relearn/active/selections"
        NEED=("$CAMP/plan.json" "$CAMP/stages/baseline.json" "$SEL/margin_k_B500_r2.json" "$SEL/margin_k_B500_r3.json"
              "$DATA_DIR/relearn/active/splits/val_a_split.json" "$DATA_DIR/relearn/variants/e1/train_task_5_fix.json"
              "$DATA_DIR/relearn/variants/al_margin_k_B500_r3_real/train_task_5_fix.json"
              "$DATA_DIR/relearn/variants/al_margin_k_B500_r3_real/train_rows.json")
        for v in al_margin_k_B500_r2 al_margin_k_B500_r3_real al_margin_k_B500_r3_all; do
            NEED+=("$CAMP/stages/$v.json" "$CAMP/runs/EXP-FB-FT-$v/task_5/task_final.pth"
                   "$CAMP/runs/EXP-FB-FT-$v/task_5/pred_val.npz" "$CAMP/scored/$v/calibration_count_m2.json"
                   "$CAMP/scored/$v/calibration_count_groups.json")
        done
        for f in "${NEED[@]}"; do
            [[ -f "$f" ]] || { echo "[THIẾU] $f -> chạy scripts/install_r4.sh trước" | tee -a "$MAIN_LOG"; exit 3; }
        done
        WMD5="${WEIGHTS_MD5:-$CAMP/../weights_md5.txt}"
        if [[ "${SKIP_MD5:-0}" != "1" ]]; then
            [[ -f "$WMD5" ]] || { echo "[THIẾU] $WMD5 (install_r4.sh bước 3; SKIP_MD5=1 để bỏ qua có chủ ý)" | tee -a "$MAIN_LOG"; exit 3; }
            (cd "$CAMP/.." && md5sum -c --quiet --ignore-missing "$WMD5") || { echo "[THIẾU] md5 weights lệch $WMD5" | tee -a "$MAIN_LOG"; exit 3; }
        fi
        echo ">> B0: test2019 cho r3_real và r3_all (chấm một lần, ~1 giờ)..." | tee -a "$MAIN_LOG"
        step B0 bash scripts/run_al.sh test al_margin_k_B500_r3_real al_margin_k_B500_r3_all --confirm-test
        echo ">> B4: al_margin_k_B500_r3_real_s1 (seed 1, ~2,7 giờ) rồi B1: al_margin_k_B500_r4 (~3 giờ)..." | tee -a "$MAIN_LOG"
        step B4+B1 bash scripts/run_al.sh al_margin_k_B500_r3_real_s1 al_margin_k_B500_r4
        if [[ "${R4_FREEZE:-0}" == "1" ]]; then
            echo ">> B3: al_margin_k_B500_r3_freeze (lr_old 0, dữ liệu của real, ~2,7 giờ)..." | tee -a "$MAIN_LOG"
            step B3 env R3_FREEZE_DATA=real bash scripts/run_al.sh al_margin_k_B500_r3_freeze
        fi
        ;;
    attn_e1|attn_e2|attn_e3|attn_e4|attn_e5|attn_m1|n3_e|n2_e)
        # docs/ai/planning/2026-11-25-ke-hoach-chay.md, 2026-11-25-plan-B-C1.md: G0 (P1 trên GPU) rồi một lượt r4 với
        # E1 = A (prompt theo từng query), E2 = A + B (prompt hiệu chỉnh cross-attn), E3 = chỉ B,
        # E4 = A + C1 (K = 100 query đề xuất từ encoder), E5 = A + B + C1;
        # M1 = A + loss biên prototype (2026-11-26-plan-M1-M6-kiem-prototype.md §3);
        # 2026-11-26-plan-N1-N5-hieu-chinh-task-va-loss.md: N3 = focal phạt lớp cùng nhóm, w = 2 rồi w = 3 (§4.4),
        # N2 = chưng cất logit từ teacher, λ = 1 (§5.4); cả hai giữ cấu hình r4
        CAMP="$REPO_DIR/data/training/update-model/relearn-fix-v2"
        R3="$CAMP/runs/EXP-FB-FT-al_margin_k_B500_r3_real/task_5/task_final.pth"
        case "$TASK" in
            attn_e1) VS=al_margin_k_B500_r4_pq ;;
            attn_e2) VS=al_margin_k_B500_r4_pqx ;;
            attn_e3) VS=al_margin_k_B500_r4_x ;;
            attn_e4) VS=al_margin_k_B500_r4_pqc ;;
            attn_e5) VS=al_margin_k_B500_r4_pqxc ;;
            attn_m1) VS=al_margin_k_B500_r4_pqm ;;
            n3_e) VS="al_margin_k_B500_r4_g2 al_margin_k_B500_r4_g3" ;;
            n2_e) VS=al_margin_k_B500_r4_lkd ;;
        esac
        IMG_ROOT="$DATA_DIR/rpc"; [[ -d "$IMG_ROOT/val2019" ]] || IMG_ROOT="$DATA_DIR/archive"
        NEED=("$CAMP/plan.json" "$CAMP/stages/baseline.json" "$CAMP/stages/al_margin_k_B500_r3_real.json" "$R3"
              "$CAMP/runs/EXP-FB-FT-al_margin_k_B500_r3_real/task_5/pred_val.npz"
              "$DATA_DIR/relearn/active/selections/margin_k_B500_r3.json"
              "$DATA_DIR/relearn/active/splits/val_a_split.json" "$DATA_DIR/relearn/variants/e1/train_task_5_fix.json"
              "$DATA_DIR/relearn/variants/al_margin_k_B500_r3_real/train_rows.json" "$DATA_DIR/eval/val_full.json")
        for f in "${NEED[@]}"; do
            [[ -f "$f" ]] || { echo "[THIẾU] $f -> chạy scripts/install_r4.sh trước" | tee -a "$MAIN_LOG"; exit 3; }
        done
        [[ -d "$IMG_ROOT/val2019" ]] || { echo "[THIẾU] ảnh val2019 trong $DATA_DIR/rpc hoặc $DATA_DIR/archive" | tee -a "$MAIN_LOG"; exit 3; }
        WMD5="${WEIGHTS_MD5:-$CAMP/../weights_md5.txt}"
        if [[ "${SKIP_MD5:-0}" != "1" ]]; then
            [[ -f "$WMD5" ]] || { echo "[THIẾU] $WMD5 (install_r4.sh bước 3; SKIP_MD5=1 để bỏ qua có chủ ý)" | tee -a "$MAIN_LOG"; exit 3; }
            (cd "$CAMP/.." && md5sum -c --quiet --ignore-missing "$WMD5") || { echo "[THIẾU] md5 weights lệch $WMD5" | tee -a "$MAIN_LOG"; exit 3; }
        fi
        echo ">> G0: P1 trên GPU với r3_real (tốc độ, kết quả không đổi; ~15 phút)..." | tee -a "$MAIN_LOG"
        step G0 python -m tools.p1_check --checkpoint "$R3" --val "$DATA_DIR/eval/val_full.json" --img-root "$IMG_ROOT" \
            --split "$DATA_DIR/relearn/active/splits/val_a_split.json" --device cuda --out "$LOG_DIR/g0_p1_check"
        REUSE="${REUSE_FEATURES:-1}"
        if [[ " ${FAILED_STEPS[*]:-} " == *" G0 "* ]]; then
            REUSE=0
            echo "[CẢNH BÁO] G0 không đạt: ${TASK#attn_} chạy với REUSE_FEATURES=0" | tee -a "$MAIN_LOG"
        fi
        for V in $VS; do
            echo ">> ${TASK#attn_}: $V (REUSE_FEATURES=$REUSE, ~2,5-3 giờ)..." | tee -a "$MAIN_LOG"
            step "${TASK#attn_}" env REUSE_FEATURES="$REUSE" bash scripts/run_al.sh "$V"
            echo ">> Phân rã lỗi của $V..." | tee -a "$MAIN_LOG"
            step BREAKDOWN python -m tools.error_breakdown --val "$CAMP/inputs/val_b/val_full.json" \
                --pred "$CAMP/runs/EXP-FB-FT-$V/task_5/pred_val.npz" --policy "$DATA_DIR/relearn/active/scored/$V.json" \
                --task-config configs/relearn/task_config_relearn.json --rows "$CAMP/inputs/$V/train_rows.json" \
                --out "$REPO_DIR/data/training/update-model/error_breakdown_$V"
        done
        ;;
    n0_dump)
        # 2026-11-26-plan-N1-N5-hieu-chinh-task-va-loss.md §2.3: r4 dự đoán lại ở độ phân giải gốc kèm logit (không train),
        # kiểm pred_val.npz cho đúng cAcc của r4 qua al_score; đúng thì đo N0 (swap_rank trên Val-A-calib) và N1
        # (task_calib_search: fit trên calib, chấm Val-B một lần)
        CAMP="$REPO_DIR/data/training/update-model/relearn-fix-v2"
        R4_RUN="${R4_RUN:-$CAMP/runs/EXP-FB-FT-al_margin_k_B500_r4/task_5}"
        SCORED="$DATA_DIR/relearn/active/scored"
        IMG_ROOT="$DATA_DIR/rpc"; [[ -d "$IMG_ROOT/val2019" ]] || IMG_ROOT="$DATA_DIR/archive"
        for f in "$R4_RUN/task_final.pth" "$R4_RUN/run_info.json" "$DATA_DIR/eval/val_full.json" \
                 "$DATA_DIR/eval/val_task_5.json" "$SCORED/al_margin_k_B500_r4.json"; do
            [[ -f "$f" ]] || { echo "[THIẾU] $f (r4: task_final.pth + run_info.json về \$R4_RUN, điểm r4 của al_score)" | tee -a "$MAIN_LOG"; exit 3; }
        done
        OUT="$CAMP/runs/N0-r4-logits"
        echo ">> N0: r4 dự đoán lại kèm logit (~30-60 phút)..." | tee -a "$MAIN_LOG"
        step N0_PREDICT python -m tools.predict_resolution --run-info "$R4_RUN/run_info.json" \
            --checkpoint "$R4_RUN/task_final.pth" --resolution 0 --dump-logits --out "$OUT" \
            --ann-dir "$DATA_DIR/eval" --img-root "$IMG_ROOT"
        step N0_SCORE python -m tools.al_score --name n0_r4_replay --pred "$OUT/task_5/pred_val.npz"
        step N0_CHECK python -c 'import json, sys
a, b = (json.load(open(p))["cAcc_m2"] for p in sys.argv[1:])
print(f"N0: cAcc lượt dump {a:.6f}, r4 {b:.6f}")
sys.exit(abs(a - b) > 1e-9)' "$SCORED/n0_r4_replay.json" "$SCORED/al_margin_k_B500_r4.json"
        if [[ " ${FAILED_STEPS[*]:-} " == *" N0_"* ]]; then
            echo "[LỖI] N0: lượt dump không ra đúng cAcc của r4 -> dừng, không đo N0/N1" | tee -a "$MAIN_LOG"
        else
            mkdir -p "$REPO_DIR/data/training/update-model/n1_n5"
            step N0_SWAP_RANK python -m tools.swap_rank --name r4 --logits "$OUT/task_5/logits_val.npz" \
                --policy "$SCORED/al_margin_k_B500_r4.json" --out "$REPO_DIR/data/training/update-model/n1_n5/swap_rank_r4.json"
            step N1_TASK_CALIB python -m tools.task_calib_search --name r4 --logits "$OUT/task_5/logits_val.npz" \
                --out "$REPO_DIR/data/training/update-model/n1_n5/task_calib_r4.json"
        fi
        ;;
    m6_res)
        # 2026-11-26-plan-M1-M6-kiem-prototype.md §2.2: r4 dự đoán lại ở nhiều độ phân giải (không train), chấm từng cái
        CAMP="$REPO_DIR/data/training/update-model/relearn-fix-v2"
        R4_RUN="${R4_RUN:-$CAMP/runs/EXP-FB-FT-al_margin_k_B500_r4/task_5}"
        IMG_ROOT="$DATA_DIR/rpc"; [[ -d "$IMG_ROOT/val2019" ]] || IMG_ROOT="$DATA_DIR/archive"
        for f in "$R4_RUN/task_final.pth" "$R4_RUN/run_info.json" "$DATA_DIR/eval/val_full.json" \
                 "$DATA_DIR/eval/val_task_5.json"; do
            [[ -f "$f" ]] || { echo "[THIẾU] $f (r4: tải task_final.pth + run_info.json về \$R4_RUN)" | tee -a "$MAIN_LOG"; exit 3; }
        done
        for N in ${M6_RESOLUTIONS:-640 800 960}; do
            OUT="$CAMP/runs/M6-r4-res$N"
            echo ">> M6: r4 ở độ phân giải $N..." | tee -a "$MAIN_LOG"
            step "M6_$N" python -m tools.predict_resolution --run-info "$R4_RUN/run_info.json" \
                --checkpoint "$R4_RUN/task_final.pth" --resolution "$N" --out "$OUT" \
                --ann-dir "$DATA_DIR/eval" --img-root "$IMG_ROOT"
            step "M6_SCORE_$N" python -m tools.al_score --name "m6_r4_res$N" --pred "$OUT/task_5/pred_val.npz"
        done
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

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    echo "[LỖI] Task $TASK: bước lỗi ${FAILED_STEPS[*]}; máy chỉ STOP, không destroy" | tee -a "$MAIN_LOG" "$LOG_DIR/run_metadata.txt"
    AUTO_DESTROY=0; AUTO_STOP=1
    exit 1
fi
echo "Task $TASK đã hoàn thành tốt đẹp!" | tee -a "$MAIN_LOG"
