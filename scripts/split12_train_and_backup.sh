#!/usr/bin/env bash
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
export AC_DATA="${AC_DATA:-$(cd "$REPO/../autocheckout-data" 2>/dev/null && pwd || echo /workspace/autocheckout-data)}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive:AutoCheckout/ket_qua_gpu}"
SYNC_EVERY_MIN="${SYNC_EVERY_MIN:-20}"
AUTO_DESTROY="${AUTO_DESTROY:-0}"; AUTO_STOP="${AUTO_STOP:-0}"
SETTINGS="${SETTINGS:-configs/training/full_syn_12x.json}"
CAMPAIGN="${CAMPAIGN:-data/training/full-syn-12x}"

MODE="${1:-}"; shift || true
EXTRA=("$@")
[[ "$MODE" == "--" ]] && { MODE=custom; CUSTOM=("$@"); }
case "$MODE" in smoke|train|custom) ;; *) sed -n 2,25p "$0"; exit 2;; esac

RUN_ID="${RESUME_FROM:-split12_${MODE}_$(date '+%Y%m%d_%H%M%S')}"
DEST="$RCLONE_REMOTE/$RUN_ID"
LOG_DIR="$REPO/data/training/split12_logs/$RUN_ID"; mkdir -p "$LOG_DIR" "$CAMPAIGN"
MAIN_LOG="$LOG_DIR/master_execution.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"; }

VENV="${VENV:-/workspace/venv-pdp}"
if [[ -z "${VIRTUAL_ENV:-}" && -f "$VENV/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
if ! python -c "import sys, torch, pytorch_lightning, transformers, pycocotools; assert sys.version_info[:2] == (3, 10)" 2>/dev/null; then
    echo "Python không đúng môi trường: $(command -v python) ($(python --version 2>&1))."
    echo "Cần venv Python 3.10 có torch, pytorch_lightning, transformers, pycocotools (mục 5 hướng dẫn):"
    echo "  source /workspace/venv-pdp/bin/activate   (chưa có thì tạo theo mục 5)"
    exit 3
fi
command -v rclone >/dev/null || { echo "THIẾU rclone: cài và cấu hình trước (bước 4 hướng dẫn). Không chạy khi chưa có backup."; exit 3; }
rclone lsd "$RCLONE_REMOTE" >/dev/null 2>&1 || rclone mkdir "$RCLONE_REMOTE" || { echo "Không ghi được $RCLONE_REMOTE"; exit 3; }

sync_now() {
    rclone copy "$CAMPAIGN" "$DEST/campaign" --exclude "**/cache/**" --exclude "**/__pycache__/**" "$1" || true
    {   echo "RUN_ID=$RUN_ID  cập nhật $(date '+%F %T')"
        echo "stages:"; ls -1 "$CAMPAIGN/stages" 2>/dev/null | sed 's/^/  /'
        echo "task_final.pth:"; find "$CAMPAIGN" -type f -name task_final.pth 2>/dev/null | sed "s|$CAMPAIGN/|  |" | sort
        echo "đang chạy (log cuối):"; tail -n 5 "$MAIN_LOG" 2>/dev/null | sed 's/^/  /'
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /'
        df -h /workspace 2>/dev/null | tail -1 | sed 's/^/Disk: /'
    } > "$LOG_DIR/progress.txt"
    rclone copy "$LOG_DIR" "$DEST/logs" "$1" || true
}

finish() {
    local CODE=$?
    [[ -n "${SYNC_PID:-}" ]] && { pkill -P "$SYNC_PID" 2>/dev/null; kill "$SYNC_PID" 2>/dev/null; } || true
    log "KẾT THÚC mã $CODE — đồng bộ lần cuối lên $DEST"
    { echo "End: $(date '+%F %T')"; echo "Exit: $CODE"; } >> "$LOG_DIR/run_metadata.txt"
    (cd "$CAMPAIGN" && find . -type f -name "*.pth" -exec md5sum {} \; | sort -k2) > "$LOG_DIR/weights_md5.txt" 2>/dev/null || true
    sync_now -v
    local N_LOCAL N_REMOTE
    N_LOCAL=$(find "$CAMPAIGN" -type f -name "*.pth" | wc -l)
    N_REMOTE=$(rclone ls "$DEST/campaign" --include "*.pth" 2>/dev/null | wc -l)
    if rclone check "$CAMPAIGN" "$DEST/campaign" --one-way --size-only --exclude "**/cache/**" --exclude "**/__pycache__/**" -q 2>>"$LOG_DIR/rclone_check.txt"; then
        log "[KIỂM TRA] Drive đủ file (.pth máy $N_LOCAL / Drive $N_REMOTE)"
    else
        log "[CẢNH BÁO] Drive thiếu file (.pth máy $N_LOCAL / Drive $N_REMOTE) -> KHÔNG destroy, chỉ STOP. Vào lại máy, chạy: rclone copy $CAMPAIGN $DEST/campaign"
        AUTO_DESTROY=0; AUTO_STOP=1
    fi
    rclone copy "$LOG_DIR" "$DEST/logs" -q || true
    if [[ "$AUTO_DESTROY" == 1 || "$AUTO_STOP" == 1 ]] && [[ -n "${VAST_API_KEY:-}" ]]; then
        pip install -q -U vastai >/dev/null 2>&1; vastai set api-key "$VAST_API_KEY" >/dev/null
        ID="${INSTANCE_ID:-$(vastai show instances --raw 2>/dev/null | python -c 'import sys,json;d=json.load(sys.stdin);print(d[0]["id"] if d else "")' 2>/dev/null)}"
        if [[ -n "$ID" ]]; then
            [[ "$AUTO_DESTROY" == 1 ]] && { log "DESTROY máy $ID"; vastai destroy instance "$ID"; } || { log "STOP máy $ID"; vastai stop instance "$ID"; }
        else log "Không dò được INSTANCE_ID: tự Destroy trên web Vast"; fi
    fi
}
trap finish EXIT

if [[ -n "${RESUME_FROM:-}" ]]; then
    log "Khôi phục campaign từ $DEST/campaign"
    rclone copy "$DEST/campaign" "$CAMPAIGN" --progress || { log "Không kéo được campaign"; exit 4; }
    log "Đã có $(find "$CAMPAIGN" -type f -name task_final.pth | wc -l) task_final.pth, stages: $(ls "$CAMPAIGN/stages" 2>/dev/null | tr '\n' ' ')"
fi

{   echo "Run: $RUN_ID  Mode: $MODE  Start: $(date '+%F %T')  Host: $(hostname)"
    echo "Branch: $(git branch --show-current)  Commit: $(git rev-parse HEAD)"
    nvidia-smi; free -h; df -h /workspace 2>/dev/null
    python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
    pip list 2>/dev/null
} > "$LOG_DIR/run_metadata.txt" 2>&1
mkdir -p "$LOG_DIR/config_snapshot"
cp -r configs/training "$LOG_DIR/config_snapshot/" 2>/dev/null; cp configs/tasks_100-8x12-13_seed0.json "$LOG_DIR/config_snapshot/" 2>/dev/null
git status --porcelain > "$LOG_DIR/config_snapshot/git_status.txt"; git diff > "$LOG_DIR/config_snapshot/git_diff.patch"
{ echo "AC_DATA=$AC_DATA"; echo "SETTINGS=$SETTINGS"; echo "CAMPAIGN=$CAMPAIGN"; echo "RCLONE_REMOTE=$RCLONE_REMOTE"
  for f in synth_v2_12x/manifest.json synth_v2_12x/task_config.json eval_12x/manifest.json eval_12x/val_full.json \
           models/full-syn-r1/runs/EXP-FULL-SYN/task_1/task_final.pth; do
      [[ -f "$AC_DATA/$f" ]] && echo "md5 $(md5sum "$AC_DATA/$f")" || echo "THIẾU $AC_DATA/$f"; done
} > "$LOG_DIR/config_snapshot/env_and_inputs.txt"

sync_now -q
if [[ "$SYNC_EVERY_MIN" -gt 0 ]]; then
    ( while sleep $((SYNC_EVERY_MIN * 60)); do sync_now -q; echo "[SYNC] $(date '+%F %T')" >> "$LOG_DIR/sync.log"; done ) &
    SYNC_PID=$!; log "Backup lên $DEST mỗi $SYNC_EVERY_MIN phút (PID $SYNC_PID)"
fi

need() { [[ -e "$1" ]] || { log "[THIẾU] $1"; MISSING=1; }; }
MISSING=0
for f in synth_v2_12x/images/task_1 synth_v2_12x/images/task_9 synth_v2_12x/tasks/synth_v2/train_task_9.json \
         eval_12x/val_full.json eval_12x/val_task_9.json rpc/val2019 pretrained/hf-cache \
         models/full-syn-r1/runs/EXP-FULL-SYN/task_1/task_final.pth models/full-syn-r1/runs/EXP-FULL-SYN-FSA/task_1/hf_model \
         relearn/active/splits/val_a_split.json relearn/meta/val_b.json; do need "$AC_DATA/$f"; done
need tools/run_split.py; need "$SETTINGS"; need configs/tasks_100-8x12-13_seed0.json
[[ "$MISSING" == 0 || "$MODE" == custom ]] || { log "Thiếu đầu vào/code (xem trên). Code 12x phải xong Task 1–7 của 01-plan và đã push."; exit 3; }

case "$MODE" in
  smoke)
    log "== SMOKE 1/4: mã kiểm dữ liệu 12x"
    (cd "$AC_DATA" && sha256sum -c --quiet SHA256SUMS_12x) 2>&1 | tee -a "$MAIN_LOG"; [[ ${PIPESTATUS[0]} == 0 ]] || exit 10
    log "== SMOKE 2/4: test CPU"
    python -m pytest -q -p no:cacheprovider tests/test_make_task_config.py tests/test_pdp_expand.py tests/test_full_syn.py \
        tests/test_al_score.py tests/test_run_split.py -k "not naive_lower_bound" 2>&1 | tee -a "$MAIN_LOG"
    [[ ${PIPESTATUS[0]} == 0 ]] || exit 11
    log "== SMOKE 3/4: dry run cả chuỗi trên dữ liệu đồ chơi (CPU, vài phút)"
    python -m tools.run_split --settings "$SETTINGS" --dry-run 2>&1 | tee -a "$MAIN_LOG"; [[ ${PIPESTATUS[0]} == 0 ]] || exit 12
    log "== SMOKE 4/4: smoke GPU: chia lại checkpoint task 1 + task 2 chạy 200 bước, dừng, chạy tiếp"
    python -m tools.run_split --settings "$SETTINGS" --smoke 2>&1 | tee -a "$MAIN_LOG"; [[ ${PIPESTATUS[0]} == 0 ]] || exit 13
    log "SMOKE ĐẠT"
    ;;
  train)
    log "== TRAIN full-syn-12x (task 1 dự đoán lại -> task 2..9, có cổng dừng)"
    python -m tools.run_split --settings "$SETTINGS" "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 | tee -a "$MAIN_LOG"; exit "${PIPESTATUS[0]}"
    ;;
  custom)
    log "== LỆNH: ${CUSTOM[*]}"
    "${CUSTOM[@]}" 2>&1 | tee -a "$MAIN_LOG"; exit "${PIPESTATUS[0]}"
    ;;
esac
