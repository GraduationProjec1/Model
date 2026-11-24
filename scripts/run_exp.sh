#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?usage: run_exp.sh configs/exp/<name>.sh [--shutdown]}
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-/data/rpc}
RUNS=${RUNS:-/data/runs}
PYTHON=${PYTHON:-python}
PRED_SPLITS=${PRED_SPLITS:-val,test}
SHUTDOWN_CMD=${SHUTDOWN_CMD:-sudo shutdown -h now}

if [[ ${2:-} == --shutdown ]]; then
    trap 'echo "run_exp.sh: exit status $?, shutting down"; $SHUTDOWN_CMD' EXIT
fi

# shellcheck source=/dev/null
source "$CONFIG"
: "${EXP:?config must set EXP}" "${N_TASKS:?config must set N_TASKS}" "${TASK_CFG:?}" "${TASK_DIR:?}"
RUN_DIR=$RUNS/$EXP
mkdir -p "$RUN_DIR"
cp "$CONFIG" "$RUN_DIR/config.sh"
echo "== $EXP: $N_TASKS tasks in $RUN_DIR ($(date '+%F %T'))"

if [[ -n ${REUSE_TASK1:-} && ! -e $RUN_DIR/task_1 ]]; then
    ln -s "$RUNS/$REUSE_TASK1/task_1" "$RUN_DIR/task_1"
    echo "task 1 reused from $REUSE_TASK1"
fi

for t in $(seq "${START_TASK:-1}" "$N_TASKS"); do
    task_dir=$RUN_DIR/task_$t
    mode=()
    missing=0
    for split in ${PRED_SPLITS//,/ }; do
        [[ -f $task_dir/pred_$split.npz ]] || missing=1
    done
    if [[ -f $task_dir/task_final.pth ]]; then
        if [[ $missing == 0 ]]; then
            echo "-- task $t: done, skipped"
            continue
        fi
        echo "-- task $t: trained, rewriting predictions"
        mode=(--predict_only 1)
    else
        echo "-- task $t: training ($(date '+%F %T'))"
    fi
    (cd "$REPO/pdp" && $PYTHON main.py "${ARGS[@]}" --output_dir "$RUN_DIR" --start_task "$t" --n_tasks "$t" \
        --pred_splits "$PRED_SPLITS" "${mode[@]+"${mode[@]}"}")
done

if [[ -n ${SKIP_EVAL:-} ]]; then
    echo "== $EXP finished, evaluation skipped ($(date '+%F %T'))"
    exit 0
fi
echo "-- evaluation ($(date '+%F %T'))"
cd "$REPO"
for split in val test; do
    $PYTHON -m tools.eval_cl --run-dir "$RUN_DIR" --split "$split" --ann "$TASK_DIR/${split}_full.json" \
        --task-config "$TASK_CFG" > "$RUN_DIR/metrics_cl_$split.md"
done
if [[ ${EVAL_MODE:-legacy} == calibrated ]]; then
    policy=${CALIBRATION_POLICY:-$RUN_DIR/calibration_count.json}
    if [[ ! -f $policy ]]; then
        calibration_args=(--nms-grid "${COUNT_NMS_GRID:-0.45}")
        if [[ -n ${COUNT_THRESHOLD_GRID:-} ]]; then
            calibration_args+=(--threshold-grid "$COUNT_THRESHOLD_GRID")
        fi
        $PYTHON -m tools.eval_count --run-dir "$RUN_DIR" --val-ann "$TASK_DIR/val_full.json" \
            --task-config "$TASK_CFG" --calibrate-only --policy-out "$policy" "${calibration_args[@]}" \
            > "$RUN_DIR/calibration_count.log"
    fi
    $PYTHON -m tools.eval_count --run-dir "$RUN_DIR" --test-ann "$TASK_DIR/test_full.json" \
        --task-config "$TASK_CFG" --policy-in "$policy" > "$RUN_DIR/metrics_count_test.md"
    echo "== $EXP finished ($(date '+%F %T'))"
    exit 0
fi
$PYTHON -m tools.eval_count --run-dir "$RUN_DIR" --val-ann "$TASK_DIR/val_full.json" \
    --test-ann "$TASK_DIR/test_full.json" --task-config "$TASK_CFG" > "$RUN_DIR/metrics_count_test.md"
$PYTHON -m tools.eval_count --run-dir "$RUN_DIR" --val-ann "$TASK_DIR/val_full.json" \
    --test-ann "$TASK_DIR/test_full.json" --task-config "$TASK_CFG" --nms-iou 0.5 > "$RUN_DIR/metrics_count_test_nms0.5.md"
echo "== $EXP finished ($(date '+%F %T'))"
