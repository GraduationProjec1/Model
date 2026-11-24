source "$REPO/configs/exp/common.sh"
TASK_DIR=$DATA/tasks/smoke
EXP=SMOKE
N_TASKS=2
ARGS=("${COMMON_ARGS[@]}" --task_ann_dir "$TASK_DIR" --epochs 1)
