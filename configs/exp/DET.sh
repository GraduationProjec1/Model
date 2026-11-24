source "$REPO/configs/exp/common.sh"
EXP=DET
SKIP_EVAL=1
N_TASKS=1
ARGS=("${COMMON_ARGS[@]}" "${FINETUNE_ARGS[@]}" "${STANDARD_ARGS[@]}"
    --task_config "$REPO/configs/tasks_agnostic.json" --n_classes 2
    --task_ann_dir "$DATA/tasks/agnostic_task1" --pred_ann_dir "$TASK_DIR")
ARGS+=(--tf32 0)
