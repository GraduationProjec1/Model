source "$REPO/configs/exp/common.sh"
TASK_DIR=$DATA/tasks/pilot_100-4x25_seed0
EXP=FSA_pilot
N_TASKS=1
ARGS=("${COMMON_ARGS[@]}" "${FINETUNE_ARGS[@]}" --task_ann_dir "$TASK_DIR" --epochs 4 --save_hf 1)
ARGS+=(--eff_batch_size 32 --tf32 0 --pseudo_dedup_iou 0)
