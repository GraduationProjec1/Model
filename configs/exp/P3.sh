source "$REPO/configs/exp/common.sh"
TASK_DIR=$DATA/tasks/pilot_100-4x25_seed0
EXP=P3
N_TASKS=2
ARGS=("${COMMON_ARGS[@]}" --task_ann_dir "$TASK_DIR" --epochs 4 --repo_name "$RUNS/FSA_pilot/task_1/hf_model")
ARGS+=(--eff_batch_size 32 --tf32 0 --pseudo_dedup_iou 0)
