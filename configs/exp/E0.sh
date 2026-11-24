source "$REPO/configs/exp/common.sh"
EXP=E0
START_TASK=5
N_TASKS=5
ARGS=("${COMMON_ARGS[@]}" "${FINETUNE_ARGS[@]}" --joint 1 --train_suffix _capped --epochs 12)
