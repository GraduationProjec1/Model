source "$REPO/configs/exp/common.sh"
EXP=FSA
N_TASKS=1
ARGS=("${COMMON_ARGS[@]}" "${FINETUNE_ARGS[@]}" "${STANDARD_ARGS[@]}" --save_hf 1)
ARGS+=(--tf32 0)
