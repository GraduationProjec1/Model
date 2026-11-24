source "$REPO/configs/exp/common.sh"
EXP=A4
N_TASKS=3
REUSE_TASK1=E4
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" "${E4_ARGS[@]}" --pseudo_topk 5)
