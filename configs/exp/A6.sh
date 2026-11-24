source "$REPO/configs/exp/common.sh"
EXP=A6
N_TASKS=3
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" "${E4_ARGS[@]}" --augment 0)
