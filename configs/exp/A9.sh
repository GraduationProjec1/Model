source "$REPO/configs/exp/common.sh"
EXP=A9
N_TASKS=3
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" "${E4_ARGS[@]}" --ddl_lambda 0)
