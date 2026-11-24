source "$REPO/configs/exp/common.sh"
EXP=A5
N_TASKS=3
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" "${E4_ARGS[@]}" --repo_name SenseTime/deformable-detr)
