source "$REPO/configs/exp/common.sh"
EXP=E2
N_TASKS=5
ARGS=("${COMMON_ARGS[@]}" "${STANDARD_ARGS[@]}" "${FSA_ARGS[@]}" --use_shared 0 --pseudo threshold --pseudo_thresh_high 0.65
    --pseudo_topk 5)
