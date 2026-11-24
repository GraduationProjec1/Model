EXP=${EXP:-EXP-FULL-SYN}
source "$REPO/configs/exp/native/full_syn.sh"
ARGS+=(--repo_name "${FSA_HF:?FSA_HF must point to the hf_model of the FSA run}"
    --pseudo_dedup_iou 0.5 --pseudo_gt_iou 0.5)
