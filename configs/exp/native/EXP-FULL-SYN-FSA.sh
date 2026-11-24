EXP=${EXP:-EXP-FULL-SYN-FSA}
source "$REPO/configs/exp/native/full_syn.sh"
ARGS+=(--use_prompts 0 --local_query 0 --pseudo none --freeze "" --optim_groups detr --lr_backbone_names backbone
    --save_hf 1 --select_best_epoch 1)
