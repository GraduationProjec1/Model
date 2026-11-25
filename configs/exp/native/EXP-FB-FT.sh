EXP=${EXP:-EXP-FB-FT}
source "$REPO/configs/exp/native/EXP-FULL-SYN.sh"
: "${PREV_CKPT:?PREV_CKPT must point to the task_final.pth to improve}"
ARGS+=(--pseudo none --init_new_prompts 0 --new_params "class_embed,prompts.private${NEW_PARAMS_EXTRA:+,$NEW_PARAMS_EXTRA}"
    --lr "${LR:-2e-5}" --lr_old "${LR_OLD:-5e-6}" --lr_drop "${LR_DROP:-3}"
    --train_rows "${TRAIN_ROWS:?TRAIN_ROWS must point to the JSON list of labels to retrain}"
    --train_suffix _fix
    --augment "${AUGMENT:-0}" --tune_old_private "${TUNE_OLD_PRIVATE:-0}"
    --seed "${SEED:-0}")
