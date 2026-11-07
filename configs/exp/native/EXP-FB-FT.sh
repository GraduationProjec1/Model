# Update model, mode improve (docs/project/feedback-finetune-plan-2026-11-01.md): task 5 of the round-1 config trained
# again from the round-1 task-5 model (PREV_CKPT, added as --prev_ckpt by common.sh) on train_task_5_fix.json, a file
# with every object labelled (feedback images, synthetic images of the confused SKUs, replay). Only the classifier
# rows of TRAIN_ROWS train (F1/F2); no teacher or pseudo-labels (F4); the private prompts of task 5 are kept (F3);
# query_tf and the shared prompts move to lr_old (F5); low lr with a step down (F6). Set by autocheckout/update_model.py.
EXP=${EXP:-EXP-FB-FT}
source "$REPO/configs/exp/native/EXP-FULL-SYN.sh"
: "${PREV_CKPT:?PREV_CKPT must point to the task_final.pth to improve}"
ARGS+=(--pseudo none --init_new_prompts 0 --new_params class_embed,prompts.private
    --lr "${LR:-2e-5}" --lr_old "${LR_OLD:-5e-6}" --lr_drop "${LR_DROP:-3}"
    --train_rows "${TRAIN_ROWS:?TRAIN_ROWS must point to the JSON list of labels to retrain}"
    --train_suffix _fix
    --augment "${AUGMENT:-0}" --tune_old_private "${TUNE_OLD_PRIVATE:-0}")  # round 3 (design 2026-11-07-vong-3); off = as before
