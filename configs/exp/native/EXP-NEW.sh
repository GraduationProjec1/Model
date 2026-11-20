# Update model, mode new (docs/project/feedback-finetune-plan-2026-11-01.md Phần 2, N2-N5; autocheckout/new_skus.py):
# one task of products the model does not know, trained the PDP way on the task-5 model expanded to new class slots.
# Only the classifier rows of the new task and its private prompts train at LR (N2, N5); old rows keep zero gradient
# (engine.py), shared parts at LR_OLD, backbone/encoder/decoder frozen (EXP-FULL-SYN). No teacher and no pseudo-labels:
# the synthetic images of the new products hold no old product, and the task-5 model labels the new products as old
# ones (decision of 20/11/2026, replaces N3). New prompt slots initialised (N4). I2 augmentation without flips (text on
# the packs). Set by autocheckout/new_skus.py, which adds --n_classes of the new config.
EXP=${EXP:-EXP-NEW}
source "$REPO/configs/exp/native/EXP-FULL-SYN.sh"
ARGS+=(--pseudo none --init_new_prompts 1 --new_params class_embed,prompts.private
    --lr "${LR:-1e-4}" --lr_old "${LR_OLD:-1e-5}" --lr_drop "${LR_DROP:-9}"
    --augment "${AUGMENT:-1}" --augment_flip 0
    --train_suffix "${TRAIN_SUFFIX:-}")  # _pilot: the pilot's subset of train_task_<t>.json
