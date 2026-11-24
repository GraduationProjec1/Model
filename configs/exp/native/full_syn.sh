source "$REPO/configs/exp/native/EXP-B2.sh"
ARGS+=(--train_img_dir "${SYNTH_ROOT:?SYNTH_ROOT must point to the synthetic release}")
