EXP=${EXP:-EXP-FULL-SYN-M5}
source "$REPO/configs/exp/native/EXP-FULL-SYN.sh"
ARGS+=(--tune_after_task1 stages_3,encoder --feature_kd 1)
