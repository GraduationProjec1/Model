# FULL-SYN lower bound (docs/al/PLAN.md, B2): EXP-FULL-SYN without its two ways of keeping old SKUs, as E1 was for
# the September release: no private prompt pool (one shared pool only) and no pseudo-labels of old SKUs. Tasks 2-5
# start from task 1 of FULL-SYN r1 (full_syn.stage_seed), so the two runs differ only in these switches.
EXP=${EXP:-EXP-FULL-SYN-NAIVE}
source "$REPO/configs/exp/native/EXP-FULL-SYN.sh"
ARGS+=(--use_private 0 --pseudo none)
