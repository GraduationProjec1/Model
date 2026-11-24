#!/usr/bin/env bash
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUNS=${RUNS:-/data/runs}
DATA=${DATA:-/data/rpc}
LOG=$RUNS/pilot_chain.log
trap 'echo "pilot chain finished $(date "+%F %T")" >> "$LOG"; ${SHUTDOWN_CMD:-sudo shutdown -h now}' EXIT
source "$HOME/venvs/pdp/bin/activate"
cd "$REPO"
for exp in P2 P1 FSA_pilot P3; do
    bash scripts/run_exp.sh "configs/exp/$exp.sh" > "$RUNS/$exp.log" 2>&1
    echo "$exp exit=$? $(date "+%F %T")" >> "$LOG"
done
eval "ARGS=($(REPO=$REPO DATA=$DATA RUNS=$RUNS bash -c 'source configs/exp/P2.sh; printf "%q " "${ARGS[@]}"'))"
(cd pdp && python ppg_audit.py "${ARGS[@]}" --output_dir "$RUNS/P2" --audit_task 2 --audit_images 1000) \
    > "$RUNS/P2_ppg_audit.log" 2>&1
echo "V4 exit=$? $(date "+%F %T")" >> "$LOG"
