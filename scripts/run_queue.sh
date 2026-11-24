#!/usr/bin/env bash
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUNS=${RUNS:-/data/runs}
CONFIG_DIR=${CONFIG_DIR:-$REPO/configs/exp}
QUEUE=$RUNS/queue.txt
LOG=$RUNS/queue.log
trap 'echo "queue stopped $(date "+%F %T")" >> "$LOG"; ${SHUTDOWN_CMD:-sudo shutdown -h now}' EXIT
if [[ -f $HOME/venvs/pdp/bin/activate ]]; then
    source "$HOME/venvs/pdp/bin/activate"
fi
cd "$REPO"
touch "$LOG"

next_experiment() {
    local name
    [[ -f $QUEUE ]] || return 0
    while read -r name _; do
        [[ -z $name || $name == \#* ]] && continue
        grep -q "^$name exit=" "$LOG" || { echo "$name"; return 0; }
    done < "$QUEUE"
}

while exp=$(next_experiment) && [[ -n $exp ]]; do
    bash scripts/run_exp.sh "$CONFIG_DIR/$exp.sh" >> "$RUNS/$exp.log" 2>&1
    status=$?
    if (( status > 128 )); then
        echo "$exp interrupted (status $status) $(date "+%F %T")" >> "$LOG"
        exit "$status"
    fi
    echo "$exp exit=$status $(date "+%F %T")" >> "$LOG"
done
echo "queue empty $(date "+%F %T")" >> "$LOG"
