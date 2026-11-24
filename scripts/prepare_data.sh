#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-/data/rpc}
RAW=$DATA/raw/retail_product_checkout
TASKS=100-4x25_seed0

echo "== DL1: audit"
python -m tools.audit_rpc --raw "$RAW" --out results/data_audit --md5

echo "== DL2: resize to 800x800 + merged annotation"
python -m tools.resize --raw "$RAW" --out-dir "$DATA/checkout_800" --ann-out "$DATA/ann/checkout_800.json" \
    --draw 20 --draw-dir "$DATA/draw_check"

echo "== DL3: group split"
python -m tools.make_split --ann "$DATA/ann/checkout_800.json" --out-dir "$DATA/splits" --config-dir configs/splits \
    --stratify none

echo "== DL4: task config"
python -m tools.make_task_config --categories "$RAW/instances_test2019.json" --name "$TASKS" \
    --sizes 100,25,25,25,25 --reserved 24 --seed 0

echo "== DL5: per-task files"
python -m tools.make_task_json --task-config "configs/tasks_$TASKS.json" \
    --train-source "real=$DATA/splits/train.json" --val "$DATA/splits/val.json" --test "$DATA/splits/test.json" \
    --out-dir "$DATA/tasks/$TASKS" --joint --agnostic-out "$DATA/tasks/agnostic_task1"

echo "== DL6: pilot (train_pilot, tasks 1-2)"
python -m tools.make_task_json --task-config "configs/tasks_$TASKS.json" \
    --train-source "real=$DATA/splits/train_pilot.json" --val "$DATA/splits/val.json" --test "$DATA/splits/test.json" \
    --out-dir "$DATA/tasks/pilot_$TASKS" --tasks 1,2
