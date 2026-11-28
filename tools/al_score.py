"""A5: cAcc of a model on Val-B with counting thresholds chosen on Val-A-calib (never on Val-B, never on an image any
AL selection may train on). One threshold per task group (``select_group_thresholds``), NMS 0.45, as the FULL-SYN
policies. Writes ``<data>/relearn/active/scored/<name>.json``.

    python -m tools.al_score --name baseline_r1 \\
        --pred ../autocheckout-data/models/full-syn-r1/runs/EXP-FULL-SYN/task_5/pred_val.npz

Any stage of any task config (chia 12+13 SKU, docs/design/2026-11-09-chia-12-13-sku, plan 3.5): ``--stage`` (default
5), ``--task-config`` and ``--eval-dir`` (default: the data folder's eval/), ``--groups task`` (one threshold per task
of the config) or ``--groups r1`` (one per task of the r1 config ``--ref-config``, so a 9-task model gets as many
thresholds as r1), ``--target pool`` (Val-A pool instead of Val-B), ``--grid-min`` (lower end of the threshold grid,
default 0.10) and ``--out``. Before the last task the score is cAcc-M2: images without a learned object are dropped
and so are the boxes on objects of classes not learned yet (``counting.drop_unlearned``, IoU ``--unlearned-iou``).

    python -m tools.al_score --name 12x_t9_r1 --stage 9 --groups r1 \\
        --task-config ../autocheckout-data/eval_12x/task_config.json --eval-dir ../autocheckout-data/eval_12x \\
        --pred data/training/full-syn-12x/runs/EXP-FULL-SYN/task_9/pred_val.npz --out results/split12/12x_t9_r1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autocheckout.cl_metrics import load_coco
from autocheckout.counting import (
    count_matrix,
    counting_scores,
    dedup_detections,
    drop_unlearned,
    gt_counts,
    label_thresholds,
    select_group_thresholds,
    select_group_thresholds_robust,
)
from autocheckout.datapaths import Layout
from autocheckout.feedback_fix import subset
from autocheckout.io import load_json, md5_file, save_json
from autocheckout.predictions import Predictions, load_predictions
from autocheckout.taskcfg import TaskConfig
from tools.eval_count import CALIBRATION_GRID

STAGE, NMS = 5, 0.45
UNLEARNED_IOU = 0.5


def group_labels(cfg: TaskConfig, stage: int = STAGE, by: str = 'task',
                 ref_cfg: TaskConfig | None = None) -> tuple[list[list[int]], list[str]]:
    """(label groups, their names) of the classes learned up to ``stage``: one group per task of ``cfg``
    (``task_<t>``), or (``by='r1'``) one per task of ``ref_cfg`` that has learned classes, matched by RPC id
    (``r1_task_<t>``)."""
    if by == 'task':
        return [list(cfg.task(t).labels) for t in range(1, stage + 1)], [f'task_{t}' for t in range(1, stage + 1)]
    if by != 'r1' or ref_cfg is None:
        raise ValueError("groups must be 'task', or 'r1' with the r1 task config")
    to_rpc, seen = cfg.label_to_rpc(), range(cfg.seen_classes(stage))
    groups, names = [], []
    for task in ref_cfg.data_tasks:
        rpc = {c.rpc_category_id for c in task.classes}
        labels = [label for label in seen if to_rpc.get(label) in rpc]
        if labels:
            groups.append(labels)
            names.append(f'r1_task_{task.task_id}')
    if sum(len(g) for g in groups) != len(seen):
        raise ValueError(f'{cfg.name}: learned classes missing from {ref_cfg.name}')
    return groups, names


def _groups(cfg: TaskConfig) -> tuple[list[list[int]], list[int]]:
    groups, _ = group_labels(cfg)
    return groups, sorted(label for g in groups for label in g)


def threshold_grid(minimum: float | None = None) -> tuple[float, ...]:
    """CALIBRATION_GRID (0.10-0.90 by 0.01), or the same steps from ``minimum`` (R1: r1's task-1 group hit 0.10)."""
    if minimum is None:
        return CALIBRATION_GRID
    if not 0 < minimum <= CALIBRATION_GRID[0]:
        raise ValueError(f'grid minimum must be in (0, {CALIBRATION_GRID[0]}]')
    return tuple(round(minimum + 0.01 * i, 2) for i in range(int(round((0.90 - minimum) / 0.01)) + 1))


def _view(coco_full: dict, preds: Predictions, ids: list[int], seen: int, unlearned_iou: float | None):
    """(COCO of the scored images, top-1 rows on them): every image of ``ids``, or with ``unlearned_iou`` the
    cAcc-M2 view (images with a learned object, without the rows on unlearned objects)."""
    coco = load_coco(subset(coco_full, ids))
    rows = preds.subset(np.isin(preds.image_id, ids))
    if unlearned_iou:
        return drop_unlearned(coco, rows, seen, unlearned_iou)
    return coco, rows


def count_on(coco_full: dict, cfg: TaskConfig, preds: Predictions, ids: list[int], thresholds: list[float], *,
             groups: list[list[int]] | None = None, unlearned_iou: float | None = None) -> dict:
    """cAcc-M2 (and ACD, mCCD, mCIoU) of ``preds`` on the images ``ids`` of ``coco_full`` with fixed thresholds."""
    groups = groups or _groups(cfg)[0]
    labels = sorted(label for g in groups for label in g)
    coco, rows = _view(coco_full, preds, sorted(ids), len(labels), unlearned_iou)
    ids = sorted(coco.getImgIds())
    processed = dedup_detections(rows, NMS)
    gt = gt_counts(coco, ids, labels)
    pred = count_matrix(processed, ids, labels, label_thresholds(groups, thresholds))
    right = np.all(pred == gt, axis=1)
    return {'target_images': len(ids), 'cAcc_m2': float(right.mean()) if len(ids) else float('nan'),
            **{k: v for k, v in counting_scores(pred, gt).items() if k in ('ACD', 'mCCD', 'mCIoU')},
            'right_images': [int(i) for i, ok in zip(ids, right, strict=True) if ok]}


def score(val_full: dict, cfg: TaskConfig, preds: Predictions, calib: list[int], target: list[int], *,
          groups: tuple[list[list[int]], list[str]] | None = None, grid: tuple[float, ...] = CALIBRATION_GRID,
          unlearned_iou: float | None = None, robust: bool = False) -> dict:
    """Thresholds chosen on ``calib`` images only, then applied to ``target`` images."""
    label_groups, names = groups or group_labels(cfg)
    seen = sum(len(g) for g in label_groups)
    coco, rows = _view(val_full, preds, sorted(calib), seen, unlearned_iou)
    deduped = dedup_detections(rows, NMS)
    if robust:
        vector, calib_scores = select_group_thresholds_robust(coco, deduped, label_groups, grid)
    else:
        vector, calib_scores = select_group_thresholds(coco, deduped, label_groups, grid)
    result = {'thresholds': dict(zip(names, vector, strict=True)), 'nms_iou': NMS,
              'calib_images': len(coco.getImgIds()), 'calib_cAcc': calib_scores['cAcc'],
              **count_on(val_full, cfg, preds, target, vector, groups=label_groups, unlearned_iou=unlearned_iou)}
    if robust:
        result['robust_calib'] = True
    if grid != CALIBRATION_GRID or unlearned_iou:
        result.update(grid=[grid[0], grid[-1]], unlearned_iou=unlearned_iou,
                      at_floor=[n for n, v in zip(names, vector, strict=True) if v <= grid[0]])
    return result


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', type=Path, help='the data folder (default: ../autocheckout-data or $AC_DATA)')
    ap.add_argument('--pred', type=Path, required=True, help='task_<stage>/pred_val.npz of the model (all of val)')
    ap.add_argument('--name', required=True)
    ap.add_argument('--stage', type=int, default=STAGE, help='task the model has learned up to (default 5)')
    ap.add_argument('--task-config', type=Path, help="default: the eval dir's task_config.json")
    ap.add_argument('--eval-dir', type=Path, help='val_full.json / test_full.json (default: <data>/eval)')
    ap.add_argument('--groups', choices=('task', 'r1'), default='task')
    ap.add_argument('--ref-config', type=Path, help='r1 task config of --groups r1 (default: <data>/eval)')
    ap.add_argument('--target', choices=('valb', 'pool'), default='valb', help='Val-B, or the Val-A pool')
    ap.add_argument('--grid-min', type=float, help='lower end of the threshold grid (default 0.10)')
    ap.add_argument('--robust-calib', action='store_true',
                    help='use 5-fold midpoint/ACD threshold selection (min 0.18) to avoid calib overfitting')
    ap.add_argument('--unlearned-iou', type=float, default=UNLEARNED_IOU,
                    help='cAcc-M2 IoU, applied before the last task only')
    ap.add_argument('--out', type=Path, help='default: <data>/relearn/active/scored/<name>.json')
    ap.add_argument('--test-pred', type=Path, help='task_<stage>/pred_test.npz: also score test2019 (once) with the '
                                                   'same thresholds, into active/scored_test/<name>.json')
    args = ap.parse_args(argv)
    data = Layout.at(args.data)
    rel = data.relearn
    eval_dir = args.eval_dir or data.eval
    val_full = load_json(eval_dir / 'val_full.json')
    preds = load_predictions(args.pred)
    if preds.meta.get('ann_md5') not in (None, md5_file(eval_dir / 'val_full.json')):
        raise ValueError(f'{args.pred} was not predicted on {eval_dir / "val_full.json"}')
    cfg = TaskConfig.load(args.task_config or eval_dir / 'task_config.json')
    if not 1 <= args.stage <= len(cfg.data_tasks):
        raise ValueError(f'stage must be 1..{len(cfg.data_tasks)}')
    ref = TaskConfig.load(args.ref_config or data.eval / 'task_config.json') if args.groups == 'r1' else None
    groups = group_labels(cfg, args.stage, args.groups, ref)
    unlearned = args.unlearned_iou if args.stage < len(cfg.data_tasks) else None
    split = load_json(rel / 'active/splits/val_a_split.json')
    calib = split['calib']
    target = split['pool'] if args.target == 'pool' else sorted(i['id'] for i in load_json(rel / 'meta/val_b.json')['images'])
    result = {'name': args.name, 'pred': str(args.pred), 'pred_md5': md5_file(args.pred),
              **score(val_full, cfg, preds, calib, target, groups=groups, grid=threshold_grid(args.grid_min),
                      unlearned_iou=unlearned, robust=args.robust_calib)}
    if (args.stage, args.groups, args.target, args.task_config, args.eval_dir) != (STAGE, 'task', 'valb', None, None):
        result.update(stage=args.stage, task_config=cfg.name, groups=args.groups, target=args.target)
    save_json(args.out or rel / f'active/scored/{args.name}.json', result, indent=1)
    if args.test_pred:
        out = rel / f'active/scored_test/{args.name}.json'
        if out.is_file():
            raise FileExistsError(f'{out}: test2019 is scored once')
        test_full = load_json(eval_dir / 'test_full.json')
        test = load_predictions(args.test_pred)
        if test.meta.get('ann_md5') not in (None, md5_file(eval_dir / 'test_full.json')):
            raise ValueError(f'{args.test_pred} was not predicted on {eval_dir / "test_full.json"}')
        thresholds = list(result['thresholds'].values())
        save_json(out, {'name': args.name, 'pred': str(args.test_pred), 'thresholds': result['thresholds'],
                        **count_on(test_full, cfg, test, [i['id'] for i in test_full['images']], thresholds,
                                   groups=groups[0], unlearned_iou=unlearned)}, indent=1)
        result['test2019'] = {k: v for k, v in load_json(out).items() if k != 'right_images'}
    return result


if __name__ == '__main__':
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k != 'right_images'}, indent=1))
