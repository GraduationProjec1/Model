"""A5: cAcc of a task-5 model on Val-B with counting thresholds chosen on Val-A-calib (never on Val-B, never on an
image any AL selection may train on). One threshold per task group (``select_group_thresholds``), NMS 0.45, as the
FULL-SYN policies. Writes ``<data>/relearn/active/scored/<name>.json``.

    python -m tools.al_score --name baseline_r1 \\
        --pred ../autocheckout-data/models/full-syn-r1/runs/EXP-FULL-SYN/task_5/pred_val.npz
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
    gt_counts,
    label_thresholds,
    select_group_thresholds,
)
from autocheckout.datapaths import Layout
from autocheckout.feedback_fix import subset
from autocheckout.io import load_json, md5_file, save_json
from autocheckout.predictions import Predictions, load_predictions
from autocheckout.taskcfg import TaskConfig
from tools.eval_count import CALIBRATION_GRID

STAGE, NMS = 5, 0.45


def _groups(cfg: TaskConfig) -> tuple[list[list[int]], list[int]]:
    groups = [list(cfg.task(t).labels) for t in range(1, STAGE + 1)]
    return groups, sorted(label for g in groups for label in g)


def count_on(coco_full: dict, cfg: TaskConfig, preds: Predictions, ids: list[int], thresholds: list[float]) -> dict:
    """cAcc-M2 (and ACD, mCCD, mCIoU) of ``preds`` on the images ``ids`` of ``coco_full`` with fixed thresholds."""
    groups, labels = _groups(cfg)
    ids = sorted(ids)
    processed = dedup_detections(preds.subset(np.isin(preds.image_id, ids)), NMS)
    gt = gt_counts(load_coco(subset(coco_full, ids)), ids, labels)
    pred = count_matrix(processed, ids, labels, label_thresholds(groups, thresholds))
    right = np.all(pred == gt, axis=1)
    return {'target_images': len(ids), 'cAcc_m2': float(right.mean()),
            **{k: v for k, v in counting_scores(pred, gt).items() if k in ('ACD', 'mCCD', 'mCIoU')},
            'right_images': [int(i) for i, ok in zip(ids, right, strict=True) if ok]}


def score(val_full: dict, cfg: TaskConfig, preds: Predictions, calib: list[int], target: list[int]) -> dict:
    """Thresholds chosen on ``calib`` images only, then applied to ``target`` images."""
    groups, _ = _groups(cfg)
    processed = dedup_detections(preds.subset(np.isin(preds.image_id, calib)), NMS)
    vector, calib_scores = select_group_thresholds(load_coco(subset(val_full, calib)), processed, groups,
                                                   CALIBRATION_GRID)
    return {'thresholds': {f'task_{t}': v for t, v in zip(range(1, STAGE + 1), vector, strict=True)},
            'nms_iou': NMS, 'calib_images': len(calib), 'calib_cAcc': calib_scores['cAcc'],
            **count_on(val_full, cfg, preds, target, vector)}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', type=Path, help='the data folder (default: ../autocheckout-data or $AC_DATA)')
    ap.add_argument('--pred', type=Path, required=True, help='task_5/pred_val.npz of the model (all of val)')
    ap.add_argument('--name', required=True)
    ap.add_argument('--test-pred', type=Path, help='task_5/pred_test.npz: also score test2019 (once) with the same '
                                                   'thresholds, into active/scored_test/<name>.json')
    args = ap.parse_args(argv)
    data = Layout.at(args.data)
    rel = data.relearn
    val_full = load_json(data.eval / 'val_full.json')
    preds = load_predictions(args.pred)
    if preds.meta.get('ann_md5') not in (None, md5_file(data.eval / 'val_full.json')):
        raise ValueError(f'{args.pred} was not predicted on eval/val_full.json')
    cfg = TaskConfig.load(data.eval / 'task_config.json')
    calib = load_json(rel / 'active/splits/val_a_split.json')['calib']
    val_b = sorted(i['id'] for i in load_json(rel / 'meta/val_b.json')['images'])
    result = {'name': args.name, 'pred': str(args.pred), 'pred_md5': md5_file(args.pred),
              **score(val_full, cfg, preds, calib, val_b)}
    save_json(rel / f'active/scored/{args.name}.json', result, indent=1)
    if args.test_pred:
        out = rel / f'active/scored_test/{args.name}.json'
        if out.is_file():
            raise FileExistsError(f'{out}: test2019 is scored once')
        test_full = load_json(data.eval / 'test_full.json')
        test = load_predictions(args.test_pred)
        if test.meta.get('ann_md5') not in (None, md5_file(data.eval / 'test_full.json')):
            raise ValueError(f'{args.test_pred} was not predicted on eval/test_full.json')
        thresholds = [result['thresholds'][f'task_{t}'] for t in range(1, STAGE + 1)]
        save_json(out, {'name': args.name, 'pred': str(args.test_pred), 'thresholds': result['thresholds'],
                        **count_on(test_full, cfg, test, [i['id'] for i in test_full['images']], thresholds)},
                  indent=1)
        result['test2019'] = {k: v for k, v in load_json(out).items() if k != 'right_images'}
    return result


if __name__ == '__main__':
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k != 'right_images'}, indent=1))
