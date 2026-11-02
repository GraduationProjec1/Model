"""A5: cAcc of a task-5 model on Val-B with counting thresholds chosen on Val-A-calib (never on Val-B, never on an
image any AL selection may train on). One threshold per task group (``select_group_thresholds``), NMS 0.45, as the
FULL-SYN policies. Writes ``<release>/active/scored/<name>.json``.

    python -m tools.al_score --release ../relearn_fix_v2 --name baseline_r1 \\
        --pred ../relearn_fix_v2/seed/full-syn-r1/runs/EXP-FULL-SYN/task_5/pred_val.npz
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
from autocheckout.feedback_fix import subset
from autocheckout.io import load_json, md5_file, save_json
from autocheckout.predictions import Predictions, load_predictions
from autocheckout.taskcfg import TaskConfig
from tools.eval_count import CALIBRATION_GRID

STAGE, NMS = 5, 0.45


def score(val_full: dict, cfg: TaskConfig, preds: Predictions, calib: list[int], target: list[int]) -> dict:
    """Thresholds chosen on ``calib`` images only, then applied to ``target`` images."""
    groups = [list(cfg.task(t).labels) for t in range(1, STAGE + 1)]
    labels = sorted(label for g in groups for label in g)
    processed = dedup_detections(preds.subset(np.isin(preds.image_id, calib + target)), NMS)
    vector, calib_scores = select_group_thresholds(load_coco(subset(val_full, calib)),
                                                   processed.subset(np.isin(processed.image_id, calib)),
                                                   groups, CALIBRATION_GRID)
    ids = sorted(target)
    gt = gt_counts(load_coco(subset(val_full, ids)), ids, labels)
    pred = count_matrix(processed, ids, labels, label_thresholds(groups, vector))
    right = np.all(pred == gt, axis=1)
    return {'thresholds': {f'task_{t}': v for t, v in zip(range(1, STAGE + 1), vector, strict=True)},
            'nms_iou': NMS, 'calib_images': len(calib), 'calib_cAcc': calib_scores['cAcc'],
            'target_images': len(ids), 'cAcc_m2': float(right.mean()), **{k: v for k, v in counting_scores(pred, gt)
                                                                          .items() if k in ('ACD', 'mCCD', 'mCIoU')},
            'right_images': [int(i) for i, ok in zip(ids, right, strict=True) if ok]}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--release', type=Path, required=True)
    ap.add_argument('--pred', type=Path, required=True, help='task_5/pred_val.npz of the model (all of val)')
    ap.add_argument('--name', required=True)
    args = ap.parse_args(argv)
    rel = args.release
    val_full = load_json(rel / 'eval/val_full.json')
    preds = load_predictions(args.pred)
    if preds.meta.get('ann_md5') not in (None, md5_file(rel / 'eval/val_full.json')):
        raise ValueError(f'{args.pred} was not predicted on eval/val_full.json')
    calib = load_json(rel / 'active/splits/val_a_split.json')['calib']
    val_b = sorted(i['id'] for i in load_json(rel / 'meta/val_b.json')['images'])
    result = {'name': args.name, 'pred': str(args.pred), 'pred_md5': md5_file(args.pred),
              **score(val_full, TaskConfig.load(rel / 'eval/task_config.json'), preds, calib, val_b)}
    save_json(rel / f'active/scored/{args.name}.json', result, indent=1)
    return result


if __name__ == '__main__':
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k != 'right_images'}, indent=1))
