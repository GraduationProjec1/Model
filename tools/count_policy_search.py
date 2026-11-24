"""B1: counting policy search (NMS and per-SKU thresholds) chosen on Val-A-calib only, scored on Val-B once.

Policies, all built from the same task-5 ``pred_val.npz`` (no training):

  base          NMS 0.45, 5 group thresholds        (= tools/al_score.py, the 25 existing runs)
  nms=<x>       other NMS value (or 'off'), 5 group thresholds re-selected on calib
  nms=<x>+sku   the NMS above, plus own thresholds for at most ``--max-sku`` weak SKUs. A SKU threshold is taken
                only when calib right images rise by >= ``--min-gain`` images (anti over-fitting on 581 images).

Candidate SKUs are the SKUs with >= ``--min-objects`` GT objects in calib, ranked by calib count error under the
group policy (not by Val-B). Val-B is only used to score the final policies and for the sign test against ``base``.

    python -m tools.count_policy_search --name al_margin_k_B500_r2 --val inputs/val_b/val_full.json --val-a inputs/val_a.json \\
        --pred runs/<run>/task_5/pred_val.npz --calib relearn/active/splits/val_a_split.json \\
        --task-config configs/relearn/task_config_relearn.json --out results/count_policy_<name>.json
"""

from __future__ import annotations

import argparse
import json
from math import comb
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
from autocheckout.io import load_json, save_json
from autocheckout.predictions import load_predictions
from autocheckout.taskcfg import TaskConfig
from tools.eval_count import CALIBRATION_GRID

STAGE = 5
NMS_GRID: tuple[float | None, ...] = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, None)


def sign_test(fixed: int, broken: int) -> float:
    """Two-sided exact sign test p-value."""
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


class Scorer:
    """Counts a fixed set of images with a per-label threshold vector."""

    def __init__(self, coco_full: dict, ids: list[int], labels: list[int], top1):
        self.ids, self.labels, self.top1 = sorted(ids), labels, top1
        self.gt = gt_counts(load_coco(subset(coco_full, self.ids)), self.ids, labels)

    def counts(self, vector: np.ndarray) -> np.ndarray:
        return count_matrix(self.top1, self.ids, self.labels, vector)

    def right(self, vector: np.ndarray) -> np.ndarray:
        return np.all(self.counts(vector) == self.gt, axis=1)

    def recall(self, vector: np.ndarray, label: int) -> float:
        """Share of the GT objects of ``label`` still counted (per image, at most the GT count)."""
        c = self.labels.index(label)
        gt = self.gt[:, c]
        return float(np.minimum(self.counts(vector)[:, c], gt).sum() / max(1, gt.sum()))

    def scores(self, vector: np.ndarray) -> dict:
        return counting_scores(count_matrix(self.top1, self.ids, self.labels, vector), self.gt)


def candidate_skus(calib: Scorer, vector: np.ndarray, min_objects: int, top: int) -> list[int]:
    pred = count_matrix(calib.top1, calib.ids, calib.labels, vector)
    err = np.abs(pred - calib.gt).sum(axis=0)
    ok = [(-int(err[c]), calib.labels[c]) for c in range(len(calib.labels))
          if calib.gt[:, c].sum() >= min_objects and err[c] > 0]
    return [label for _, label in sorted(ok)[:top]]


def add_sku_thresholds(calib: Scorer, vector: np.ndarray, skus: list[int], grid, min_gain: int,
                       max_passes: int = 3,
                       max_recall_drop: float | None = None) -> tuple[np.ndarray, dict[int, float]]:
    """Coordinate ascent over the SKUs: a SKU takes the grid value with the most right calib images, only when it
    beats its current value by >= ``min_gain`` images (ties: the value nearest the group threshold).
    ``max_recall_drop`` (round 3, B1): a value is allowed only when the SKU's recall on the selection images stays
    within that drop of its recall under the group policy ``vector`` (None: no limit, the original search)."""
    vec, own = vector.copy(), {}
    floor = {label: calib.recall(vector, label) - max_recall_drop for label in skus} \
        if max_recall_drop is not None else {}
    for _ in range(max_passes):
        changed = False
        for label in skus:
            current = int(calib.right(vec).sum())
            keep = vec[label]
            best_t, best_n = keep, current
            for t in grid:
                vec[label] = t
                if label in floor and calib.recall(vec, label) < floor[label] - 1e-12:
                    continue
                n = int(calib.right(vec).sum())
                if n > best_n or (n == best_n and abs(t - keep) < abs(best_t - keep) and best_t != keep):
                    best_t, best_n = t, n
            if best_n - current >= min_gain and best_t != keep:
                vec[label], changed = best_t, True
                own[label] = float(best_t)
            else:
                vec[label] = keep
        if not changed:
            break
    return vec, own


def run(val_full: dict, cfg: TaskConfig, preds, calib_ids: list[int], target_ids: list[int], *, nms_grid=NMS_GRID,
        grid=CALIBRATION_GRID, max_sku: int = 20, min_objects: int = 30, min_gain: int = 2,
        select_ids: list[int] | None = None, max_recall_drop: float | None = None) -> dict:
    """``select_ids``: the images every choice is made on (default ``calib_ids``). ``base`` is always the calib-only
    policy of tools/al_score.py, so the numbers stay comparable with the existing runs."""
    select_ids = select_ids or calib_ids
    groups = [list(cfg.task(t).labels) for t in range(1, STAGE + 1)]
    labels = sorted(label for g in groups for label in g)
    out: dict = {'policies': {}}
    base_right = None
    if select_ids != calib_ids:
        top1 = dedup_detections(preds, 0.45)
        base_thr, _ = select_group_thresholds(load_coco(subset(val_full, sorted(calib_ids))),
                                              top1.subset(np.isin(top1.image_id, calib_ids)), groups, grid)
        base_right = Scorer(val_full, target_ids, labels, top1).right(label_thresholds(groups, base_thr))
    for nms in nms_grid:
        top1 = dedup_detections(preds, nms) if nms is not None else preds.top1_per_query()
        calib = Scorer(val_full, select_ids, labels, top1)
        target = Scorer(val_full, target_ids, labels, top1)
        calib_coco = load_coco(subset(val_full, calib.ids))
        calib_top1 = top1.subset(np.isin(top1.image_id, calib.ids))
        thr, calib_scores = select_group_thresholds(calib_coco, calib_top1, groups, grid)
        vec = label_thresholds(groups, thr)
        variants = {'': (vec, {})}
        if max_sku:
            skus = candidate_skus(calib, vec, min_objects, max_sku)
            variants['+sku'] = add_sku_thresholds(calib, vec, skus, grid, min_gain, max_recall_drop=max_recall_drop)
        for suffix, (v, own) in variants.items():
            name = f"nms={'off' if nms is None else nms}{suffix}"
            right = target.right(v)
            sc = target.scores(v)
            if nms == 0.45 and not suffix and base_right is None:
                base_right, name = right, 'base'
            out['policies'][name] = {
                'nms': nms, 'group_thresholds': thr, 'sku_thresholds': {str(k): x for k, x in own.items()},
                'label_thresholds': [float(x) for x in v],
                'calib_cAcc': float(calib.right(v).mean()), 'val_b_cAcc': float(right.mean()),
                'ACD': sc['ACD'], 'mCCD': sc['mCCD'], '_right': right}
    for p in out['policies'].values():
        r = p.pop('_right')
        fixed, broken = int((r & ~base_right).sum()), int((~r & base_right).sum())
        p.update(fixed_vs_base=fixed, broken_vs_base=broken, sign_p=sign_test(fixed, broken),
                 delta_vs_base=float(r.mean() - base_right.mean()))
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--name', required=True)
    ap.add_argument('--val', type=Path, required=True, help='Val-B labels: the images that are scored')
    ap.add_argument('--val-a', type=Path, required=True, help='Val-A labels: where the calib images are')
    ap.add_argument('--pred', type=Path, required=True)
    ap.add_argument('--calib', type=Path, required=True, help='val_a_split.json (key "calib")')
    ap.add_argument('--task-config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--train-ids', type=Path, help='selections/<name>.json of the model: with it, the choices are '
                    'made on calib + the Val-A pool images the model did not train on, not on calib alone')
    ap.add_argument('--max-sku', type=int, default=20)
    ap.add_argument('--min-objects', type=int, default=30)
    ap.add_argument('--min-gain', type=int, default=2)
    ap.add_argument('--max-recall-drop', type=float, help='round 3: a SKU threshold may lower that SKU\'s recall on '
                    'the selection images by at most this much (default: no limit)')
    args = ap.parse_args(argv)
    val_b, val_a = load_json(args.val), load_json(args.val_a)
    calib = load_json(args.calib)['calib']
    target = sorted(i['id'] for i in val_b['images'])
    if set(calib) & set(target):
        raise ValueError('calib images are in Val-B')
    select = None
    if args.train_ids:
        trained = set(load_json(args.train_ids)['image_ids'])
        select = sorted(set(calib) | (set(load_json(args.calib)['pool']) - trained))
    keep = set(select or calib)
    val_full = {**val_b, 'images': val_b['images'] + [i for i in val_a['images'] if i['id'] in keep],
                'annotations': val_b['annotations'] + [a for a in val_a['annotations'] if a['image_id'] in keep]}
    cfg = TaskConfig.load(args.task_config)
    preds = load_predictions(args.pred)
    if not np.isin(sorted(keep | set(target)), preds.image_id).all():
        raise ValueError(f'{args.pred}: no predictions for every selection and Val-B image (use runs/<run>/task_5)')
    result = {'name': args.name, 'calib_images': len(calib), 'target_images': len(target),
              **run(val_full, cfg, preds, calib, target, max_sku=args.max_sku,
                    min_objects=args.min_objects, min_gain=args.min_gain, select_ids=select,
                    max_recall_drop=args.max_recall_drop),
              'select_images': len(select or calib), 'max_recall_drop': args.max_recall_drop}
    save_json(args.out, result, indent=1)
    return result


if __name__ == '__main__':
    r = main()
    print(json.dumps({k: {m: v[m] for m in ('calib_cAcc', 'val_b_cAcc', 'delta_vs_base', 'fixed_vs_base',
                                            'broken_vs_base', 'sign_p')} for k, v in r['policies'].items()}, indent=1))
