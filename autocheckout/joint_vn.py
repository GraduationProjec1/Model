"""Scores of a model that learns RPC and VN products together (A6, AN/cyclegan/03_QUYET_DINH_VA_PLAN.md).

Config: every task holds RPC SKUs (rpc_category_id <= 200) and VN SKUs (201-225, tools/make_merged_task_config.py);
the VN photos hold VN products only, the RPC images RPC products only. Validation = RPC images + VN photos, one
prediction pass. Counting thresholds (AN/cyclegan/02_KIEM_TRA_CHIA_TASK_VA_VAL_TEST.md section 5.3), as
new_skus.score_stage (DL-07 of Checkout-system):

* RPC part of every task: one threshold per task, chosen on Val-A-calib counting the RPC labels
  (``select_group_thresholds``);
* VN: one threshold for every learned VN SKU (52 val photos are too few to split it by task), chosen on the VN val
  photos with the RPC thresholds fixed, every learned label counted (an RPC label on a VN product is an error the VN
  threshold must live with).

Never chosen on Val-B or test. The report counts every learned label (cAcc-M2, D16) and splits VN into the
products of task 1 and those added in tasks 2-6 (too few VN objects per task to score each task alone), with the
VN -> RPC and RPC -> VN confusions measured apart.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from autocheckout.cl_metrics import box_iou, group_ap, load_coco
from autocheckout.counting import (
    count_matrix,
    counting_scores,
    dedup_detections,
    gt_counts,
    label_thresholds,
    select_group_thresholds,
)
from autocheckout.feedback_fix import subset
from autocheckout.predictions import Predictions
from autocheckout.taskcfg import TaskConfig
from tools.al_score import NMS, UNLEARNED_IOU, _view
from tools.eval_count import CALIBRATION_GRID

VN_FIRST_ID = 201   # rpc_category_id of VN01 in the merged configs (make_merged_task_config.py)
MATCH_IOU = 0.5     # as new_skus.cross_counts


def split_groups(cfg: TaskConfig, stage: int) -> list[tuple[str, list[int], str]]:
    """(name, labels, origin) of the RPC and the VN part of every task up to ``stage``; empty parts are left out."""
    out = []
    for t in range(1, stage + 1):
        for origin in ('rpc', 'vn'):
            labels = [c.label for c in cfg.task(t).classes
                      if c.rpc_category_id is not None and (c.rpc_category_id >= VN_FIRST_ID) == (origin == 'vn')]
            if labels:
                out.append((f'task_{t}_{origin}', labels, origin))
    return out


def _cacc(coco, rows: Predictions, labels: list[int], vector: np.ndarray) -> tuple[float, list[int], dict]:
    ids = sorted(coco.getImgIds())
    gt = gt_counts(coco, ids, labels)
    pred = count_matrix(rows, ids, labels, vector)
    right = np.all(pred == gt, axis=1)
    return (float(right.mean()) if ids else float('nan'), [i for i, ok in zip(ids, right, strict=True) if ok],
            counting_scores(pred, gt))


def choose_thresholds(val_full: dict, cfg: TaskConfig, preds: Predictions, stage: int, *, calib: list[int],
                      vn_val: list[int], grid=CALIBRATION_GRID, unlearned_iou: float | None = UNLEARNED_IOU
                      ) -> tuple[dict[str, float], dict[str, Any]]:
    """({'task_<t>_rpc': threshold, ..., 'vn': threshold}, selection scores). RPC groups on ``calib``, then the one
    VN threshold on ``vn_val`` (highest cAcc, ties to the smaller, as ``select_threshold``)."""
    groups = split_groups(cfg, stage)
    seen = cfg.seen_classes(stage)
    rpc = [(n, g) for n, g, o in groups if o == 'rpc']
    vn = [(n, g) for n, g, o in groups if o == 'vn']
    coco, rows = _view(val_full, preds, sorted(calib), seen, unlearned_iou)
    rpc_vector, rpc_scores = select_group_thresholds(coco, dedup_detections(rows, NMS), [g for _, g in rpc], grid)
    chosen = dict(zip([n for n, _ in rpc], rpc_vector, strict=True))
    info = {'calib_images': len(coco.getImgIds()), 'calib_cAcc': rpc_scores['cAcc']}
    if vn:
        coco_v, rows_v = _view(val_full, preds, sorted(vn_val), seen, unlearned_iou)
        processed = dedup_detections(rows_v, NMS)
        all_groups = [g for _, g in rpc] + [g for _, g in vn]
        labels = sorted(label for g in all_groups for label in g)

        best, best_cacc = None, -1.0
        for t in grid:
            vector = label_thresholds(all_groups, [*rpc_vector, *[t] * len(vn)])
            value = _cacc(coco_v, processed, labels, vector)[0]
            if value > best_cacc:
                best, best_cacc = float(t), value
        chosen['vn'] = best
        info.update(vn_val_images=len(coco_v.getImgIds()), vn_val_cAcc=best_cacc)
    return chosen, info


def _claimed(coco: dict, rows: Predictions, targets: set[int], others: set[int]) -> dict[str, Any]:
    """Objects of ``targets`` that a counted detection of an ``others`` label covers (IoU >= MATCH_IOU)."""
    boxes: dict[int, list] = {}
    for a in coco['annotations']:
        if a['category_id'] in targets:
            boxes.setdefault(a['image_id'], []).append(a['bbox'])
    rows = rows.subset(np.isin(rows.label, list(others)))
    objects = claimed = 0
    for image_id, gt in boxes.items():
        gt = np.array([[x, y, x + w, y + h] for x, y, w, h in gt], dtype=np.float64)
        mine = rows.image_id == image_id
        iou = box_iou(gt, rows.boxes[mine])
        claimed += int((iou >= MATCH_IOU).any(axis=1).sum()) if iou.size else 0
        objects += len(gt)
    return {'objects': objects, 'claimed': claimed, 'rate': claimed / objects if objects else 0.0}


def report(coco_full: dict, cfg: TaskConfig, preds: Predictions, stage: int, thresholds: dict[str, float], *,
           rpc_ids: list[int], vn_ids: list[int], unlearned_iou: float | None = UNLEARNED_IOU) -> dict[str, Any]:
    """Scores with fixed thresholds on RPC images ``rpc_ids`` (Val-B or test2019) and VN photos ``vn_ids`` (VN val or
    test). cAcc-M2 counts every learned label; the ``*_labels_only`` variants count one origin."""
    groups = split_groups(cfg, stage)
    seen = cfg.seen_classes(stage)
    all_groups = [g for _, g, _ in groups]
    vector = label_thresholds(all_groups, [thresholds['vn' if o == 'vn' else n] for n, _, o in groups])
    labels = sorted(label for g in all_groups for label in g)
    rpc_labels = sorted(label for _, g, o in groups if o == 'rpc' for label in g)
    vn_labels = sorted(label for _, g, o in groups if o == 'vn' for label in g)

    def part(ids: list[int], own: list[int], name: str) -> dict[str, Any]:
        coco, rows = _view(coco_full, preds, sorted(ids), seen, unlearned_iou)
        processed = dedup_detections(rows, NMS)
        cacc, right, scores = _cacc(coco, processed, labels, vector)
        on = preds.subset(np.isin(preds.image_id, ids))
        return {'images': len(coco.getImgIds()), 'cAcc_m2': cacc,
                f'cAcc_m2_{name}_labels_only': _cacc(coco, processed, own, vector)[0],
                **{k: scores[k] for k in ('ACD', 'mCCD', 'mCIoU')},
                'AP50': group_ap(load_coco(subset(coco_full, ids)), on, own)['AP50'] if own else None,
                'right_images': right}

    out: dict[str, Any] = {'stage': stage, 'thresholds': thresholds, 'nms_iou': NMS}
    out['rpc'] = part(rpc_ids, rpc_labels, 'rpc') if rpc_ids else None
    out['vn'] = part(vn_ids, vn_labels, 'vn') if vn_ids else None
    counted = dedup_detections(preds.subset(np.isin(preds.image_id, [*rpc_ids, *vn_ids])), NMS)
    inside = counted.label < len(vector)
    counted = counted.subset(inside & (counted.score >= vector[np.where(inside, counted.label, 0)]))
    out['cross'] = {}
    if rpc_ids and vn_labels:
        rpc_coco = subset(coco_full, rpc_ids)
        out['cross']['rpc_to_vn'] = _claimed(rpc_coco, counted, set(rpc_labels), set(vn_labels))
        n_vn = int((np.isin(counted.image_id, rpc_ids) & np.isin(counted.label, vn_labels)).sum())
        out['cross']['vn_label_detections_per_rpc_image'] = n_vn / len(rpc_ids)
    if vn_ids:
        vn_coco = subset(coco_full, vn_ids)
        out['cross']['vn_to_rpc'] = _claimed(vn_coco, counted, set(vn_labels), set(rpc_labels))
        vn_groups = {'vn_task_1': [lb for n, g, o in groups if o == 'vn' and n == 'task_1_vn' for lb in g],
                     'vn_task_2_6': [lb for n, g, o in groups if o == 'vn' and n != 'task_1_vn' for lb in g]}
        out['vn']['groups'] = {}
        for name, group in vn_groups.items():
            if not group:
                continue
            ids = sorted({a['image_id'] for a in vn_coco['annotations'] if a['category_id'] in group})
            coco, rows = _view(coco_full, preds, ids, seen, unlearned_iou)
            out['vn']['groups'][name] = {
                'images': len(ids), 'objects': sum(1 for a in vn_coco['annotations'] if a['category_id'] in group),
                'cAcc': _cacc(coco, dedup_detections(rows, NMS), group, vector)[0],
                'AP50': group_ap(load_coco(vn_coco), preds.subset(np.isin(preds.image_id, vn_ids)), group)['AP50']}
    return out
