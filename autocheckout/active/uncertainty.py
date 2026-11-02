"""A2: how unsure the model is about each image, from its saved predictions only (no label is read).

Boxes: top-1 class per query, class-agnostic NMS (``NMS_IOU``), score >= ``FLOOR``. For a box, the second class is
the next best class of the SAME query in the saved top-k rows (absent -> score 0), as ``checkout-system`` asks the
cashier (``ac_worker.alternatives``):

- ``margin = s1 - s2``;
- ``entropy``: Shannon entropy of the query's saved class scores (at most ``TOPK``), renormalised to sum 1, divided
  by log(TOPK). The model uses sigmoid scores and the file keeps only the top rows per image, so this is an entropy
  over the saved candidates, not over all 200 classes.

Image (counted boxes = score >= ``COUNT_SCORE``, the "< 0.30 asks the cashier" rule of the counter):

- ``min_margin``: smallest margin of a counted box; ``mean_k_margin``: mean of its ``K`` smallest margins;
- ``max_entropy``: largest entropy of a counted box;
- ``n_low``: boxes with FLOOR <= score < COUNT_SCORE (objects the model may be missing);
- ``top_labels``: first and second class of the most unsure counted box (used to balance SKUs).
An image without a counted box gets margin 0 when it has low boxes (nothing is sure) and 1 otherwise (nothing seen).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from autocheckout.counting import dedup_detections
from autocheckout.predictions import Predictions

FLOOR, COUNT_SCORE, NMS_IOU, K, TOPK = 0.05, 0.30, 0.45, 3, 5


def _second_and_entropy(preds: Predictions) -> dict[tuple[int, int], tuple[int, float, float]]:
    """(image, query) -> (second label, second score, entropy) over the saved rows of that query."""
    order = np.lexsort((preds.label, -preds.score, preds.query, preds.image_id))
    img, qry = preds.image_id[order], preds.query[order]
    lab, sco = preds.label[order], preds.score[order]
    starts = np.flatnonzero(np.r_[True, (img[1:] != img[:-1]) | (qry[1:] != qry[:-1])])
    out = {}
    for s, e in zip(starts, np.r_[starts[1:], len(order)], strict=True):
        scores = sco[s:min(e, s + TOPK)].astype(np.float64)
        second = (int(lab[s + 1]), float(sco[s + 1])) if e - s > 1 else (-1, 0.0)
        total = scores.sum()
        p = scores / total if total > 0 else np.ones_like(scores) / len(scores)
        p = p[p > 0]
        ent = float(-(p * np.log(p)).sum() / math.log(TOPK))
        out[(int(img[s]), int(qry[s]))] = (*second, ent)
    return out


def box_table(preds: Predictions, nms_iou: float = NMS_IOU, floor: float = FLOOR) -> list[dict]:
    """One dict per kept box: image_id, label, score, second_label, second_score, margin, entropy."""
    kept = dedup_detections(preds, nms_iou)
    kept = kept.subset(kept.score >= floor)
    extra = _second_and_entropy(preds)
    rows = []
    for i in range(len(kept)):
        image_id, query = int(kept.image_id[i]), int(kept.query[i])
        l2, s2, ent = extra[(image_id, query)]
        s1 = float(kept.score[i])
        rows.append({'image_id': image_id, 'label': int(kept.label[i]), 'score': s1, 'second_label': l2,
                     'second_score': s2, 'margin': s1 - s2, 'entropy': ent})
    return rows


def image_scores(preds: Predictions, image_ids: Iterable[int], k: int = K) -> list[dict]:
    """One dict per image of ``image_ids`` (sorted by id); see the module docstring."""
    ids = sorted(set(int(i) for i in image_ids))
    sub = preds.subset(np.isin(preds.image_id, ids))
    by_image: dict[int, list[dict]] = {i: [] for i in ids}
    for row in box_table(sub):
        by_image[row['image_id']].append(row)
    out = []
    for image_id in ids:
        boxes = by_image[image_id]
        counted = sorted((b for b in boxes if b['score'] >= COUNT_SCORE), key=lambda b: (b['margin'], -b['score']))
        n_low = sum(1 for b in boxes if b['score'] < COUNT_SCORE)
        if counted:
            margins = [b['margin'] for b in counted]
            worst = counted[0]
            entry = {'min_margin': margins[0], 'mean_k_margin': float(np.mean(margins[:k])),
                     'max_entropy': max(b['entropy'] for b in counted),
                     'top_labels': [worst['label']] + ([worst['second_label']] if worst['second_label'] >= 0 else [])}
        else:
            fallback = 0.0 if n_low else 1.0
            entry = {'min_margin': fallback, 'mean_k_margin': fallback, 'max_entropy': 0.0, 'top_labels': []}
        out.append({'image_id': image_id, 'n_counted': len(counted), 'n_low': n_low, **entry})
    return out
