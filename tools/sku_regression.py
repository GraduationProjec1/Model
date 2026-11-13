"""Per-SKU comparison of two models on the same images (update-model plan, Task 2 and the gates of I10).

Both prediction files are counted with their own counting policy (class-agnostic NMS of the top-1 row of each
query, one score threshold per task group, as ``full_syn.score_val`` and the relearn gate), then every GT object
is matched to at most one counted box (IoU >= 0.5, best-overlapping objects first). The report gives the cAcc of
both models, the images fixed (wrong -> right) and broken (right -> wrong) with a one-sided sign test, the share
of objects counted right per SKU, the SKUs that fell, and the confusion pairs before and after.

    python -m tools.sku_regression --val val_full.json --task-config task_config.json --stage 5 \\
        --base-pred base/task_5/pred_val.npz --base-policy base/calibration_count_m2.json \\
        --new-pred new/task_5/pred_val.npz --new-policy new/calibration_count_m2.json \\
        [--rows train_rows.json] [--json report.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binom

from autocheckout.cl_metrics import box_iou
from autocheckout.counting import dedup_detections, label_thresholds
from autocheckout.io import load_json, save_json
from autocheckout.predictions import Predictions, load_predictions
from autocheckout.taskcfg import TaskConfig

MATCH_IOU = 0.5
MIN_OBJECTS = 30  # SKUs with fewer GT objects are not judged for a fall
FALL = 0.02  # fall of the share of objects counted right that is still reported (gate I10, until 12/11/2026)
# Gate rule since 12/11/2026 (docs/02-cac-lan-train/08-2026-11-12-tong-ket-va-ke-hoach.md, 2.4): an SKU "fell" when its
# share of right objects dropped by more than FALL_RECALL AND it lost at least FALL_OBJECTS right objects, so that a
# 3-object wobble on a 150-object SKU no longer stops a run that fixed hundreds of images.
FALL_RECALL = 0.03
FALL_OBJECTS = 5


def fell(sku: dict[str, Any]) -> bool:
    """Gate rule of 12/11/2026 for one ``per_sku`` entry."""
    drop = sku['base'] - sku['new']
    lost = round(sku['objects'] * drop, 6)  # right objects lost; rounded so 125 * 0.04 counts as 5
    return sku['objects'] >= MIN_OBJECTS and drop > FALL_RECALL and lost >= FALL_OBJECTS


def policy_vector(policy: dict[str, Any], cfg: TaskConfig, stage: int) -> tuple[np.ndarray, float]:
    """Per-label score thresholds and NMS IoU of stage ``stage`` of a calibration_count*.json file."""
    entry = policy['stages'][str(stage)]
    groups = [list(cfg.task(t).labels) for t in range(1, stage + 1)]
    if entry.get('group_thresholds'):
        thresholds = [entry['group_thresholds'][f'task_{t}'] for t in range(1, stage + 1)]
    else:
        thresholds = [entry['score_threshold']] * stage
    return label_thresholds(groups, thresholds), float(entry.get('nms_iou') or 0)


def counted(preds: Predictions, thresholds: np.ndarray, nms_iou: float) -> Predictions:
    """The boxes a model counts: top-1 row per query, NMS when the policy has one, score >= its label's threshold."""
    rows = dedup_detections(preds, nms_iou) if nms_iou else preds.top1_per_query()
    seen = rows.label < len(thresholds)
    keep = np.zeros(len(rows), dtype=bool)
    keep[seen] = rows.score[seen] >= thresholds[rows.label[seen]]
    return rows.subset(keep)


def match_image(gt_labels: np.ndarray, gt_boxes: np.ndarray, labels: np.ndarray, boxes: np.ndarray) -> dict:
    """Greedy matching of one image: objects with the best overlap first, each box used once."""
    result = {'right': Counter(), 'missed': Counter(), 'swapped': Counter(), 'extra': Counter()}
    used: set[int] = set()
    if len(labels):
        ious = box_iou(gt_boxes, boxes)
        for g in np.argsort(-ious.max(axis=1), kind='stable'):
            candidates = [j for j in np.argsort(-ious[g], kind='stable') if ious[g, j] >= MATCH_IOU and j not in used]
            if not candidates:
                result['missed'][int(gt_labels[g])] += 1
                continue
            used.add(int(candidates[0]))
            predicted = int(labels[candidates[0]])
            if predicted == gt_labels[g]:
                result['right'][int(gt_labels[g])] += 1
            else:
                result['swapped'][(int(gt_labels[g]), predicted)] += 1
    else:
        result['missed'].update(int(c) for c in gt_labels)
    result['extra'].update(int(labels[j]) for j in range(len(labels)) if j not in used)
    return result


def evaluate(val: dict[str, Any], preds: Predictions, labels: Iterable[int]) -> dict[str, Any]:
    """Per image: exact count (cAcc); per object: right, missed, swapped pairs, extra boxes. GT and boxes of labels
    outside ``labels`` are ignored, as cAcc counts only the learned classes."""
    labels = set(labels)
    gt = defaultdict(list)
    for ann in val['annotations']:
        if ann['category_id'] in labels:
            x, y, w, h = ann['bbox']
            gt[ann['image_id']].append((ann['category_id'], [x, y, x + w, y + h]))
    rows = defaultdict(list)
    for i in np.flatnonzero(np.isin(preds.label, list(labels))):
        rows[int(preds.image_id[i])].append(i)
    images = sorted({ann['image_id'] for ann in val['annotations'] if ann['category_id'] in labels})
    correct, totals = {}, {key: Counter() for key in ('right', 'missed', 'swapped', 'extra', 'objects')}
    for image_id in images:
        gt_labels = np.array([c for c, _ in gt[image_id]], dtype=np.int64)
        gt_boxes = np.array([b for _, b in gt[image_id]], dtype=np.float64).reshape(-1, 4)
        index = np.array(rows.get(image_id, []), dtype=np.int64)
        labels_i, boxes_i = preds.label[index].astype(np.int64), preds.boxes[index]
        correct[image_id] = Counter(gt_labels.tolist()) == Counter(labels_i.tolist())
        for key, value in match_image(gt_labels, gt_boxes, labels_i, boxes_i).items():
            totals[key].update(value)
        totals['objects'].update(gt_labels.tolist())
    return {'correct': correct, **totals}


def compare(val: dict[str, Any], base: Predictions, new: Predictions, labels: Iterable[int],
            rows: Iterable[int] = (), top: int = 20) -> dict[str, Any]:
    """Report of ``new`` against ``base`` on the same images (both already ``counted``)."""
    labels, rows = sorted(labels), set(rows)
    a, b = evaluate(val, base, labels), evaluate(val, new, labels)
    images = sorted(a['correct'])
    fixed = sum(1 for i in images if not a['correct'][i] and b['correct'][i])
    broken = sum(1 for i in images if a['correct'][i] and not b['correct'][i])
    def over(r: dict, label: int) -> int:  # round 3 v1.5: boxes counted as ``label`` without its object
        return r['extra'][label] + sum(c for (t, p), c in r['swapped'].items() if p == label and t != label)

    per_sku = []
    for label in labels:
        n = a['objects'][label]
        if n:
            per_sku.append({'label': label, 'objects': n, 'base': a['right'][label] / n,
                            'new': b['right'][label] / n, 'in_rows': label in rows,
                            'over_base': over(a, label), 'over_new': over(b, label)})
    falls = [s for s in per_sku if fell(s)]
    falls_2pt = [s for s in per_sku if s['objects'] >= MIN_OBJECTS and s['base'] - s['new'] > FALL]

    def pairs(swapped: Counter) -> Counter:
        total = Counter()
        for (truth, predicted), count in swapped.items():
            total[tuple(sorted((truth, predicted)))] += count
        return total

    before, after = pairs(a['swapped']), pairs(b['swapped'])
    in_rows = [p for p in set(before) | set(after) if rows and set(p) <= rows]
    # 13/11/2026: the pair of the gate's "largest pair may not rise" (most base confusions, ties by pair)
    largest = min((p for p in in_rows if before[p]), key=lambda p: (-before[p], p), default=None)
    return {
        'images': len(images),
        'cAcc_base': float(np.mean([a['correct'][i] for i in images])) if images else float('nan'),
        'cAcc_new': float(np.mean([b['correct'][i] for i in images])) if images else float('nan'),
        'fixed': fixed, 'broken': broken,
        'sign_test_p': float(binom.sf(fixed - 1, fixed + broken, 0.5)) if fixed + broken else 1.0,
        'pair_confusions_in_rows': {'base': sum(before[p] for p in in_rows), 'new': sum(after[p] for p in in_rows)},
        'largest_pair_in_rows': (None if largest is None else
                                 {'pair': list(largest), 'base': before[largest], 'new': after[largest]}),
        'other_objects_called_rows': {
            'base': sum(c for (t, p), c in a['swapped'].items() if rows and t not in rows and p in rows),
            'new': sum(c for (t, p), c in b['swapped'].items() if rows and t not in rows and p in rows)},
        'falls': sorted(falls, key=lambda s: s['new'] - s['base']),
        'falls_2pt': sorted(falls_2pt, key=lambda s: s['new'] - s['base']),  # the rule before 12/11/2026, reported only
        'per_sku': per_sku,
        'top_pairs': [{'pair': list(p), 'base': before[p], 'new': after[p]}
                      for p, _ in (before + after).most_common(top)],
    }


def markdown(report: dict[str, Any], names: dict[int, str] | None = None) -> str:
    names = names or {}
    name = lambda label: f"{label} {names.get(label, '')}".strip()  # noqa: E731
    lines = ["| Ảnh | cAcc base | cAcc new | fixed | broken | p (sign test) |", '|---|---|---|---|---|---|',
             f"| {report['images']} | {report['cAcc_base']:.4f} | {report['cAcc_new']:.4f} | {report['fixed']} | "
             f"{report['broken']} | {report['sign_test_p']:.3g} |", '',
             f"Nhầm trong các cặp của train_rows: {report['pair_confusions_in_rows']['base']} → "
             f"{report['pair_confusions_in_rows']['new']}; vật SKU khác bị gọi thành SKU của train_rows: "
             f"{report['other_objects_called_rows']['base']} → {report['other_objects_called_rows']['new']}", '',
             f"SKU (≥ {MIN_OBJECTS} vật) tụt > {FALL}: {len(report['falls'])}"]
    lines += [f"- {name(s['label'])}: {s['base']:.3f} → {s['new']:.3f} ({s['objects']} vật)" for s in report['falls']]
    lines += ['', '| Cặp | base | new |', '|---|---|---|']
    lines += [f"| {name(p['pair'][0])} / {name(p['pair'][1])} | {p['base']} | {p['new']} |"
              for p in report['top_pairs']]
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--val', required=True, help='COCO file of the images compared (e.g. Val-B)')
    parser.add_argument('--task-config', required=True)
    parser.add_argument('--stage', type=int, required=True, help='stage of the policies (learned classes)')
    for side in ('base', 'new'):
        parser.add_argument(f'--{side}-pred', required=True)
        parser.add_argument(f'--{side}-policy', required=True, help='calibration_count*.json')
    parser.add_argument('--rows', help='JSON list of the labels the new model retrained (--train_rows)')
    parser.add_argument('--json', help='write the report here')
    args = parser.parse_args(argv)

    cfg = TaskConfig.load(args.task_config)
    val = load_json(args.val)
    ids = np.array(sorted(image['id'] for image in val['images']))
    models = []
    for side in ('base', 'new'):
        preds = load_predictions(getattr(args, f'{side}_pred'))
        preds = preds.subset(np.isin(preds.image_id, ids))
        thresholds, nms_iou = policy_vector(load_json(getattr(args, f'{side}_policy')), cfg, args.stage)
        models.append(counted(preds, thresholds, nms_iou))
    rows = json.loads(Path(args.rows).read_text()) if args.rows else []
    report = compare(val, *models, range(cfg.seen_classes(args.stage)), rows)
    print(markdown(report, cfg.label_names()))
    if args.json:
        save_json(args.json, report, indent=1)
    return report


if __name__ == '__main__':
    main()
