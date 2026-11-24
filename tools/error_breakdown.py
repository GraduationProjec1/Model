"""B0: why the images a model counts wrong are wrong: missed, extra or swapped objects, per SKU and per level.

Counts as ``tools/al_score.py`` (class-agnostic NMS of the top-1 row of each query, one score threshold per task
group), then matches every GT object to at most one counted box (IoU >= 0.5, best-overlapping objects first, as
``tools/sku_regression.match_image``) and names each error:

  swap            a counted box on the object, of another SKU
  miss_nms        no counted box; a box of its SKU above the threshold was removed by NMS
  miss_threshold  no counted box; a box of its SKU on it scored under the threshold
  miss            no counted box, and no box of its SKU on it
  extra_dup       a counted box left over on an object already counted (IoU >= 0.5)
  extra_shift     a counted box left over that half-covers an object (0.1 <= IoU < 0.5)
  extra_bg        a counted box left over on no object (IoU < 0.1)

    python -m tools.error_breakdown --val inputs/val_b/val_full.json --pred runs/<run>/task_5/pred_val.npz \\
        --policy relearn/active/scored/<name>.json --task-config configs/relearn/task_config_relearn.json \\
        [--rows inputs/<name>/train_rows.json] --out results/error_breakdown_<name>

``--policy`` is a ``tools/al_score.py`` result (thresholds chosen on Val-A-calib) or a calibration_count*.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from autocheckout.cl_metrics import box_iou
from autocheckout.counting import dedup_detections, label_thresholds
from autocheckout.io import load_json, save_json
from autocheckout.predictions import load_predictions
from autocheckout.taskcfg import TaskConfig
from tools.sku_regression import MATCH_IOU, policy_vector

STAGE = 5
SHIFT_IOU = 0.1
KINDS = ('swap', 'miss_nms', 'miss_threshold', 'miss', 'extra_dup', 'extra_shift', 'extra_bg')


def _match(gt_boxes: np.ndarray, boxes: np.ndarray) -> dict[int, int]:
    """GT index -> counted box index, greedy as sku_regression.match_image."""
    pairs: dict[int, int] = {}
    if len(gt_boxes) and len(boxes):
        ious, used = box_iou(gt_boxes, boxes), set()
        for g in np.argsort(-ious.max(axis=1), kind='stable'):
            free = [j for j in np.argsort(-ious[g], kind='stable') if ious[g, j] >= MATCH_IOU and j not in used]
            if free:
                pairs[int(g)] = int(free[0])
                used.add(int(free[0]))
    return pairs


def _on(box: np.ndarray, label: int, labels: np.ndarray, boxes: np.ndarray) -> bool:
    """A box of ``label`` covers ``box`` (IoU >= 0.5)."""
    same = labels == label
    return bool(same.any() and box_iou(box[None], boxes[same]).max() >= MATCH_IOU)


def explain_image(gt_labels: np.ndarray, gt_boxes: np.ndarray, labels: np.ndarray, boxes: np.ndarray,
                  low_labels: np.ndarray, low_boxes: np.ndarray, nms_labels: np.ndarray,
                  nms_boxes: np.ndarray) -> list[tuple[str, int, int]]:
    """Errors of one image as (kind, GT label, counted label); -1 where there is none. ``labels``/``boxes``: the
    counted boxes; ``low_*``: boxes kept by NMS but under their threshold; ``nms_*``: boxes over their threshold
    that NMS removed."""
    pairs = _match(gt_boxes, boxes)
    errors = []
    for g, label in enumerate(gt_labels.tolist()):
        if g in pairs:
            if labels[pairs[g]] != label:
                errors.append(('swap', label, int(labels[pairs[g]])))
        elif _on(gt_boxes[g], label, nms_labels, nms_boxes):
            errors.append(('miss_nms', label, -1))
        elif _on(gt_boxes[g], label, low_labels, low_boxes):
            errors.append(('miss_threshold', label, -1))
        else:
            errors.append(('miss', label, -1))
    taken = set(pairs.values())
    for j in range(len(labels)):
        if j not in taken:
            best = box_iou(boxes[j:j + 1], gt_boxes).max() if len(gt_boxes) else 0.0
            kind = 'extra_dup' if best >= MATCH_IOU else 'extra_shift' if best >= SHIFT_IOU else 'extra_bg'
            errors.append((kind, -1, int(labels[j])))
    return errors


def thresholds_of(policy: dict[str, Any], cfg: TaskConfig) -> tuple[np.ndarray, float]:
    if 'thresholds' in policy:
        groups = [list(cfg.task(t).labels) for t in range(1, STAGE + 1)]
        vector = [policy['thresholds'][f'task_{t}'] for t in range(1, STAGE + 1)]
        return label_thresholds(groups, vector), float(policy['nms_iou'])
    return policy_vector(policy, cfg, STAGE)


def breakdown(val: dict[str, Any], preds, thresholds: np.ndarray, nms_iou: float, rows: set[int]) -> dict:
    learned = len(thresholds)
    ids = sorted(i['id'] for i in val['images'])
    level = {i['id']: i.get('level', '?') for i in val['images']}
    preds = preds.subset(np.isin(preds.image_id, ids) & (preds.label < learned))
    top1 = preds.top1_per_query()
    kept = dedup_detections(preds, nms_iou) if nms_iou else top1
    kept_keys = set(zip(kept.image_id.tolist(), kept.query.tolist()))
    removed = np.array([k not in kept_keys for k in zip(top1.image_id.tolist(), top1.query.tolist())], dtype=bool)
    over_top1 = top1.score >= thresholds[top1.label]
    over_kept = kept.score >= thresholds[kept.label]
    by_image = defaultdict(lambda: defaultdict(list))
    for name, rows_, mask in (('counted', kept, over_kept), ('low', kept, ~over_kept), ('nms', top1, removed & over_top1)):
        for i in np.flatnonzero(mask):
            by_image[int(rows_.image_id[i])][name].append((int(rows_.label[i]), rows_.boxes[i]))
    gt = defaultdict(list)
    for a in val['annotations']:
        if a['category_id'] < learned:
            x, y, w, h = a['bbox']
            gt[a['image_id']].append((a['category_id'], [x, y, x + w, y + h]))

    def arrays(items):
        return (np.array([c for c, _ in items], dtype=np.int64),
                np.array([b for _, b in items], dtype=np.float64).reshape(-1, 4))

    images = []
    for image_id in ids:
        g_labels, g_boxes = arrays(gt[image_id])
        parts = {k: arrays(by_image[image_id][k]) for k in ('counted', 'low', 'nms')}
        right = Counter(g_labels.tolist()) == Counter(parts['counted'][0].tolist())
        errors = explain_image(g_labels, g_boxes, *parts['counted'], *parts['low'], *parts['nms'])
        images.append({'id': image_id, 'level': level[image_id], 'right': right, 'objects': len(g_labels),
                       'errors': [list(e) for e in errors]})
    return summarize(images, rows)


def summarize(images: list[dict], rows: set[int]) -> dict:
    wrong = [i for i in images if not i['right']]
    kinds = Counter(e[0] for i in wrong for e in i['errors'])
    only_kind = Counter()
    for i in wrong:
        found = {e[0] for e in i['errors']}
        if len(found) == 1:
            only_kind[found.pop()] += 1
    families = {'swap': {'swap'}, 'miss_or_extra_policy': {'miss_nms', 'miss_threshold', 'extra_dup'},
                'miss': {'miss'}, 'extra_other': {'extra_shift', 'extra_bg'}}
    only_family = Counter(name for i in wrong for name, members in families.items()
                          if i['errors'] and {e[0] for e in i['errors']} <= members)
    n_errors = Counter(min(len(i['errors']), 4) for i in wrong)
    levels = defaultdict(Counter)
    for i in images:
        levels[i['level']]['images'] += 1
        levels[i['level']]['wrong'] += not i['right']

    involved, only_sku = Counter(), Counter()
    sku_kinds = defaultdict(Counter)
    pairs, only_pair = Counter(), Counter()
    for i in wrong:
        skus_per_error = [{s for s in (e[1], e[2]) if s >= 0} for e in i['errors']]
        for e, skus in zip(i['errors'], skus_per_error):
            for s in skus:
                involved[s] += 1
                sku_kinds[s][e[0]] += 1
            if e[0] == 'swap':
                pairs[tuple(sorted((e[1], e[2])))] += 1
        if skus_per_error:
            common = set.intersection(*skus_per_error)
            for s in common:
                only_sku[s] += 1
            pair_sets = {tuple(sorted((e[1], e[2]))) for e in i['errors'] if e[0] == 'swap'}
            if len(pair_sets) == 1 and all(e[0] == 'swap' for e in i['errors']):
                only_pair[pair_sets.pop()] += 1
    inconsistent = {'wrong_without_error': sum(1 for i in wrong if not i['errors']),
                    'right_with_error': sum(1 for i in images if i['right'] and i['errors'])}
    top = sorted(involved, key=lambda s: (-only_sku[s], -involved[s], s))[:30]
    return {
        'images': len(images), 'right': len(images) - len(wrong), 'wrong': len(wrong),
        'cAcc': round((len(images) - len(wrong)) / len(images), 6), 'inconsistent': inconsistent,
        'errors_by_kind': {k: kinds[k] for k in KINDS},
        'wrong_images_with_one_kind_only': {k: only_kind[k] for k in KINDS},
        'wrong_images_with_one_family_only': {k: only_family[k] for k in families},
        'families': {k: sorted(v) for k, v in families.items()},
        'wrong_images_by_error_count': {('4+' if k == 4 else str(k)): n_errors[k] for k in sorted(n_errors)},
        'levels': {k: {'images': v['images'], 'wrong': v['wrong'], 'cAcc': round(1 - v['wrong'] / v['images'], 4)}
                   for k, v in sorted(levels.items())},
        'skus': [{'label': s, 'in_rows': s in rows, 'errors': involved[s], 'images_only_this_sku': only_sku[s],
                  **{k: sku_kinds[s][k] for k in KINDS}} for s in top],
        'pairs': [{'pair': list(p), 'in_rows': set(p) <= rows, 'swaps': n, 'images_only_this_pair': only_pair[p]}
                  for p, n in pairs.most_common(20)],
        'skus_in_rows_share_of_errors': round(sum(1 for i in wrong for e in i['errors']
                                                  if {e[1], e[2]} & rows) / max(1, sum(kinds.values())), 4),
        'wrong_images_touching_rows': sum(1 for i in wrong if any({e[1], e[2]} & rows for e in i['errors'])),
        'wrong_images_only_rows': sum(1 for i in wrong if i['errors'] and all(
            {s for s in (e[1], e[2]) if s >= 0} <= rows for e in i['errors'])),
        'per_image': wrong,
    }


def markdown(r: dict, names: dict[int, str]) -> str:
    n = lambda s: f"{s} {names.get(s, '')}"  # noqa: E731
    lines = [f"# Phân rã lỗi: {r['wrong']} / {r['images']} ảnh sai (cAcc {r['cAcc']:.4f})", '',
             '| Loại lỗi | Số vật | Ảnh chỉ có loại này |', '|---|---|---|']
    lines += [f"| {k} | {r['errors_by_kind'][k]} | {r['wrong_images_with_one_kind_only'][k]} |" for k in KINDS]
    lines += ['', '| Nhóm lỗi | Ảnh chỉ có nhóm này |', '|---|---|']
    lines += [f"| {k} ({', '.join(r['families'][k])}) | {v} |" for k, v in r['wrong_images_with_one_family_only'].items()]
    lines += ['', '| Số lỗi trong ảnh sai | Ảnh |', '|---|---|']
    lines += [f'| {k} | {v} |' for k, v in r['wrong_images_by_error_count'].items()]
    lines += ['', '| Độ khó | Ảnh | Sai | cAcc |', '|---|---|---|---|']
    lines += [f"| {k} | {v['images']} | {v['wrong']} | {v['cAcc']:.4f} |" for k, v in r['levels'].items()]
    lines += ['', f"Ảnh sai có dính 17 SKU học lại: {r['wrong_images_touching_rows']}; "
                  f"ảnh sai chỉ do 17 SKU: {r['wrong_images_only_rows']}; "
                  f"tỷ lệ lỗi dính 17 SKU: {r['skus_in_rows_share_of_errors']:.2%}", '',
              '| SKU | Trong 17 SKU | Lỗi | Ảnh chỉ sai vì SKU này | ' + ' | '.join(KINDS) + ' |',
              '|' + '---|' * (4 + len(KINDS))]
    lines += [f"| {n(s['label'])} | {'có' if s['in_rows'] else ''} | {s['errors']} | {s['images_only_this_sku']} | "
              + ' | '.join(str(s[k]) for k in KINDS) + ' |' for s in r['skus'][:20]]
    lines += ['', '| Cặp nhầm | Trong 17 SKU | Lần nhầm | Ảnh chỉ sai vì cặp này |', '|---|---|---|---|']
    lines += [f"| {n(p['pair'][0])} ↔ {n(p['pair'][1])} | {'có' if p['in_rows'] else ''} | {p['swaps']} | "
              f"{p['images_only_this_pair']} |" for p in r['pairs']]
    return '\n'.join(lines) + '\n'


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--val', type=Path, required=True)
    ap.add_argument('--pred', type=Path, required=True)
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--task-config', type=Path, required=True)
    ap.add_argument('--rows', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = TaskConfig.load(args.task_config)
    thresholds, nms_iou = thresholds_of(load_json(args.policy), cfg)
    rows = set(load_json(args.rows)) if args.rows else set()
    report = breakdown(load_json(args.val), load_predictions(args.pred), thresholds, nms_iou, rows)
    report['inputs'] = {'val': str(args.val), 'pred': str(args.pred), 'policy': str(args.policy),
                        'nms_iou': nms_iou, 'thresholds_by_task': sorted(set(thresholds.tolist()))}
    raw = json.loads(args.task_config.read_text())
    names = {c['label']: c['name'] for t in raw['tasks'] for c in t['classes']}
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out / 'error_breakdown.json', report, indent=1)
    (args.out / 'error_breakdown.md').write_text(markdown(report, names))
    return report


if __name__ == '__main__':
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k not in ('per_image', 'skus', 'pairs')}, indent=1))
