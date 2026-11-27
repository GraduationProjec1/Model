"""N0 (docs/ai/planning/2026-11-26-plan-N1-N5-hieu-chinh-task-va-loss.md §2): where the true class stands in the
queries that count an object as another SKU (swaps), from the logits of ``pdp/main.py --dump_logits 1``.

Counts like ``tools/al_score.py`` (top-1 class of each query, class-agnostic NMS, one threshold per task from
``--policy``), matches every GT object to at most one counted box (``tools/error_breakdown._match``), and for every
swap records the rank of the true class among the logits of that query, whether the winning and the true class are
in the same task / supercategory, and ``logit_win - logit_true`` per (task_win, task_true).

N1 (per-task calibration) can only fix a swap whose true class is second and in another task, so the gate of §2.4
reads the share of cross-task swaps with the true class at rank 2 and the direction of the logit gaps. Run it on
Val-A-calib only (default), never on Val-B.

    python -m tools.swap_rank --name r4 --logits runs/<run>/task_5/logits_val.npz \\
        --policy relearn/active/scored/al_margin_k_B500_r4.json [--data ../autocheckout-data]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autocheckout.counting import dedup_detections
from autocheckout.datapaths import Layout
from autocheckout.io import load_json, md5_file, save_json
from autocheckout.logit_dump import LogitDump, load_logits
from autocheckout.predictions import Predictions
from autocheckout.taskcfg import TaskConfig
from tools.error_breakdown import _match, thresholds_of

RESULTS = Path('results/swap_rank')
RANK2_GATE = 0.40
DIRECTION_GATE = 0.60


@dataclass(frozen=True)
class Matched:
    """GT objects matched to a counted query: their image, GT label, counted label and the query's logits."""

    image_id: np.ndarray
    true_label: np.ndarray
    pred_label: np.ndarray
    logits: np.ndarray

    def __len__(self) -> int:
        return len(self.image_id)


def match_queries(dump: LogitDump, val: dict, image_ids, nms_iou: float, thresholds: np.ndarray | None = None,
                  preds: Predictions | None = None) -> Matched:
    """Every GT object of ``image_ids`` (classes the dump has logits for) matched to a counted query: the top-1
    rows of ``preds`` (default: the uncalibrated dump) after NMS and, with ``thresholds``, over their threshold."""
    learned = dump.num_classes
    ids = set(int(i) for i in image_ids)
    preds = dump.predictions() if preds is None else preds
    kept = dedup_detections(preds.subset(np.isin(preds.image_id, list(ids))), nms_iou)
    if thresholds is not None:
        kept = kept.subset(kept.score >= thresholds[kept.label])
    row_of = {(int(i), int(q)): (n, k) for n, (i, queries) in enumerate(zip(dump.image_id, dump.query, strict=True))
              for k, q in enumerate(queries)}
    gt = defaultdict(list)
    for a in val['annotations']:
        if a['image_id'] in ids and a['category_id'] < learned:
            x, y, w, h = a['bbox']
            gt[a['image_id']].append((a['category_id'], [x, y, x + w, y + h]))
    columns = {'image_id': [], 'true_label': [], 'pred_label': [], 'logits': []}
    for image_id in sorted(ids):
        objects = gt.get(image_id, [])
        rows = np.flatnonzero(kept.image_id == image_id)
        if not objects or not len(rows):
            continue
        gt_boxes = np.array([b for _, b in objects], dtype=np.float64)
        for g, j in _match(gt_boxes, kept.boxes[rows]).items():
            n, k = row_of[(image_id, int(kept.query[rows[j]]))]
            columns['image_id'].append(image_id)
            columns['true_label'].append(objects[g][0])
            columns['pred_label'].append(int(kept.label[rows[j]]))
            columns['logits'].append(dump.logits[n, k].astype(np.float32))
    return Matched(np.array(columns['image_id'], dtype=np.int64), np.array(columns['true_label'], dtype=np.int64),
                   np.array(columns['pred_label'], dtype=np.int64),
                   np.stack(columns['logits']) if columns['logits'] else np.zeros((0, learned), dtype=np.float32))


def swap_table(matched: Matched, task_of: dict[int, int], supercategory: dict[int, str | None]) -> dict:
    """Ranks, task and supercategory of the true class of every swap, and the logit gaps per task pair."""
    swaps = np.flatnonzero(matched.true_label != matched.pred_label)
    ranks, kinds = Counter(), Counter()
    gaps = defaultdict(list)
    for s in swaps:
        true, win, logits = int(matched.true_label[s]), int(matched.pred_label[s]), matched.logits[s]
        rank = 1 + int((logits > logits[true]).sum())
        cross = task_of[win] != task_of[true]
        same_super = supercategory.get(win) is not None and supercategory.get(win) == supercategory.get(true)
        ranks['2' if rank == 2 else '3+' if rank > 2 else '1'] += 1
        kinds['cross_task'] += cross
        kinds['cross_task_rank2'] += cross and rank == 2
        kinds['same_supercategory'] += same_super
        kinds['cross_task_newer_wins'] += cross and task_of[win] > task_of[true]
        gaps[(task_of[win], task_of[true])].append(float(logits[win] - logits[true]))
    n, cross = len(swaps), kinds['cross_task']
    tasks = sorted(set(task_of.values()))
    return {
        'matched_objects': len(matched), 'swaps': n, 'true_rank': {k: ranks[k] for k in ('1', '2', '3+')},
        **{k: kinds[k] for k in ('cross_task', 'cross_task_rank2', 'same_supercategory', 'cross_task_newer_wins')},
        'share_cross_task': kinds['cross_task'] / n if n else None,
        'share_cross_task_rank2': kinds['cross_task_rank2'] / cross if cross else None,
        'share_newer_wins': kinds['cross_task_newer_wins'] / cross if cross else None,
        'share_same_supercategory': kinds['same_supercategory'] / n if n else None,
        'tasks': tasks,
        'gap_count': [[len(gaps[(w, t)]) for t in tasks] for w in tasks],
        'gap_mean': [[float(np.mean(gaps[(w, t)])) if gaps[(w, t)] else None for t in tasks] for w in tasks],
    }


def gate(table: dict) -> dict:
    """§2.4: N1 is worth doing when >= 40% of the cross-task swaps have the true class second and the newer task
    clearly wins (>= 60% of them)."""
    rank2, newer = table['share_cross_task_rank2'], table['share_newer_wins']
    go = rank2 is not None and rank2 >= RANK2_GATE and newer is not None and newer >= DIRECTION_GATE
    return {'rank2_gate': RANK2_GATE, 'direction_gate': DIRECTION_GATE, 'do_n1': bool(go)}


def label_maps(cfg: TaskConfig, learned: int) -> tuple[dict[int, int], dict[int, str | None]]:
    supercategory = {c.label: c.supercategory for task in cfg.tasks for c in task.classes}
    return ({label: cfg.task_of_label(label) for label in range(learned)},
            {label: supercategory.get(label) for label in range(learned)})


def check_dump(dump: LogitDump, val_path: Path) -> None:
    if dump.meta.get('ann_md5') not in (None, md5_file(val_path)):
        raise ValueError(f'the logits were not predicted on {val_path}')


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--name', required=True)
    ap.add_argument('--logits', type=Path, required=True, help='logits_val.npz of the model (all of val)')
    ap.add_argument('--policy', type=Path, required=True, help='tools/al_score.py result of the same model')
    ap.add_argument('--data', type=Path, help='the data folder (default: ../autocheckout-data or $AC_DATA)')
    ap.add_argument('--eval-dir', type=Path, help='val_full.json (default: <data>/eval)')
    ap.add_argument('--task-config', type=Path, help="default: the eval dir's task_config.json")
    ap.add_argument('--split', type=Path, help='default: <data>/relearn/active/splits/val_a_split.json')
    ap.add_argument('--part', default='calib', help='key of --split whose images are read (default calib)')
    ap.add_argument('--out', type=Path, help=f'default: {RESULTS}/<name>.json')
    args = ap.parse_args(argv)
    data = Layout.at(args.data)
    eval_dir = args.eval_dir or data.eval
    val_path = eval_dir / 'val_full.json'
    dump = load_logits(args.logits)
    check_dump(dump, val_path)
    cfg = TaskConfig.load(args.task_config or eval_dir / 'task_config.json')
    thresholds, nms_iou = thresholds_of(load_json(args.policy), cfg)
    ids = load_json(args.split or data.relearn / 'active/splits/val_a_split.json')[args.part]
    matched = match_queries(dump, load_json(val_path), ids, nms_iou, thresholds)
    table = swap_table(matched, *label_maps(cfg, dump.num_classes))
    result = {'name': args.name, 'logits': str(args.logits), 'policy': str(args.policy), 'images': args.part,
              'image_count': len(ids), **table, **gate(table)}
    save_json(args.out or RESULTS / f'{args.name}.json', result, indent=1)
    return result


if __name__ == '__main__':
    print(json.dumps(main(), indent=1))
