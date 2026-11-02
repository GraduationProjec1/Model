"""A1-A3 on a self-contained release (relearn_fix_v2): Val-A split, uncertainty of the round-1 model on the pool,
one selection file per (strategy, budget, seed), plus the references ``oracle`` (real images of the base variant,
chosen with the labels) and ``all`` (the whole pool). Writes ``<release>/active/``.

    python -m tools.al_select --release ../relearn_fix_v2 --budgets 250 500 850 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from autocheckout.active import select as sel
from autocheckout.active.splits import split_pool_calib
from autocheckout.active.uncertainty import image_scores
from autocheckout.io import load_json, md5_file, save_json
from autocheckout.predictions import load_predictions

PRED = 'seed/full-syn-r1/runs/EXP-FULL-SYN/task_5/pred_val.npz'


def _md5_ids(ids) -> str:
    return hashlib.md5(json.dumps(sorted(int(i) for i in ids)).encode()).hexdigest()


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--release', type=Path, required=True)
    ap.add_argument('--base', default='pilot', help='variant whose real images are the oracle reference')
    ap.add_argument('--calib-frac', type=float, default=0.2)
    ap.add_argument('--split-seed', type=int, default=0)
    ap.add_argument('--budgets', type=int, nargs='+', default=[250, 500, 850])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--strategies', nargs='+', default=list(sel.STRATEGIES))
    ap.add_argument('--alpha', type=float, default=1.5)
    args = ap.parse_args(argv)
    rel, out = args.release, args.release / 'active'

    val_a, val_b = load_json(rel / 'meta/val_a.json'), load_json(rel / 'meta/val_b.json')
    ids_a, ids_b = {i['id'] for i in val_a['images']}, {i['id'] for i in val_b['images']}
    if ids_a & ids_b:
        raise ValueError('Val-A and Val-B share images')
    preds = load_predictions(rel / PRED)
    if preds.meta.get('ann_md5') != md5_file(rel / 'eval/val_full.json'):
        raise ValueError('predictions were not made on eval/val_full.json')
    if not ids_a <= set(np.unique(preds.image_id).tolist()):
        raise ValueError('predictions miss Val-A images')

    pool, calib = split_pool_calib(val_a, args.calib_frac, args.split_seed)
    split = {'pool': pool, 'calib': calib, 'calib_frac': args.calib_frac, 'seed': args.split_seed,
             'val_a_md5': _md5_ids(ids_a), 'n_pool': len(pool), 'n_calib': len(calib)}
    save_json(out / 'splits/val_a_split.json', split)

    scores = image_scores(preds, pool)  # predictions only: no label is read for the pool
    save_json(out / 'scores/r1_pool.json', {'pred': PRED, 'pred_md5': md5_file(rel / PRED), 'images': scores})

    common = {'pool_md5': _md5_ids(pool), 'scores_md5': md5_file(out / 'scores/r1_pool.json'),
              'created_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}
    by_id, written = {s['image_id']: s for s in scores}, []

    def write(name: str, strategy: str, budget: int, seed: int | None, ids: list[int], opts: dict) -> None:
        if set(ids) & (ids_b | set(calib)):
            raise ValueError(f'{name}: selection leaks Val-B or calib images')
        save_json(out / f'selections/{name}.json',
                  {'strategy': strategy, 'budget': budget, 'seed': seed, 'opts': opts, **common, 'image_ids': ids,
                   'per_image_score': [by_id[i] for i in ids]})
        written.append(name)

    for budget in args.budgets:
        for strategy in args.strategies:
            for seed in (args.seeds if strategy == 'random' else [0]):
                opts = {'alpha': args.alpha} if strategy == 'margin_bal' else {}
                ids = sel.select(strategy, budget, scores, seed=seed, **opts)
                name = f'{strategy}_B{budget}' + (f'_s{seed}' if strategy == 'random' else '')
                write(name, strategy, budget, seed if strategy == 'random' else None, ids, opts)
    base = load_json(rel / f'variants/{args.base}/train_task_5_fix.json')
    oracle = sorted({i['orig_id'] for i in base['images'] if i.get('source') == 'real'} & set(pool))
    write('oracle', 'oracle', len(oracle), None, oracle, {'base': args.base, 'note': 'chosen with labels'})
    write('all', 'all', len(pool), None, list(pool), {})

    margins = np.array([s['min_margin'] for s in scores])
    summary = {'n_pool': len(pool), 'n_calib': len(calib), 'oracle_in_pool': len(oracle),
               'oracle_dropped_to_calib': len({i['orig_id'] for i in base['images'] if i.get('source') == 'real'}
                                              & set(calib)),
               'min_margin_percentiles': dict(zip(['p5', 'p25', 'p50', 'p75', 'p95'],
                                                  np.percentile(margins, [5, 25, 50, 75, 95]).round(4).tolist())),
               'images_without_counted_box': int(sum(s['n_counted'] == 0 for s in scores)),
               'selections': written}
    save_json(out / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    print(json.dumps(main(), indent=1))
