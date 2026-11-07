"""Round 3 (docs/design/2026-11-07-vong-3, §6.7): did a relearned model learn the real images it trained on?
cAcc-M2 on the Val-A images of its AL selection (trained), on the Val-A pool images it did not train on and on calib,
all with the Val-A-calib group thresholds of tools/al_score.py (NMS 0.45). Trained ≈ untrained means underfit;
much higher means the model learns them and the limit is generalisation. AL strategies pick hard images, so compare
the trained number of an AL run with a random run.

    python -m tools.fit_check --pred <campaign>/runs/EXP-FB-FT-<run>/task_5/pred_val.npz --selection margin_k_B500_r2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autocheckout.datapaths import Layout
from autocheckout.io import load_json
from autocheckout.predictions import Predictions, load_predictions
from autocheckout.taskcfg import TaskConfig
from tools.al_score import STAGE, count_on, score


def fit_scores(val_full: dict, cfg: TaskConfig, preds: Predictions, split: dict, trained_ids,
               thresholds: list[float] | None = None) -> dict:
    """cAcc of the trained / untrained pool / calib images of Val-A. ``thresholds`` (5 group thresholds) default to
    the ones chosen on calib, as al_score does."""
    calib, pool, trained = sorted(split['calib']), set(split['pool']), set(trained_ids)
    if trained & set(calib):
        raise ValueError('the selection trains on calib images')
    if not np.isin(sorted(pool | set(calib)), preds.image_id).all():
        raise ValueError('no predictions for every Val-A image (scored/<run>/task_5 holds Val-B only: use runs/)')
    if thresholds is None:
        thresholds = [score(val_full, cfg, preds, calib, calib)['thresholds'][f'task_{t}'] for t in range(1, STAGE + 1)]
    out = {'fit_thresholds': list(map(float, thresholds))}
    for key, ids in (('trained', trained & pool), ('pool_untrained', pool - trained), ('calib', set(calib))):
        r = count_on(val_full, cfg, preds, sorted(ids), thresholds) if ids else {'cAcc_m2': float('nan')}
        out[f'cAcc_{key}'], out[f'images_{key}'] = r['cAcc_m2'], len(ids)
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', type=Path, help='the data folder (default: ../autocheckout-data or $AC_DATA)')
    ap.add_argument('--pred', type=Path, required=True, help='task_5/pred_val.npz of the model (all of val)')
    ap.add_argument('--selection', required=True, help='name in relearn/active/selections/ the model trained on')
    ap.add_argument('--active', type=Path, help='folder with splits/ and selections/ (default: <data>/relearn/active)')
    args = ap.parse_args(argv)
    data = Layout.at(args.data)
    active = args.active or data.relearn / 'active'
    return fit_scores(load_json(data.eval / 'val_full.json'), TaskConfig.load(data.eval / 'task_config.json'),
                      load_predictions(args.pred), load_json(active / 'splits/val_a_split.json'),
                      load_json(active / 'selections' / f'{args.selection}.json')['image_ids'])


if __name__ == '__main__':
    print(json.dumps(main(), indent=1))
