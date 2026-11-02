"""A9: second active-learning round. The first-round model (``al_<selection>``) scores the pool images not labelled
yet; the same strategy picks ``budget`` more; the labelled set grows to both rounds' images (selection
``<selection>_r2``) and its variant ``al_<selection>_r2`` is built. Train it with
``update_model --stage al --variant al_<selection>_r2`` (round2 recipe, from the first-round model).

    python -m tools.al_round2 --from al_margin_k_B500
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from autocheckout.active import select as sel
from autocheckout.active.uncertainty import image_scores
from autocheckout.datapaths import REPO, Layout
from autocheckout.io import load_json, md5_file, save_json
from autocheckout.predictions import load_predictions
from tools import al_build_variant

DEFAULT_CAMPAIGN = REPO / 'data/training/update-model/relearn-fix-v2'


def round2(data_root: Path | None, campaign: Path, first: str, budget: int | None = None) -> dict:
    release = Layout.at(data_root).relearn
    if not first.startswith('al_') or first.endswith('_r2'):
        raise ValueError(f'--from must be a first-round al_* run, got {first}')
    name = first.removeprefix('al_')
    chosen = load_json(release / f'active/selections/{name}.json')
    if chosen['strategy'] in ('oracle', 'all'):
        raise ValueError(f'{name}: references have no second round')
    pred = campaign / 'runs' / f'EXP-FB-FT-{first}' / 'task_5/pred_val.npz'
    if not pred.is_file():
        raise FileNotFoundError(f'{pred}: train {first} first')
    split = load_json(release / 'active/splits/val_a_split.json')
    labelled = list(chosen['image_ids'])
    remaining = sorted(set(split['pool']) - set(labelled))
    scores = image_scores(load_predictions(pred), remaining)
    budget = budget or chosen['budget']
    new = sel.select(chosen['strategy'], budget, scores, seed=chosen['seed'] or 0, **chosen.get('opts', {}))
    by_id = {s['image_id']: s for s in scores}
    target = f'{name}_r2'
    save_json(release / f'active/scores/{first}_remaining.json', {'pred': str(pred), 'images': scores})
    save_json(release / f'active/selections/{target}.json',
              {**{k: chosen[k] for k in ('strategy', 'seed', 'pool_md5')}, 'budget': len(labelled) + len(new),
               'opts': {**chosen.get('opts', {}), 'round': 2, 'from': first, 'round_budget': budget},
               'scores_md5': md5_file(release / f'active/scores/{first}_remaining.json'), 'pred_md5': md5_file(pred),
               'created_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
               'image_ids': labelled + new, 'new_image_ids': new,
               'per_image_score': chosen.get('per_image_score', []) + [by_id[i] for i in new]})
    return al_build_variant.build(data_root, target)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', type=Path, help='the data folder (default: ../autocheckout-data or $AC_DATA)')
    ap.add_argument('--campaign', type=Path, default=DEFAULT_CAMPAIGN,
                    help='campaign folder of update_model (default: data/training/update-model/relearn-fix-v2)')
    ap.add_argument('--from', dest='first', required=True, help='first-round run, e.g. al_margin_k_B500')
    ap.add_argument('--budget', type=int, help='images added in round 2 (default: the first round budget)')
    args = ap.parse_args(argv)
    return round2(args.data, args.campaign, args.first, args.budget)


if __name__ == '__main__':
    m = main()
    print(json.dumps({k: m[k] for k in ('variant', 'images', 'labelled_images', 'labelled_boxes', 'round')}))
