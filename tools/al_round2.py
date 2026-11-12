"""A9: second active-learning round. The first-round model (``al_<selection>``) scores the pool images not labelled
yet; the same strategy picks ``budget`` more; the labelled set grows to both rounds' images (selection
``<selection>_r2``) and its variant ``al_<selection>_r2`` is built. Train it with
``update_model --stage al --variant al_<selection>_r2`` (round2 recipe, from the first-round model).

    python -m tools.al_round2 --from al_margin_k_B500

Round 4 (docs/design/2026-11-12-dot-gpu-1, B1): ``--from al_<selection>_r3_real`` reads the round-2 selection, scores
the pool images left with the round-3 model, writes ``<selection>_r3`` and builds ``al_<selection>_r4`` with the
round-3 data options of ``real`` (rows_r3, repeat R4_REPEAT, v2_keep 0.5, no_face_v2 600, base e1).

    python -m tools.al_round2 --from al_margin_k_B500_r3_real --budget 500
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
R4_FROM = 'al_%s_r3_real'  # the only round-3 run a round-4 run may start from (12/11/2026, summary §6)
R4_REPEAT = 6  # 1,500 real images x 6 keep real at ~46% of the samples, as the 1,000 x 8 of r3_real
BOOKKEEPING = ('round', 'from', 'round_budget')  # opts of a selection that are not options of select()
R4_BUILD = dict(base='e1', rows=REPO / 'configs/relearn/rows_r3.json', repeat=R4_REPEAT, v2_keep=0.5, no_face_v2=600)


def _rounds(first: str) -> tuple[str, str, int, str, dict]:
    """(selection read, selection written, its round, variant built, build options) of a next round from ``first``."""
    if not first.startswith('al_') or first.endswith('_r2'):
        raise ValueError(f'--from must be a first-round al_* run or al_<selection>_r3_real, got {first}')
    if '_r3_' in first:
        selection = first.removeprefix('al_').rsplit('_r3_', 1)[0]
        if first != R4_FROM % selection:
            raise ValueError(f'round 4 starts from {R4_FROM % selection}, got {first}')
        return f'{selection}_r2', f'{selection}_r3', 3, f'al_{selection}_r4', dict(R4_BUILD)
    name = first.removeprefix('al_')
    return name, f'{name}_r2', 2, f'al_{name}_r2', {}


def round2(data_root: Path | None, campaign: Path, first: str, budget: int | None = None,
           build=al_build_variant.build) -> dict:
    release = Layout.at(data_root).relearn
    name, target, round_no, variant, options = _rounds(first)
    chosen = load_json(release / f'active/selections/{name}.json')
    if chosen['strategy'] in ('oracle', 'all'):
        raise ValueError(f'{name}: references have no second round')
    pred = campaign / 'runs' / f'EXP-FB-FT-{first}' / 'task_5/pred_val.npz'
    if not pred.is_file():
        raise FileNotFoundError(f'{pred}: train {first} first')
    labelled = list(chosen['image_ids'])
    written = release / f'active/selections/{target}.json'
    if written.is_file():  # shipped from the machine that selected (--select-only): kept only if nothing changed
        done = load_json(written)
        opts = done.get('opts', {})
        new = list(done.get('new_image_ids', []))
        stale = []
        if done.get('pred_md5') != md5_file(pred) or opts.get('from') != first:
            stale.append(f'selected from other predictions than {pred}')
        if done.get('image_ids') != labelled + new or done.get('strategy') != chosen['strategy'] or \
                done.get('pool_md5') != chosen['pool_md5']:
            stale.append(f'does not continue the current {name}')
        if budget is not None and opts.get('round_budget') != budget:
            stale.append(f"holds {opts.get('round_budget')} new images, --budget asks {budget}")
        if stale:
            raise ValueError(f'{written}: ' + '; '.join(stale) + '; delete it to select again')
        print(f'{written.name}: reused ({len(new)} new images selected on {done.get("created_at")})')
    else:
        split = load_json(release / 'active/splits/val_a_split.json')
        remaining = sorted(set(split['pool']) - set(labelled))
        scores = image_scores(load_predictions(pred), remaining)
        budget = budget or chosen['budget']
        strategy_opts = {k: v for k, v in chosen.get('opts', {}).items() if k not in BOOKKEEPING}
        new = sel.select(chosen['strategy'], budget, scores, seed=chosen['seed'] or 0, **strategy_opts)
        by_id = {s['image_id']: s for s in scores}
        save_json(release / f'active/scores/{first}_remaining.json', {'pred': str(pred), 'images': scores})
        save_json(written,
                  {**{k: chosen[k] for k in ('strategy', 'seed', 'pool_md5')}, 'budget': len(labelled) + len(new),
                   'opts': {**chosen.get('opts', {}), 'round': round_no, 'from': first, 'round_budget': budget},
                   'scores_md5': md5_file(release / f'active/scores/{first}_remaining.json'),
                   'pred_md5': md5_file(pred),
                   'created_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                   'image_ids': labelled + new, 'new_image_ids': new,
                   'per_image_score': chosen.get('per_image_score', []) + [by_id[i] for i in new]})
    if build is None:  # --select-only: the selection on a machine without the relearn release (checked, shipped)
        return {'variant': variant, 'selection': target, 'round': round_no, 'labelled_images': len(labelled) + len(new),
                'new_images': len(new), 'images': None, 'labelled_boxes': None}
    if options:
        return build(data_root, target, name=variant, **options)
    return build(data_root, target)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', type=Path, help='the data folder (default: ../autocheckout-data or $AC_DATA)')
    ap.add_argument('--campaign', type=Path, default=DEFAULT_CAMPAIGN,
                    help='campaign folder of update_model (default: data/training/update-model/relearn-fix-v2)')
    ap.add_argument('--from', dest='first', required=True,
                    help='first-round run (e.g. al_margin_k_B500) or al_<selection>_r3_real for round 4')
    ap.add_argument('--budget', type=int, help='images added in this round (default: the first round budget)')
    ap.add_argument('--select-only', action='store_true',
                    help='write the selection only, no variant (a Mac without the relearn release); the variant is '
                         'then built from the shipped selection (scripts/run_al.sh)')
    args = ap.parse_args(argv)
    return round2(args.data, args.campaign, args.first, args.budget, build=None if args.select_only else
                  al_build_variant.build)


if __name__ == '__main__':
    m = main()
    print(json.dumps({k: m[k] for k in ('variant', 'images', 'labelled_images', 'labelled_boxes', 'round')}))
