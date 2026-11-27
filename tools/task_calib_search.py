"""N1 (docs/ai/planning/2026-11-26-plan-N1-N5-hieu-chinh-task-va-loss.md §3): per-task logit calibration fitted on
Val-A-calib, scored on Val-B once, from the logits of ``pdp/main.py --dump_logits 1`` (no training).

For each variant (``base`` = no calibration, ``bias_only``, ``scale_bias``):
  1. fit on the GT objects of the calib images matched to a counted query (NMS 0.45, no threshold);
  2. stability: split calib in two halves (``--seed``), fit and choose the thresholds on one half, score the other,
     both ways; ``stable`` = the calibration beats ``base`` on both halves;
  3. fit again on all of calib, re-choose the 5 group thresholds on calib (the sigmoid changed) and score Val-B
     with ``tools/al_score.score``, plus the sign test against ``base`` and the swaps left on Val-B.
A fit image that is a Val-B image is an error. Writes ``results/task_calib/<name>.json`` and, per variant,
``<name>.<variant>.json``, the file ``pdp/main.py --task_calib`` reads.

    python -m tools.task_calib_search --name r4 --logits runs/<run>/task_5/logits_val.npz [--data ../autocheckout-data]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from autocheckout.counting import label_thresholds
from autocheckout.datapaths import Layout
from autocheckout.feedback_fix import subset
from autocheckout.io import load_json, save_json
from autocheckout.logit_dump import LogitDump, load_logits
from autocheckout.predictions import Predictions
from autocheckout.task_calib import DEFAULT_L2, MODES, TaskCalib, assert_disjoint, cross_entropy, fit_task_calib
from autocheckout.taskcfg import TaskConfig
from tools.al_score import NMS, STAGE, UNLEARNED_IOU, group_labels, score
from tools.count_policy_search import sign_test
from tools.error_breakdown import breakdown
from tools.swap_rank import Matched, check_dump, label_maps, match_queries

RESULTS = Path('results/task_calib')
MIN_GAIN, MAX_SIGN_P = 0.005, 0.1


def _rows(matched: Matched, ids) -> Matched:
    keep = np.isin(matched.image_id, list(ids))
    return Matched(matched.image_id[keep], matched.true_label[keep], matched.pred_label[keep], matched.logits[keep])


def _fit(matched: Matched, groups, mode: str | None, l2: float) -> TaskCalib | None:
    return None if mode is None else fit_task_calib(matched.logits, matched.true_label, groups, mode=mode, l2=l2)


class Search:
    """The dump, the GT and the scoring rules shared by every variant."""

    def __init__(self, dump: LogitDump, val_full: dict, cfg: TaskConfig, calib: list[int], target: list[int],
                 l2: float = DEFAULT_L2):
        assert_disjoint(calib, target)
        self.dump, self.val_full, self.cfg, self.l2 = dump, val_full, cfg, l2
        self.calib, self.target = sorted(calib), sorted(target)
        self.groups = group_labels(cfg, STAGE)
        self.unlearned = UNLEARNED_IOU if STAGE < len(cfg.data_tasks) else None
        self.task_of, self.supercategory = label_maps(cfg, dump.num_classes)
        self.matched = match_queries(dump, val_full, self.calib, NMS)

    def predictions(self, calib: TaskCalib | None) -> Predictions:
        return self.dump.predictions(calib, producer='task_calib' if calib else 'logit_dump')

    def score(self, preds: Predictions, choose_on: list[int], target: list[int]) -> dict:
        return score(self.val_full, self.cfg, preds, choose_on, target, groups=self.groups,
                     unlearned_iou=self.unlearned)

    def halves(self, mode: str | None, seed: int) -> list[dict]:
        """Fit and thresholds on one half of calib, cAcc on the other, both ways."""
        order = np.random.default_rng(seed).permutation(self.calib)
        half = len(order) // 2
        parts = (sorted(order[:half].tolist()), sorted(order[half:].tolist()))
        out = []
        for fit_ids, check_ids in (parts, parts[::-1]):
            calib = _fit(_rows(self.matched, fit_ids), self.groups[0], mode, self.l2)
            out.append({'fit_images': len(fit_ids), 'check_images': len(check_ids),
                        'cAcc': self.score(self.predictions(calib), fit_ids, check_ids)['cAcc_m2']})
        return out

    def swaps(self, preds: Predictions, thresholds: list[float]) -> dict:
        """Errors left on Val-B with the chosen thresholds; swaps split by task and supercategory."""
        report = breakdown(subset(self.val_full, self.target), preds, label_thresholds(self.groups[0], thresholds),
                           NMS, set())
        kinds = Counter()
        for image in report['per_image']:
            for kind, true, win in image['errors']:
                if kind != 'swap':
                    continue
                kinds['cross_task'] += self.task_of[win] != self.task_of[true]
                same = self.supercategory.get(win) is not None
                kinds['same_supercategory'] += same and self.supercategory[win] == self.supercategory.get(true)
        return {**report['errors_by_kind'], 'swap_cross_task': kinds['cross_task'],
                'swap_same_supercategory': kinds['same_supercategory']}

    def variant(self, mode: str | None, seed: int) -> tuple[dict, TaskCalib | None, np.ndarray]:
        calib = _fit(self.matched, self.groups[0], mode, self.l2)
        preds = self.predictions(calib)
        result = self.score(preds, self.calib, self.target)
        right = np.isin(self.target, result.pop('right_images'))
        thresholds = list(result['thresholds'].values())
        reference = calib or TaskCalib.identity(self.groups[0])
        out = {**result, 'calibration': None if calib is None else calib.to_dict(),
               'calib_cross_entropy': cross_entropy(reference, self.matched.logits, self.matched.true_label),
               'halves': self.halves(mode, seed), 'valb_errors': self.swaps(preds, thresholds)}
        return out, calib, right


def decide(variant: dict) -> str:
    """§3.4: adopt when Val-B gains >= 0.5 point with sign p < 0.1 and both calib halves gain."""
    if variant['delta_vs_base'] >= MIN_GAIN and variant['sign_p'] < MAX_SIGN_P and variant['stable_halves']:
        return 'adopt'
    return 'reject'


def run(search: Search, seed: int = 0) -> dict:
    variants, files = {}, {}
    base, _, base_right = search.variant(None, seed)
    variants['base'] = {**base, 'delta_vs_base': 0.0, 'fixed_vs_base': 0, 'broken_vs_base': 0, 'sign_p': 1.0}
    for mode in MODES:
        result, calib, right = search.variant(mode, seed)
        fixed, broken = int((right & ~base_right).sum()), int((~right & base_right).sum())
        result.update(delta_vs_base=result['cAcc_m2'] - base['cAcc_m2'], fixed_vs_base=fixed, broken_vs_base=broken,
                      sign_p=sign_test(fixed, broken))
        result['stable_halves'] = all(v['cAcc'] > b['cAcc'] for v, b in zip(result['halves'], base['halves'],
                                                                             strict=True))
        result['decision'] = decide(result)
        variants[mode], files[mode] = result, calib
    return {'matched_calib_objects': len(search.matched), 'variants': variants, '_files': files}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--name', required=True)
    ap.add_argument('--logits', type=Path, required=True, help='logits_val.npz of the model (all of val)')
    ap.add_argument('--data', type=Path, help='the data folder (default: ../autocheckout-data or $AC_DATA)')
    ap.add_argument('--eval-dir', type=Path, help='val_full.json (default: <data>/eval)')
    ap.add_argument('--task-config', type=Path, help="default: the eval dir's task_config.json")
    ap.add_argument('--l2', type=float, default=DEFAULT_L2, help='pull of (a, b) towards (1, 0)')
    ap.add_argument('--seed', type=int, default=0, help='split of calib into two halves')
    ap.add_argument('--out', type=Path, help=f'default: {RESULTS}/<name>.json')
    args = ap.parse_args(argv)
    data = Layout.at(args.data)
    eval_dir = args.eval_dir or data.eval
    val_path = eval_dir / 'val_full.json'
    dump = load_logits(args.logits)
    check_dump(dump, val_path)
    calib = load_json(data.relearn / 'active/splits/val_a_split.json')['calib']
    target = [i['id'] for i in load_json(data.relearn / 'meta/val_b.json')['images']]
    search = Search(dump, load_json(val_path), TaskConfig.load(args.task_config or eval_dir / 'task_config.json'),
                    calib, target, l2=args.l2)
    result = run(search, args.seed)
    out = args.out or RESULTS / f'{args.name}.json'
    for mode, calibration in result.pop('_files').items():
        path = out.with_name(f'{out.stem}.{mode}.json')
        calibration.save(path)
        result['variants'][mode]['file'] = str(path)
    result = {'name': args.name, 'logits': str(args.logits), 'l2': args.l2, 'seed': args.seed, **result}
    save_json(out, result, indent=1)
    return result


if __name__ == '__main__':
    r = main()
    columns = ('cAcc_m2', 'delta_vs_base', 'fixed_vs_base', 'broken_vs_base', 'sign_p', 'stable_halves', 'decision')
    table = {}
    for name, v in r['variants'].items():
        errors = v['valb_errors']
        table[name] = {**{k: v.get(k) for k in columns}, 'swaps': errors['swap'],
                       'cross': errors['swap_cross_task'], 'same_super': errors['swap_same_supercategory']}
    print(json.dumps(table, indent=1))
