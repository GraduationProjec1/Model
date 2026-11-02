"""A3: which ``budget`` images of the pool get labelled. Works on ``uncertainty.image_scores`` only: no label is
read here (the reference selections ``oracle`` and ``all`` are built by tools/al_select.py, outside this module).

- ``random``: shuffled with ``seed``;
- ``margin`` / ``margin_k``: smallest ``min_margin`` / ``mean_k_margin`` first;
- ``entropy``: largest ``max_entropy`` first;
- ``margin_bal``: ``margin`` order, but an image is taken only while every class of its ``top_labels`` has been
  taken fewer than ``q = ceil(alpha * budget / n_classes)`` times (n_classes: classes seen in the pool's
  ``top_labels``); when the pass ends short, the rest is filled in ``margin`` order.
Ties: more low boxes first, then the smaller image id.
"""

from __future__ import annotations

import math
import random
from collections import Counter

STRATEGIES = ('random', 'margin', 'margin_k', 'entropy', 'margin_bal')


def _order(scores: list[dict], key: str, reverse: bool = False) -> list[dict]:
    sign = -1 if reverse else 1
    return sorted(scores, key=lambda s: (sign * s[key], -s['n_low'], s['image_id']))


def select(strategy: str, budget: int, scores: list[dict], seed: int = 0, alpha: float = 1.5) -> list[int]:
    """Image ids picked by ``strategy``, in the order they were picked."""
    if strategy not in STRATEGIES:
        raise ValueError(f'unknown strategy {strategy!r} (one of {STRATEGIES})')
    if not 0 < budget <= len(scores):
        raise ValueError(f'budget {budget} must be in 1..{len(scores)} (pool size)')
    if strategy == 'random':
        ids = sorted(s['image_id'] for s in scores)
        random.Random(seed).shuffle(ids)
        return ids[:budget]
    if strategy == 'entropy':
        return [s['image_id'] for s in _order(scores, 'max_entropy', reverse=True)[:budget]]
    if strategy in ('margin', 'margin_k'):
        key = 'min_margin' if strategy == 'margin' else 'mean_k_margin'
        return [s['image_id'] for s in _order(scores, key)[:budget]]
    ordered = _order(scores, 'min_margin')
    n_classes = len({c for s in scores for c in s['top_labels']}) or 1
    q = math.ceil(alpha * budget / n_classes)
    taken, picked, chosen = Counter(), [], set()
    for s in ordered:
        if len(picked) == budget:
            break
        if all(taken[c] < q for c in s['top_labels']):
            picked.append(s['image_id'])
            chosen.add(s['image_id'])
            taken.update(s['top_labels'])
    for s in ordered:
        if len(picked) == budget:
            break
        if s['image_id'] not in chosen:
            picked.append(s['image_id'])
            chosen.add(s['image_id'])
    return picked
