"""A1: Val-A split into the pool (images the selector may pick for labelling) and calib (images that choose the
counting thresholds). Disjoint, so a threshold is never chosen on an image the model was trained on."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from autocheckout.feedback_fix import scene_groups

Coco = dict[str, Any]


def split_pool_calib(val_a: Coco, calib_frac: float = 0.2, seed: int = 0) -> tuple[list[int], list[int]]:
    """(pool ids, calib ids): whole scene groups on one side, about ``calib_frac`` of the images of every level in
    calib. Within a level the groups are shuffled, then each goes to the side furthest below its share."""
    if not 0 < calib_frac < 1:
        raise ValueError('calib_frac must be in (0, 1)')
    keys, level = scene_groups(val_a), {i['id']: i.get('level', '') for i in val_a['images']}
    groups: dict[str, list[int]] = defaultdict(list)
    for image_id, key in keys.items():
        groups[key].append(image_id)
    by_level: dict[str, list[str]] = defaultdict(list)
    for key in sorted(groups):
        by_level[level[groups[key][0]]].append(key)
    rng, pool, calib = random.Random(seed), [], []
    shares = (1 - calib_frac, calib_frac)
    for name in sorted(by_level):
        keys_of_level = by_level[name]
        rng.shuffle(keys_of_level)
        counts = [0, 0]
        for key in sorted(keys_of_level, key=lambda k: -len(groups[k])):
            total = counts[0] + counts[1] + len(groups[key])
            side = 0 if counts[0] - shares[0] * total <= counts[1] - shares[1] * total else 1
            (pool, calib)[side].extend(groups[key])
            counts[side] += len(groups[key])
    return sorted(pool), sorted(calib)
