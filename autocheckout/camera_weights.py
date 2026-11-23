"""Per-SKU camera weights for synth (A2 in AN/cyclegan/03_QUYET_DINH_VA_PLAN.md, decided 23/11).

Input: for every SKU, how many of its Val-A objects look most like the cutouts of each camera (counts).
A SKU's own shares are noisy when it has few objects, so they are shrunk towards the mean of its group
(supercategory): w = (count + strength * group_mean) / (n + strength). ``strength`` is not hand-picked: it is the
Dirichlet-multinomial maximum-likelihood fit over all SKUs. Every camera then keeps at least ``floor`` (3 %,
the user's choice) so rare views still appear.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.special import gammaln

EPS = 1e-3     # group mean of a camera no SKU of the group matched: tiny but positive, so the likelihood exists


def apply_floor(weights: dict, floor: float) -> dict:
    """Raise every weight to at least ``floor``; the cameras above keep their proportions; the sum stays 1."""
    w = {c: float(v) for c, v in weights.items()}
    total = sum(w.values())
    w = {c: v / total for c, v in w.items()}
    fixed: set = set()
    while True:
        low = {c for c, v in w.items() if c not in fixed and v < floor}
        if not low:
            return w
        fixed |= low
        free = [c for c in w if c not in fixed]
        rest = 1.0 - floor * len(fixed)
        mass = sum(w[c] for c in free)
        w = {c: (floor if c in fixed else w[c] * rest / mass) for c in w}


def _group_means(counts: dict, group: dict) -> dict:
    pooled = defaultdict(lambda: defaultdict(float))
    for sku, by_cam in counts.items():
        for cam, n in by_cam.items():
            pooled[group[sku]][cam] += n
    means = {}
    for g, by_cam in pooled.items():
        cams = sorted({c for s in counts for c in counts[s]})
        v = np.array([by_cam.get(c, 0.0) for c in cams]) + EPS
        means[g] = dict(zip(cams, v / v.sum()))
    return means


def _loglik(strength: float, counts: dict, group: dict, means: dict) -> float:
    total = 0.0
    for sku, by_cam in counts.items():
        cams = sorted(means[group[sku]])
        x = np.array([by_cam.get(c, 0) for c in cams], float)
        a = strength * np.array([means[group[sku]][c] for c in cams])
        total += gammaln(a.sum()) - gammaln(a.sum() + x.sum()) + (gammaln(a + x) - gammaln(a)).sum()
    return total


def fit_strength(counts: dict, group: dict) -> float:
    """Maximum-likelihood ``strength`` (Dirichlet concentration around the group means) on a log grid."""
    means = _group_means(counts, group)
    grid = np.exp(np.linspace(np.log(0.1), np.log(1e4), 400))
    ll = [_loglik(k, counts, group, means) for k in grid]
    return float(grid[int(np.argmax(ll))])


def shrunk_weights(counts: dict, group: dict, *, strength: float, floor: float) -> dict:
    means = _group_means(counts, group)
    out = {}
    for sku, by_cam in counts.items():
        m = means[group[sku]]
        n = sum(by_cam.values())
        raw = {c: (by_cam.get(c, 0) + strength * m[c]) / (n + strength) for c in m}
        out[sku] = apply_floor(raw, floor) if floor else raw
    return out


def build_skus(counts: dict, groups: dict, shapes: dict, available: dict, *, floor: float,
               strength: float | None = None) -> dict:
    """The SKU table synth_v2 reads (``skus.json``): camera weights, shape group and, for upright SKUs, the
    standing pose (camera 2, top view: only the cap shows). A camera without cutouts for the SKU gets no weight."""
    if strength is None:
        strength = fit_strength(counts, groups)
    shrunk = shrunk_weights(counts, groups, strength=strength, floor=0.0)
    skus = []
    for sku in sorted(counts):
        kept = {c: w for c, w in shrunk[sku].items() if c in available[sku]}
        weights = apply_floor(kept, floor)
        row = {"rpc_category_id": sku, "group": groups[sku], "shape_group": shapes[sku],
               "camera_weights": {str(c): round(w, 4) for c, w in sorted(weights.items())},
               "counts": {str(c): int(n) for c, n in sorted(counts[sku].items())}}
        if shapes[sku] == "upright":
            row["standing_cameras"] = [2]
        skus.append(row)
    return {"method": "Val-A best-match counts, shrunk to the group mean, floor (autocheckout/camera_weights.py)",
            "strength": strength, "floor": floor, "skus": skus}
