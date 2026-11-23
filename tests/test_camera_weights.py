"""A2 (AN/cyclegan/03_QUYET_DINH_VA_PLAN.md): per-SKU camera weights from Val-A best-match counts,
shrunk towards the group mean (strength fitted on the data) and floored at 3 %."""

import numpy as np
import pytest

from autocheckout.camera_weights import apply_floor, build_skus, fit_strength, shrunk_weights


def test_floor_lifts_small_cameras_and_keeps_the_sum():
    w = apply_floor({0: 0.90, 1: 0.08, 2: 0.0, 3: 0.02}, 0.03)
    assert sum(w.values()) == pytest.approx(1.0)
    assert min(w.values()) == pytest.approx(0.03)
    assert w[1] > w[3] == pytest.approx(0.03) and w[0] < 0.90     # the extra share comes from the cameras above the floor
    assert w[1] / w[0] == pytest.approx(0.08 / 0.90)               # cameras above the floor keep their proportions


def test_floor_leaves_weights_that_already_pass():
    w = {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}
    assert apply_floor(w, 0.03) == pytest.approx(w)


def _draw(rng, means, strength, n_skus, n_obj):
    counts = {}
    for i in range(n_skus):
        p = rng.dirichlet(np.array(means) * strength)
        counts[i] = dict(enumerate(rng.multinomial(n_obj, p)))
    return counts


def test_fit_strength_recovers_the_spread_between_skus():
    rng = np.random.default_rng(0)
    counts = _draw(rng, [0.4, 0.3, 0.2, 0.1], 8.0, 400, 150)
    group = {i: "g" for i in counts}
    assert 5.0 < fit_strength(counts, group) < 12.0


def test_skus_with_few_objects_move_more_towards_the_group():
    counts = {1: {0: 9, 1: 1, 2: 0, 3: 0}, 2: {0: 90, 1: 10, 2: 0, 3: 0}}
    counts.update({i: {0: 25, 1: 25, 2: 25, 3: 25} for i in range(3, 13)})
    group = {i: "g" for i in counts}
    w = shrunk_weights(counts, group, strength=20.0, floor=0.0)
    assert w[1][0] < w[2][0] < 0.9                                 # same share, fewer objects -> closer to the mean
    assert all(sum(v.values()) == pytest.approx(1.0) for v in w.values())


def test_group_means_are_per_group():
    counts = {1: {0: 50, 2: 0}, 2: {0: 50, 2: 0}, 3: {0: 0, 2: 50}, 4: {0: 0, 2: 50}}
    group = {1: "bottle", 2: "bottle", 3: "flat", 4: "flat"}
    w = shrunk_weights(counts, group, strength=20.0, floor=0.03)
    assert w[1][0] > 0.9 and w[3][2] > 0.9
    assert w[1][2] == pytest.approx(0.03)


def test_build_skus_floors_only_cameras_with_cutouts_and_marks_standing():
    counts = {1: {0: 60, 1: 20, 2: 0, 3: 20}, 2: {0: 0, 1: 30, 2: 70, 3: 0}, 3: {0: 50, 1: 30, 2: 0, 3: 20}}
    groups = {1: "drink", 2: "candy", 3: "drink"}
    shapes = {1: "upright", 2: "flat", 3: "upright"}
    available = {1: {0, 1, 2, 3}, 2: {0, 1, 2, 3}, 3: {0, 1, 3}}           # SKU 3 has no camera-2 cutout
    table = build_skus(counts, groups, shapes, available, floor=0.03, strength=5.0)
    by_id = {s["rpc_category_id"]: s for s in table["skus"]}
    assert table["strength"] == 5.0 and table["floor"] == 0.03
    assert by_id[1]["standing_cameras"] == [2] and "standing_cameras" not in by_id[2]
    assert by_id[1]["camera_weights"]["2"] == pytest.approx(0.03)
    assert "2" not in by_id[3]["camera_weights"]
    for s in table["skus"]:
        assert sum(s["camera_weights"].values()) == pytest.approx(1.0)
        assert min(s["camera_weights"].values()) >= 0.03 - 1e-9
        assert s["counts"] == {str(c): n for c, n in counts[s["rpc_category_id"]].items()}
