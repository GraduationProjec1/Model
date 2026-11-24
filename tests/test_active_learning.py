"""autocheckout/active: Val-A split, uncertainty scores and selection strategies (docs/al/PLAN.md, A1-A3)."""

import math

import numpy as np
import pytest

from autocheckout.active import select as sel
from autocheckout.active.splits import split_pool_calib
from autocheckout.active.uncertainty import image_scores
from autocheckout.predictions import Predictions


def rpc_val(n):
    """n RPC-named val images, pairs of images share a scene, three levels, one SKU per image."""
    images = [{'id': i, 'file_name': f'20180824-13-43-{i % 60:02d}-{(i + 1) // 2}.jpg',
               'level': ['easy', 'medium', 'hard'][(i + 1) // 2 % 3]} for i in range(1, n + 1)]
    annotations = [{'id': i, 'image_id': i, 'category_id': i, 'bbox': [0, 0, 10, 10]} for i in range(1, n + 1)]
    return {'images': images, 'annotations': annotations, 'categories': [{'id': c} for c in range(n + 1)]}


def test_split_is_disjoint_complete_reproducible_and_keeps_groups():
    val = rpc_val(40)
    pool, calib = split_pool_calib(val, calib_frac=0.25, seed=3)
    ids = {i['id'] for i in val['images']}
    assert set(pool) | set(calib) == ids and not set(pool) & set(calib)
    assert 6 <= len(calib) <= 14
    from autocheckout.feedback_fix import scene_groups
    keys = scene_groups(val)
    assert not {keys[i] for i in pool} & {keys[i] for i in calib}
    assert split_pool_calib(val, calib_frac=0.25, seed=3) == (pool, calib)
    assert split_pool_calib(val, calib_frac=0.25, seed=4) != (pool, calib)


def test_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        split_pool_calib(rpc_val(4), calib_frac=1.0)


def preds(rows):
    """rows: (image, query, label, score, box)"""
    image, query, label, score, boxes = zip(*rows, strict=True)
    return Predictions(image, query, label, score, boxes)


B1, B2 = [0, 0, 10, 10], [50, 50, 60, 60]


def test_margin_uses_second_class_of_the_same_query():
    p = preds([(1, 0, 3, 0.9, B1), (1, 0, 7, 0.6, B1), (1, 1, 5, 0.8, B2), (1, 1, 2, 0.1, B2)])
    [s] = image_scores(p, [1])
    assert s['n_counted'] == 2 and s['n_low'] == 0
    assert s['min_margin'] == pytest.approx(0.3)
    assert s['mean_k_margin'] == pytest.approx((0.3 + 0.7) / 2)
    assert s['top_labels'] == [3, 7]


def test_box_without_second_class_has_margin_equal_to_score_and_zero_entropy():
    [s] = image_scores(preds([(1, 0, 3, 0.9, B1)]), [1])
    assert s['min_margin'] == pytest.approx(0.9) and s['max_entropy'] == 0.0 and s['top_labels'] == [3]


def test_entropy_is_highest_for_equal_scores():
    even = preds([(1, 0, c, 0.5, B1) for c in range(5)])
    sure = preds([(1, 0, 0, 0.9, B1)] + [(1, 0, c, 0.01, B1) for c in range(1, 5)])
    assert image_scores(even, [1])[0]['max_entropy'] == pytest.approx(1.0)
    assert image_scores(sure, [1])[0]['max_entropy'] < 0.3


def test_duplicate_box_is_removed_by_nms_and_low_boxes_are_counted():
    p = preds([(1, 0, 3, 0.9, B1), (1, 1, 3, 0.85, B1), (1, 2, 4, 0.2, B2), (1, 3, 4, 0.01, [80, 80, 90, 90])])
    [s] = image_scores(p, [1])
    assert s['n_counted'] == 1 and s['n_low'] == 1


def test_images_without_counted_boxes():
    p = preds([(1, 0, 3, 0.2, B1), (2, 0, 3, 0.01, B1)])
    s1, s2, s3 = image_scores(p, [1, 2, 3])
    assert s1['min_margin'] == 0.0 and s2['min_margin'] == 1.0 and s3['min_margin'] == 1.0
    assert [s['image_id'] for s in (s1, s2, s3)] == [1, 2, 3]


def test_k_larger_than_box_count():
    [s] = image_scores(preds([(1, 0, 3, 0.9, B1), (1, 0, 4, 0.5, B1)]), [1], k=10)
    assert s['mean_k_margin'] == pytest.approx(0.4)


def pool(n=20):
    rng = np.random.default_rng(0)
    return [{'image_id': i, 'min_margin': float(rng.random()), 'mean_k_margin': float(rng.random()),
             'max_entropy': float(rng.random()), 'n_low': int(rng.integers(0, 3)), 'n_counted': 3,
             'top_labels': [i % 4, (i + 1) % 4]} for i in range(1, n + 1)]


@pytest.mark.parametrize('strategy', sel.STRATEGIES)
def test_every_strategy_returns_budget_distinct_pool_ids(strategy):
    scores = pool()
    picked = sel.select(strategy, 7, scores, seed=1)
    assert len(picked) == len(set(picked)) == 7 and set(picked) <= {s['image_id'] for s in scores}


def test_random_is_reproducible_and_depends_on_seed():
    scores = pool()
    assert sel.select('random', 7, scores, seed=1) == sel.select('random', 7, scores, seed=1)
    assert sel.select('random', 7, scores, seed=1) != sel.select('random', 7, scores, seed=2)


def test_margin_and_entropy_orders():
    scores = pool()
    by_id = {s['image_id']: s for s in scores}
    m = [by_id[i]['min_margin'] for i in sel.select('margin', 20, scores)]
    e = [by_id[i]['max_entropy'] for i in sel.select('entropy', 20, scores)]
    assert m == sorted(m) and e == sorted(e, reverse=True)


def test_margin_bal_caps_each_class_before_filling():
    scores = [{'image_id': i, 'min_margin': i / 100, 'mean_k_margin': 0, 'max_entropy': 0, 'n_low': 0,
               'n_counted': 1, 'top_labels': [0] if i <= 10 else [i]} for i in range(1, 21)]
    picked = sel.select('margin_bal', 6, scores, alpha=1.0)
    q = math.ceil(1.0 * 6 / 11)
    assert sum(1 for i in picked if i <= 10) == q
    assert picked[:q] == [1] and len(picked) == 6


def test_budget_and_strategy_errors():
    with pytest.raises(ValueError):
        sel.select('margin', 21, pool())
    with pytest.raises(ValueError):
        sel.select('nope', 3, pool())


def test_selection_never_needs_labels():
    scores = [{k: v for k, v in s.items()} for s in pool()]
    assert all('annotations' not in s and 'category_id' not in s for s in scores)
    sel.select('margin_bal', 5, scores)
