from collections import Counter

import numpy as np

from tools.error_breakdown import explain_image
from tools.sku_regression import match_image

E = np.zeros(0, dtype=np.int64), np.zeros((0, 4))


def b(x):
    return [x, 0, x + 10, 10]


def test_each_kind_is_named():
    gt_labels = np.array([1, 2, 3, 4, 5])
    gt_boxes = np.array([b(0), b(20), b(40), b(60), b(80)], float)
    labels = np.array([1, 9, 1, 7, 7])
    boxes = np.array([b(0), b(20), [1, 0, 11, 10], b(85), b(200)], float)
    low = np.array([3]), np.array([b(40)], float)
    nms = np.array([4]), np.array([b(60)], float)
    errors = explain_image(gt_labels, gt_boxes, labels, boxes, *low, *nms)
    assert sorted(errors) == sorted([('swap', 2, 9), ('miss_threshold', 3, -1), ('miss_nms', 4, -1),
                                     ('miss', 5, -1), ('extra_dup', -1, 1), ('extra_shift', -1, 7),
                                     ('extra_bg', -1, 7)])


def test_agrees_with_sku_regression_matching():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n, m = rng.integers(0, 6, 2)
        gt_labels, labels = rng.integers(0, 3, n), rng.integers(0, 3, m)
        gt_boxes = np.array([b(x) for x in rng.integers(0, 40, n)], float).reshape(-1, 4)
        boxes = np.array([b(x) for x in rng.integers(0, 40, m)], float).reshape(-1, 4)
        errors = explain_image(gt_labels, gt_boxes, labels, boxes, *E, *E)
        ref = match_image(gt_labels, gt_boxes, labels, boxes)
        assert Counter((e[1], e[2]) for e in errors if e[0] == 'swap') == ref['swapped']
        assert Counter(e[1] for e in errors if e[0].startswith('miss')) == ref['missed']
        assert Counter(e[2] for e in errors if e[0].startswith('extra')) == ref['extra']
