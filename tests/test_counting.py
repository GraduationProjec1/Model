import numpy as np
import pytest

from autocheckout.cl_metrics import load_coco
from autocheckout.counting import (
    THRESHOLD_GRID,
    class_group_scores,
    counting_scores,
    dedup_detections,
    gt_counts,
    oracle_by_level,
    pred_counts,
    scores_by_level,
    select_threshold,
)
from autocheckout.predictions import Predictions
from tests.coco_helpers import make_ann, make_gt, make_image, make_preds


def rpctool_score1(pred, gt, n):
    return np.sum(np.all(np.equal(gt, pred), axis=1)) / n


def rpctool_score2(pred, gt, n):
    return np.sum(np.abs(gt - pred)) / n


def rpctool_score4(pred, gt, k):
    class_cd = np.sum(np.abs(gt - pred), axis=0)
    class_gt = np.sum(gt, axis=0)
    return np.sum(np.divide(class_cd, class_gt)) / k


def rpctool_score5(pred, gt, k):
    minimum = np.minimum(pred, gt)
    maximum = np.maximum(pred, gt)
    return np.sum(np.divide(np.sum(minimum, axis=0), np.sum(maximum, axis=0))) / k


def test_counting_scores_match_rpctool_transcription_on_random_matrices():
    rng = np.random.default_rng(0)
    n_images, k = 12, 5
    gt = rng.integers(0, 4, size=(n_images, k))
    gt[0, :] = rng.integers(1, 4, size=k)
    pred = rng.integers(0, 4, size=(n_images, k))

    ours = counting_scores(pred, gt)
    assert ours["K_eff"] == k
    assert ours["cAcc"] == pytest.approx(rpctool_score1(pred, gt, n_images))
    assert ours["ACD"] == pytest.approx(rpctool_score2(pred, gt, n_images))
    assert ours["mCCD"] == pytest.approx(rpctool_score4(pred, gt, k))
    assert ours["mCIoU"] == pytest.approx(rpctool_score5(pred, gt, k))


def test_counting_scores_excludes_zero_gt_classes_from_class_averages():
    gt = np.array([[1, 0], [2, 0]])
    pred = np.array([[1, 5], [2, 0]])
    scores = counting_scores(pred, gt)
    assert scores["K"] == 2
    assert scores["K_eff"] == 1
    assert scores["mCCD"] == pytest.approx(0.0)
    assert scores["mCIoU"] == pytest.approx(1.0)
    assert scores["cAcc"] == pytest.approx(0.5)
    assert scores["ACD"] == pytest.approx((5 + 0) / 2)


def test_pred_counts_uses_top1_per_query():
    preds = make_preds([
        (1, 0, 2, 0.9, [0, 0, 10, 10]),
        (1, 0, 1, 0.3, [0, 0, 10, 10]),
        (1, 5, 0, 0.5, [0, 0, 10, 10]),
    ])
    counts = pred_counts(preds, image_ids=[1], labels=[0, 1, 2], threshold=0.0)
    assert counts.tolist() == [[1, 0, 1]]


def test_gt_counts_from_annotations():
    gt = make_gt(
        images=[make_image(1), make_image(2)],
        annotations=[
            make_ann(1, 1, 0, [0, 0, 5, 5]),
            make_ann(2, 1, 0, [10, 10, 5, 5]),
            make_ann(3, 2, 1, [0, 0, 5, 5]),
        ],
        num_labels=2,
    )
    coco_gt = load_coco(gt)
    counts = gt_counts(coco_gt, image_ids=[1, 2], labels=[0, 1])
    assert counts.tolist() == [[2, 0], [0, 1]]


def _threshold_val_fixture():
    gt = make_gt(
        images=[make_image(1), make_image(2)],
        annotations=[make_ann(1, 1, 0, [0, 0, 5, 5]), make_ann(2, 2, 0, [0, 0, 5, 5])],
        num_labels=1,
    )
    coco_gt = load_coco(gt)
    preds = make_preds([
        (1, 0, 0, 0.8, [0, 0, 5, 5]),
        (1, 1, 0, 0.2, [10, 10, 15, 15]),
        (2, 0, 0, 0.8, [0, 0, 5, 5]),
    ])
    return coco_gt, preds


def test_select_threshold_picks_the_known_best_threshold():
    coco_gt, preds = _threshold_val_fixture()
    threshold, scores = select_threshold(coco_gt, preds, labels=[0])
    assert threshold == pytest.approx(0.21)
    assert scores["cAcc"] == pytest.approx(1.0)


def test_select_threshold_tie_break_is_the_smallest_threshold():
    coco_gt, preds = _threshold_val_fixture()
    threshold, _ = select_threshold(coco_gt, preds, labels=[0], thresholds=[0.1, 0.3, 0.5, 0.9])
    assert threshold == pytest.approx(0.3)


def test_scores_by_level_and_oracle_by_level():
    gt = make_gt(
        images=[make_image(1, level="easy"), make_image(2, level="hard")],
        annotations=[make_ann(1, 1, 0, [0, 0, 5, 5]), make_ann(2, 2, 0, [0, 0, 5, 5])],
        num_labels=1,
    )
    coco_gt = load_coco(gt)
    preds = make_preds([(1, 0, 0, 0.9, [0, 0, 5, 5]), (2, 0, 0, 0.9, [0, 0, 5, 5])])

    by_level = scores_by_level(coco_gt, preds, labels=[0], threshold=0.5)
    assert by_level["overall"]["cAcc"] == pytest.approx(1.0)
    assert by_level["easy"]["cAcc"] == pytest.approx(1.0)
    assert by_level["hard"]["cAcc"] == pytest.approx(1.0)

    oracle = oracle_by_level(coco_gt, preds, labels=[0])
    assert oracle["overall"]["cAcc"] == pytest.approx(1.0)
    assert oracle["overall"]["threshold"] in THRESHOLD_GRID


def test_count_matrix_matches_a_plain_loop_on_random_rows():
    from autocheckout.counting import count_matrix

    rng = np.random.default_rng(0)
    n = 500
    preds = Predictions(image_id=rng.integers(1, 30, n), query=np.arange(n), label=rng.integers(0, 12, n),
                        score=rng.random(n), boxes=np.zeros((n, 4)))
    image_ids, labels = [3, 1, 7, 29, 100], [0, 5, 2, 11]
    expected = np.zeros((len(image_ids), len(labels)), dtype=np.int64)
    for img, lab, score in zip(preds.image_id, preds.label, preds.score, strict=True):
        if score >= 0.3 and img in image_ids and lab in labels:
            expected[image_ids.index(img), labels.index(lab)] += 1
    np.testing.assert_array_equal(count_matrix(preds, image_ids, labels, 0.3), expected)


def test_dedup_detections_keeps_the_best_row_per_object_across_classes_within_an_image():
    preds = Predictions(
        image_id=[1, 1, 1, 2, 1], query=[0, 1, 2, 0, 3], label=[0, 1, 0, 0, 2],
        score=[0.9, 0.8, 0.7, 0.6, 0.01],
        boxes=[[0, 0, 10, 10], [0, 0, 10, 9], [20, 20, 30, 30], [0, 0, 10, 10], [20, 20, 30, 30]],
    )
    kept = dedup_detections(preds, 0.5)
    assert sorted(zip(kept.image_id.tolist(), kept.query.tolist(), strict=True)) == [(1, 0), (1, 2), (2, 0)]
    assert len(dedup_detections(kept.subset(kept.score > 1), 0.5)) == 0


def test_class_group_scores_mccs_and_mccd_by_hand():
    gt = np.array([[2, 1, 0], [2, 0, 0]])
    pred = np.array([[1, 1, 3], [1, 1, 0]])
    scores = class_group_scores(pred, gt)
    assert scores["mCCS"] == pytest.approx((0.5 + 2.0) / 2)
    assert scores["mCCD"] == pytest.approx((0.5 + 1.0) / 2)
    assert scores["mCCD"] == pytest.approx(counting_scores(pred, gt)["mCCD"])
    assert (scores["K"], scores["K_eff"]) == (3, 2)
    assert np.isnan(class_group_scores(pred[:, 2:], gt[:, 2:])["mCCS"])


def test_select_group_thresholds_robust_avoids_low_threshold_collapse():
    from autocheckout.counting import select_group_thresholds, select_group_thresholds_robust

    coco_gt, preds = _threshold_val_fixture()
    # Grid [0.1, 0.25, 0.4, 0.6, 0.75]: thresholds 0.25..0.75 all give cAcc=1.0 on the fixture.
    grid = [0.1, 0.25, 0.4, 0.6, 0.75]
    small, _ = select_group_thresholds(coco_gt, preds, [[0]], thresholds=grid, tie_break="smallest")
    mid, _ = select_group_thresholds(coco_gt, preds, [[0]], thresholds=grid, tie_break="midpoint")
    rob, scores = select_group_thresholds_robust(coco_gt, preds, [[0]], thresholds=grid, folds=2, min_threshold=0.2)
    assert small == [pytest.approx(0.25)]
    assert mid[0] >= 0.4
    assert rob[0] >= 0.4
    assert scores["cAcc"] == pytest.approx(1.0)
