"""tools/sku_regression.py: per-SKU comparison of two models, as the gates of the improve mode read it."""

from pathlib import Path

import numpy as np
import pytest

from autocheckout.predictions import Predictions
from tools import sku_regression as sr

REPO = Path(__file__).resolve().parent.parent


def coco(objects_per_image):
    """Val file: image i holds the objects (label, x) given, each a 10 x 10 box at (x, 0)."""
    images, annotations = [], []
    for image_id, objects in enumerate(objects_per_image, start=1):
        images.append({'id': image_id})
        for label, x in objects:
            annotations.append({'id': len(annotations) + 1, 'image_id': image_id, 'category_id': label,
                                'bbox': [x, 0, 10, 10]})
    return {'images': images, 'annotations': annotations}


def preds(rows):
    """Counted boxes: (image_id, label, x) each a 10 x 10 box at (x, 0)."""
    rows = list(rows)
    return Predictions(image_id=np.array([r[0] for r in rows], dtype=np.int64),
                       query=np.arange(len(rows), dtype=np.int32),
                       label=np.array([r[1] for r in rows], dtype=np.int32),
                       score=np.ones(len(rows), dtype=np.float32),
                       boxes=np.array([[r[2], 0, r[2] + 10, 10] for r in rows], dtype=np.float32).reshape(-1, 4))


VAL = coco([[(0, 0), (1, 20)], [(1, 0)], [(2, 0)]])


def test_fixed_broken_pairs_and_falls():
    base = preds([(1, 0, 0), (1, 0, 20), (2, 1, 0), (3, 2, 0)])
    new = preds([(1, 0, 0), (1, 1, 20), (2, 1, 0), (3, 1, 0)])
    report = sr.compare(VAL, base, new, labels=range(3), rows=[0, 1])
    assert (report['cAcc_base'], report['cAcc_new']) == (2 / 3, 2 / 3)
    assert (report['fixed'], report['broken']) == (1, 1)
    assert report['pair_confusions_in_rows'] == {'base': 1, 'new': 0}
    assert report['other_objects_called_rows'] == {'base': 0, 'new': 1}
    shares = {s['label']: (s['base'], s['new']) for s in report['per_sku']}
    assert shares == {0: (1.0, 1.0), 1: (0.5, 1.0), 2: (1.0, 0.0)}
    assert report['top_pairs'][0]['pair'] in ([0, 1], [1, 2])
    assert report['largest_pair_in_rows'] == {'pair': [0, 1], 'base': 1, 'new': 0}


def test_largest_pair_in_rows_ignores_pairs_outside_the_rows_and_ranks_by_base():
    val = coco([[(0, 0)], [(1, 0), (1, 20)]])
    base = preds([(1, 1, 0), (2, 2, 0), (2, 2, 20)])
    new = preds([(1, 1, 0), (2, 1, 0), (2, 1, 20)])
    report = sr.compare(val, base, new, labels=range(3), rows=[0, 1])
    assert report['largest_pair_in_rows'] == {'pair': [0, 1], 'base': 1, 'new': 1}


def test_largest_pair_in_rows_is_none_without_a_confusion_in_the_rows():
    report = sr.compare(VAL, preds([(1, 0, 0), (1, 1, 20), (2, 1, 0), (3, 2, 0)]),
                        preds([(1, 0, 0), (1, 1, 20), (2, 1, 0), (3, 2, 0)]), labels=range(3), rows=[0, 1])
    assert report['largest_pair_in_rows'] is None


def test_fell_needs_both_a_recall_drop_and_lost_objects():
    """Gate rule of 12/11/2026: recall -0.03 AND >= 5 right objects lost; SKUs under 30 objects never fall."""
    def sku(objects, base, new):
        return {'label': 0, 'objects': objects, 'base': base, 'new': new}
    assert sr.fell(sku(150, 0.95, 0.90))
    assert not sr.fell(sku(150, 0.95, 0.925))
    assert not sr.fell(sku(100, 0.95, 0.91))
    assert sr.fell(sku(125, 0.95, 0.91))
    assert not sr.fell(sku(29, 1.0, 0.0))
    report = sr.compare(VAL, preds([(1, 0, 0), (1, 1, 20), (2, 1, 0), (3, 2, 0)]),
                        preds([(1, 0, 0), (1, 1, 20), (2, 1, 0)]), labels=range(3))
    assert report['falls'] == [] and report['falls_2pt'] == []


def test_a_missed_object_and_an_extra_box_make_the_image_wrong():
    report = sr.compare(VAL, preds([(1, 0, 0), (1, 1, 20), (2, 1, 0), (3, 2, 0)]),
                        preds([(1, 0, 0), (2, 1, 0), (2, 1, 50), (3, 2, 0)]), labels=range(3))
    assert (report['cAcc_base'], report['cAcc_new'], report['broken']) == (1.0, 1 / 3, 2)
    assert report['sign_test_p'] == pytest.approx(1.0)


def test_policy_vector_reads_group_thresholds():
    from autocheckout.taskcfg import TaskConfig

    from pdp_helpers import task_config_dict

    cfg = TaskConfig.from_dict(task_config_dict(sizes=(3, 2), reserved=0))
    policy = {'stages': {'2': {'group_thresholds': {'task_1': 0.1, 'task_2': 0.3}, 'nms_iou': 0.45}}}
    thresholds, nms = sr.policy_vector(policy, cfg, 2)
    assert thresholds.tolist() == pytest.approx([0.1, 0.1, 0.1, 0.3, 0.3]) and nms == 0.45


R1 = REPO / 'data/training/full-syn-r1/runs/EXP-FULL-SYN'
T6 = REPO / 'data/training/full-syn-relearn/scored/task6_alias_only'
VAL_FULL = REPO / 'data/processed/rpc_eval_official_v1/val_full.json'


@pytest.mark.skipif(not (R1.is_dir() and T6.is_dir() and VAL_FULL.is_file()), reason='round 1 / relearn data absent')
def test_reproduces_the_cacc_of_the_relearn_gate():
    report = sr.main(['--val', str(VAL_FULL), '--task-config', str(REPO / 'configs/tasks_100-4x25_seed0.json'),
                      '--stage', '5', '--base-pred', str(R1 / 'task_5/pred_val.npz'),
                      '--base-policy', str(R1 / 'calibration_count_m2.json'),
                      '--new-pred', str(T6 / 'task_5/pred_val.npz'),
                      '--new-policy', str(T6 / 'calibration_count_m2.json')])
    assert report['cAcc_base'] == pytest.approx(0.4368093922651934, abs=1e-12)
    assert report['cAcc_new'] == pytest.approx(0.4469958563535912, abs=1e-12)
    assert (report['fixed'], report['broken']) == (517, 458)


def test_per_sku_reports_extra_and_wrongly_called_boxes():
    from types import SimpleNamespace

    import numpy as np

    from tools import sku_regression as sr

    val = {'images': [{'id': 1}], 'annotations': [{'image_id': 1, 'category_id': 0, 'bbox': [0, 0, 10, 10]},
                                                    {'image_id': 1, 'category_id': 1, 'bbox': [20, 0, 10, 10]}]}

    def preds(labels, boxes):
        return SimpleNamespace(label=np.array(labels), boxes=np.array(boxes, dtype=float).reshape(-1, 4),
                               image_id=np.ones(len(labels), dtype=np.int64))

    base = preds([0, 1], [[0, 0, 10, 10], [20, 0, 30, 10]])
    new = preds([0, 0, 0], [[0, 0, 10, 10], [20, 0, 30, 10], [50, 50, 60, 60]])
    per = {s['label']: s for s in sr.compare(val, base, new, [0, 1])['per_sku']}
    assert (per[0]['over_base'], per[0]['over_new']) == (0, 2)
    assert (per[1]['over_base'], per[1]['over_new']) == (0, 0)
