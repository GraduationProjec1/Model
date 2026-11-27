"""N1 (plan N1-N5 §3): per-task logit calibration (autocheckout/task_calib.py) and its search on Val-A-calib
(tools/task_calib_search.py)."""

import json

import numpy as np
import pytest
import torch

from autocheckout.logit_dump import LogitDump, save_logits
from autocheckout.task_calib import TaskCalib, assert_disjoint, fit_task_calib
from autocheckout.taskcfg import TaskConfig
from tools import task_calib_search
from tools.task_calib_search import Search, run

from pdp_helpers import task_config_dict

GROUPS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))


def test_identity_returns_the_same_logits_and_never_changes_the_input():
    logits = np.random.default_rng(0).normal(size=(4, 9)).astype(np.float32)
    before = logits.copy()
    out = TaskCalib.identity(GROUPS).apply(logits)
    assert np.array_equal(out, before) and out is not logits
    calibrated = TaskCalib((2.0, 1.0, 1.0), (-1.0, 0.5, 0.0), GROUPS).apply(logits)
    assert np.array_equal(logits, before)
    np.testing.assert_allclose(calibrated[:, :3], 2 * before[:, :3] - 1)
    np.testing.assert_allclose(calibrated[:, 3:6], before[:, 3:6] + 0.5)


def test_apply_works_on_tensors_too():
    logits = torch.randn(2, 5, 9)
    out = TaskCalib((1.0, 1.0, 1.0), (0.0, -2.0, 0.0), GROUPS).apply(logits)
    assert torch.allclose(out[..., 3:6], logits[..., 3:6] - 2) and torch.equal(out[..., :3], logits[..., :3])


def sampled(shift_group=1, shift=1.0, n=6000, seed=0):
    """Labels drawn from softmax(z); the logits seen by the fit are z with ``shift`` added to one task."""
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=2.0, size=(n, 9))
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(9, p=row) for row in p])
    biased = z.copy()
    biased[:, list(GROUPS[shift_group])] += shift
    return biased, labels


@pytest.mark.parametrize('mode', ['bias_only', 'scale_bias'])
def test_fit_finds_the_shift_added_to_a_task(mode):
    logits, labels = sampled()
    calib = fit_task_calib(logits, labels, GROUPS, mode=mode, l2=1e-4)
    assert calib.shift[1] == pytest.approx(-1.0, abs=0.15)
    assert calib.shift[0] == pytest.approx(0.0, abs=0.15)
    assert calib.scale[-1] == 1.0 and calib.shift[-1] == 0.0
    if mode == 'bias_only':
        assert calib.scale == (1.0, 1.0, 1.0)
    else:
        assert calib.scale[1] == pytest.approx(1.0, abs=0.1)


def test_fit_removes_swaps_caused_by_a_shifted_task():
    rng = np.random.default_rng(1)
    logits = rng.normal(scale=0.05, size=(300, 9))
    labels = rng.integers(0, 9, 300)
    logits[np.arange(300), labels] += 0.5
    logits[:, 3:6] += 1.0
    assert (logits.argmax(1) != labels).sum() > 150
    calib = fit_task_calib(logits, labels, GROUPS, mode='bias_only')
    assert (calib.apply(logits).argmax(1) != labels).sum() == 0


def test_fit_images_must_not_be_scored():
    assert_disjoint([1, 2], [3, 4])
    with pytest.raises(ValueError, match='fit images are scored'):
        assert_disjoint([1, 2, 3], [3, 4])


def test_json_round_trip(tmp_path):
    calib = TaskCalib((1.5, 0.8, 1.0), (-0.3, 0.2, 0.0), GROUPS, 'scale_bias')
    assert TaskCalib.from_dict(json.loads(json.dumps(calib.to_dict()))) == calib
    calib.save(tmp_path / 'c.json')
    assert TaskCalib.load(tmp_path / 'c.json') == calib


@pytest.mark.parametrize('kwargs, message', [(dict(scale=(0.0, 1.0, 1.0)), 'scales'),
                                             (dict(groups=((0, 1), (1, 2), (3,))), 'two groups'),
                                             (dict(scale=(1.0, 1.0)), 'one scale')])
def test_invalid_calibrations_are_refused(kwargs, message):
    fields = {**dict(scale=(1.0, 1.0, 1.0), shift=(0.0, 0.0, 0.0), groups=GROUPS), **kwargs}
    with pytest.raises(ValueError, match=message):
        TaskCalib(**fields)


SIZES = (2, 1, 1, 1, 1)


def toy_val_and_dump(calib_ids, target_ids):
    """One object per image at the same place; task 1 gets +9 on its logits, so objects of later tasks are counted
    as label 0 until the calibration lowers task 1."""
    images, annotations, logits = [], [], []
    labels = {i: int(i % 6) for i in calib_ids + target_ids}
    for n, image_id in enumerate(sorted(labels)):
        images.append({'id': image_id, 'width': 100, 'height': 100, 'file_name': f'{image_id}.jpg'})
        annotations.append({'id': n + 1, 'image_id': image_id, 'category_id': labels[image_id],
                            'bbox': [0, 0, 10, 10], 'area': 100, 'iscrowd': 0})
        row = np.full((2, 6), -6.0, dtype=np.float32)
        row[0, labels[image_id]] = 2.0
        row[0, :2] += 9.0
        logits.append(row)
    ids = np.array(sorted(labels))
    boxes = np.tile(np.array([[0, 0, 10, 10], [60, 60, 70, 70]], dtype=np.float32), (len(ids), 1, 1))
    dump = LogitDump(ids, np.tile(np.array([[0, 1]]), (len(ids), 1)), np.stack(logits), boxes)
    val = {'images': images, 'annotations': annotations,
           'categories': [{'id': c, 'name': str(c)} for c in range(6)]}
    return val, dump


CALIB_IDS, TARGET_IDS = tuple(range(1, 25)), tuple(range(101, 125))


def toy_search(calib_ids=CALIB_IDS, target_ids=TARGET_IDS):
    val, dump = toy_val_and_dump(list(calib_ids), list(target_ids))
    cfg = TaskConfig.from_dict(task_config_dict(sizes=SIZES, reserved=0))
    return Search(dump, val, cfg, list(calib_ids), list(target_ids))


def test_search_refuses_val_b_images_in_the_fit():
    with pytest.raises(ValueError, match='fit images are scored'):
        toy_search(calib_ids=range(1, 25), target_ids=range(20, 40))


def test_search_fixes_the_shifted_task_on_val_b():
    result = run(toy_search())
    base, bias = result['variants']['base'], result['variants']['bias_only']
    assert base['cAcc_m2'] < 0.5 and bias['cAcc_m2'] == 1.0
    assert bias['calibration']['shift'][0] < -1.0
    assert bias['fixed_vs_base'] > 0 and bias['broken_vs_base'] == 0
    assert base['valb_errors']['swap_cross_task'] > 0 and bias['valb_errors']['swap'] == 0
    assert bias['stable_halves'] and bias['decision'] == 'adopt'
    assert len(bias['halves']) == 2


def test_main_writes_the_result_and_one_calibration_file_per_variant(tmp_path):
    val, dump = toy_val_and_dump(list(range(1, 25)), list(range(101, 125)))
    (tmp_path / 'eval').mkdir()
    (tmp_path / 'eval/val_full.json').write_text(json.dumps(val))
    (tmp_path / 'eval/task_config.json').write_text(json.dumps(task_config_dict(sizes=SIZES, reserved=0)))
    (tmp_path / 'relearn/active/splits').mkdir(parents=True)
    (tmp_path / 'relearn/meta').mkdir(parents=True)
    (tmp_path / 'relearn/active/splits/val_a_split.json').write_text(json.dumps({'calib': list(range(1, 25))}))
    (tmp_path / 'relearn/meta/val_b.json').write_text(json.dumps({'images': [{'id': i} for i in range(101, 125)]}))
    save_logits(tmp_path / 'logits_val.npz', dump)
    out = tmp_path / 'res/toy.json'
    result = task_calib_search.main(['--name', 'toy', '--logits', str(tmp_path / 'logits_val.npz'),
                                     '--data', str(tmp_path), '--out', str(out)])
    assert json.loads(out.read_text())['variants'].keys() == {'base', 'bias_only', 'scale_bias'}
    calib = TaskCalib.load(result['variants']['bias_only']['file'])
    assert calib.groups == ((0, 1), (2,), (3,), (4,), (5,)) and calib.mode == 'bias_only'
