"""N0 (plan N1-N5 §2.1): tools/swap_rank.py finds the rank, task and supercategory of the true class of each swap."""

import json

import numpy as np
import pytest

from autocheckout.logit_dump import LogitDump, save_logits
from autocheckout.taskcfg import TaskConfig
from tools import swap_rank
from tools.swap_rank import gate, label_maps, match_queries, swap_table

from pdp_helpers import task_config_dict

SIZES = (2, 1, 1, 1, 1)


def box(x):
    return [x, 0.0, x + 10.0, 10.0]


def config(supercategories=('milk', 'tea', 'milk', 'gum', 'gum', 'gum')):
    raw = task_config_dict(sizes=SIZES, reserved=0)
    for task in raw['tasks']:
        for c in task['classes']:
            c['supercategory'] = supercategories[c['label']]
    return raw


def fake_dump(image_id=1):
    """Object 0 (label 0, task 1) is counted right; object 1 (label 2, task 2) is counted as label 1 (task 1), with
    label 2 second; query 2 is background."""
    logits = np.full((1, 3, 6), -6.0, dtype=np.float32)
    logits[0, 0, 0] = 4.0
    logits[0, 1, 1], logits[0, 1, 2] = 4.0, 3.0
    boxes = np.array([[box(0), box(50), box(200)]], dtype=np.float32)
    return LogitDump(np.array([image_id]), np.array([[5, 9, 17]]), logits, boxes)


def val(image_id=1):
    return {'images': [{'id': image_id}], 'annotations': [
        {'image_id': image_id, 'category_id': 0, 'bbox': [0, 0, 10, 10]},
        {'image_id': image_id, 'category_id': 2, 'bbox': [50, 0, 10, 10]}]}


def test_counted_queries_are_matched_to_their_objects():
    matched = match_queries(fake_dump(), val(), [1], 0.45, np.full(6, 0.5))
    assert sorted(zip(matched.true_label.tolist(), matched.pred_label.tolist(), strict=True)) == [(0, 0), (2, 1)]


def test_a_swap_with_the_true_class_second_in_another_task():
    cfg = TaskConfig.from_dict(config())
    table = swap_table(match_queries(fake_dump(), val(), [1], 0.45, np.full(6, 0.5)), *label_maps(cfg, 6))
    assert table['swaps'] == 1 and table['true_rank'] == {'1': 0, '2': 1, '3+': 0}
    assert table['cross_task'] == table['cross_task_rank2'] == 1
    assert table['cross_task_newer_wins'] == 0
    assert table['same_supercategory'] == 0
    assert table['gap_mean'][0][1] == pytest.approx(1.0) and table['gap_count'][0][1] == 1


def test_same_supercategory_is_counted():
    cfg = TaskConfig.from_dict(config(('milk', 'tea', 'tea', 'gum', 'gum', 'gum')))
    table = swap_table(match_queries(fake_dump(), val(), [1], 0.45, np.full(6, 0.5)), *label_maps(cfg, 6))
    assert table['same_supercategory'] == 1


def test_images_outside_the_requested_ones_are_not_read():
    matched = match_queries(fake_dump(), val(), [2], 0.45, np.full(6, 0.5))
    assert len(matched) == 0


def test_gate_needs_rank2_and_one_direction():
    assert gate({'share_cross_task_rank2': 0.5, 'share_newer_wins': 0.7})['do_n1']
    assert not gate({'share_cross_task_rank2': 0.3, 'share_newer_wins': 0.9})['do_n1']
    assert not gate({'share_cross_task_rank2': 0.6, 'share_newer_wins': 0.5})['do_n1']


def test_main_reads_the_calib_images_and_writes_the_result(tmp_path):
    (tmp_path / 'eval').mkdir()
    (tmp_path / 'eval/val_full.json').write_text(json.dumps(val()))
    (tmp_path / 'eval/task_config.json').write_text(json.dumps(config()))
    (tmp_path / 'split.json').write_text(json.dumps({'calib': [1], 'pool': []}))
    (tmp_path / 'policy.json').write_text(json.dumps(
        {'thresholds': {f'task_{t}': 0.5 for t in range(1, 6)}, 'nms_iou': 0.45}))
    save_logits(tmp_path / 'logits_val.npz', fake_dump())
    out = tmp_path / 'r.json'
    result = swap_rank.main(['--name', 'toy', '--logits', str(tmp_path / 'logits_val.npz'), '--data', str(tmp_path),
                             '--policy', str(tmp_path / 'policy.json'), '--split', str(tmp_path / 'split.json'),
                             '--out', str(out)])
    assert result['swaps'] == 1 and json.loads(out.read_text())['images'] == 'calib'
