"""autocheckout/joint_vn.py: scores of a model that learns RPC and VN products together (A6, AN/cyclegan/03)."""

import pytest

from autocheckout import joint_vn
from autocheckout.predictions import Predictions
from autocheckout.taskcfg import TaskConfig

# task 1: RPC 3 (label 0) + VN 201 (label 1); task 2: RPC 4 (label 2) + VN 202 (label 3)
CFG = TaskConfig.from_dict({'name': 'toy225', 'seed': 0, 'tasks': [
    {'task_id': 1, 'offset': 0, 'classes': [{'label': 0, 'name': 'r3', 'rpc_category_id': 3},
                                            {'label': 1, 'name': 'VN01', 'rpc_category_id': 201}]},
    {'task_id': 2, 'offset': 2, 'classes': [{'label': 2, 'name': 'r4', 'rpc_category_id': 4},
                                            {'label': 3, 'name': 'VN02', 'rpc_category_id': 202}]}]})
A, B, C, D = [10, 10, 30, 30], [50, 10, 70, 30], [10, 50, 30, 70], [50, 50, 70, 70]  # xyxy


def xywh(box):
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def val_full():
    """RPC images 1, 2 (calib) and 3, 4 (Val-B): label 0 at A, label 2 at B. VN photos 101, 102: label 1 at C,
    label 3 at D."""
    images = [{'id': i, 'file_name': f'{i}.jpg', 'width': 100, 'height': 100} for i in (1, 2, 3, 4, 101, 102)]
    anns = [(i, 0, A) for i in (1, 2, 3, 4)] + [(i, 2, B) for i in (1, 2, 3, 4)]
    anns += [(i, 1, C) for i in (101, 102)] + [(i, 3, D) for i in (101, 102)]
    return {'images': images, 'categories': [{'id': c, 'name': str(c)} for c in range(4)],
            'annotations': [{'id': k, 'image_id': i, 'category_id': c, 'bbox': xywh(b), 'area': 400, 'iscrowd': 0}
                            for k, (i, c, b) in enumerate(anns, start=1)]}


def predictions(rows):
    image_id, query, label, score, boxes = zip(*rows, strict=False)
    return Predictions(image_id=image_id, query=query, label=label, score=score, boxes=boxes)


def clean_rows():
    rows = [(i, 0, 0, .8, A) for i in (1, 2, 3, 4)] + [(i, 1, 2, .8, B) for i in (1, 2, 3, 4)]
    rows += [(i, 0, 1, .5, C) for i in (101, 102)] + [(i, 1, 3, .5, D) for i in (101, 102)]
    return rows


def test_groups_split_every_task_by_origin():
    assert joint_vn.split_groups(CFG, 2) == [('task_1_rpc', [0], 'rpc'), ('task_1_vn', [1], 'vn'),
                                             ('task_2_rpc', [2], 'rpc'), ('task_2_vn', [3], 'vn')]
    assert joint_vn.split_groups(CFG, 1) == [('task_1_rpc', [0], 'rpc'), ('task_1_vn', [1], 'vn')]


def test_rpc_thresholds_on_calib_vn_thresholds_on_vn_val():
    # VN scores 0.5 on the VN photos; a VN false positive at 0.4 on calib image 1 must not move the VN thresholds
    rows = clean_rows() + [(1, 2, 1, .4, D)]
    th, info = joint_vn.choose_thresholds(val_full(), CFG, predictions(rows), 2, calib=[1, 2], vn_val=[101, 102])
    assert th == {'task_1_rpc': 0.1, 'task_2_rpc': 0.1, 'vn': 0.1}
    assert info['calib_cAcc'] == 1.0 and info['vn_val_cAcc'] == 1.0
    assert (info['calib_images'], info['vn_val_images']) == (2, 2)


def test_one_vn_threshold_rises_above_a_double_count_and_rpc_ones_stay():
    # photo 101: a second VN-02 box at 0.3 (a double count): the one VN threshold (02 section 5.3) is the smallest
    # value above it; the RPC thresholds come from calib alone
    rows = clean_rows() + [(101, 2, 3, .3, A)]
    th, info = joint_vn.choose_thresholds(val_full(), CFG, predictions(rows), 2, calib=[1, 2], vn_val=[101, 102])
    assert th['vn'] == pytest.approx(0.31) and th['task_1_rpc'] == th['task_2_rpc'] == 0.1
    assert info['vn_val_cAcc'] == 1.0


def test_report_counts_every_label_and_both_confusions():
    rows = clean_rows()
    rows[rows.index((101, 0, 1, .5, C))] = (101, 0, 0, .9, C)   # VN-01 of photo 101 read as RPC 3: VN -> RPC
    rows += [(3, 2, 3, .9, B)]                                   # VN-02 wins the NMS on RPC 4 of image 3: RPC -> VN
    th = {'task_1_rpc': .5, 'task_2_rpc': .5, 'vn': .5}
    r = joint_vn.report(val_full(), CFG, predictions(rows), 2, th, rpc_ids=[3, 4], vn_ids=[101, 102])
    assert r['rpc']['cAcc_m2'] == 0.5 and r['rpc']['cAcc_m2_rpc_labels_only'] == 0.5
    assert r['vn']['cAcc_m2'] == 0.5 and r['vn']['cAcc_m2_vn_labels_only'] == 0.5
    assert r['vn']['groups']['vn_task_1']['objects'] == 2 and r['vn']['groups']['vn_task_1']['cAcc'] == 0.5
    assert r['vn']['groups']['vn_task_2_6']['cAcc'] == 1.0
    assert r['vn']['groups']['vn_task_2_6']['AP50'] == pytest.approx(1.0)
    assert r['cross']['vn_to_rpc'] == {'objects': 4, 'claimed': 1, 'rate': 0.25}
    assert r['cross']['rpc_to_vn'] == {'objects': 4, 'claimed': 1, 'rate': 0.25}
    assert r['cross']['vn_label_detections_per_rpc_image'] == 0.5


def test_report_without_vn_images_scores_rpc_only():
    th = {'task_1_rpc': .5, 'vn': .5}
    r = joint_vn.report(val_full(), CFG, predictions(clean_rows()), 1, th, rpc_ids=[3, 4], vn_ids=[])
    assert r['rpc']['cAcc_m2'] == 1.0 and r['vn'] is None and 'vn_to_rpc' not in r['cross']
