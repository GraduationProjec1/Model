"""tools/eval_set_225.py: evaluation set of the joint RPC + VN campaign (A7, AN/cyclegan/03)."""

import pytest

from autocheckout.taskcfg import TaskConfig
from tools.eval_set_225 import build

# task 1: RPC 3 (label 0) + VN01 (label 1); task 2: RPC 4 (label 2) + VN02 (label 3)
CFG = TaskConfig.from_dict({'name': 'toy225', 'seed': 0, 'tasks': [
    {'task_id': 1, 'offset': 0, 'classes': [{'label': 0, 'name': 'r3', 'rpc_category_id': 3},
                                            {'label': 1, 'name': 'VN01 Sua', 'rpc_category_id': 201}]},
    {'task_id': 2, 'offset': 2, 'classes': [{'label': 2, 'name': 'r4', 'rpc_category_id': 4},
                                            {'label': 3, 'name': 'VN02 Mi', 'rpc_category_id': 202}]}]})


def coco(ids, cats, prefix):
    """images ``ids``; image i holds one object of every category in ``cats`` (old label ids)."""
    return {'images': [{'id': i, 'file_name': f'{prefix}/{i}.jpg', 'width': 100, 'height': 100} for i in ids],
            'annotations': [{'id': 100 * i + k, 'image_id': i, 'category_id': c, 'bbox': [k, k, 5, 5], 'area': 25,
                             'iscrowd': 0} for i in ids for k, c in enumerate(cats)],
            'categories': []}


def inputs():
    rpc_cats = [{'id': 7, 'name': '3_x', 'rpc_category_id': 3}, {'id': 8, 'name': '4_x', 'rpc_category_id': 4}]
    val = {**coco([1, 2, 3, 4], [7, 8], 'val2019'), 'categories': rpc_cats}
    test = {**coco([11, 12], [7, 8], 'test2019'), 'categories': rpc_cats}
    vn_cats = [{'id': 200, 'name': 'VN01 Sua'}, {'id': 201, 'name': 'VN02 Mi'}]
    vn_a = {**coco([2000000, 2000001], [200], 'checkout'), 'categories': vn_cats}
    vn_b = {**coco([2500000, 2500001], [201], 'checkout'), 'categories': vn_cats}
    split = {'val': ['checkout/2000000.jpg', 'checkout/2500000.jpg'], 'test': ['checkout/2000001.jpg',
                                                                             'checkout/2500001.jpg']}
    return dict(rpc_val=val, rpc_test=test, vn=[vn_a, vn_b], vn_split=split, calib=[1, 2], pool=[3], val_b=[4],
                cfg=CFG)


def test_files_hold_every_role_in_the_225_labels():
    files, roles = build(**inputs())
    assert roles == {'calib': [1, 2], 'pool': [3], 'val_b': [4], 'vn_val': [2000000, 2500000], 'test_rpc': [11, 12],
                     'test_vn': [2000001, 2500001]}
    val = files['val_full.json']
    assert sorted(i['id'] for i in val['images']) == [1, 2, 3, 4, 2000000, 2500000]
    labels = {(a['image_id'], a['category_id']) for a in val['annotations']}
    assert {(1, 0), (1, 2), (2000000, 1), (2500000, 3)} <= labels
    assert sorted(i['id'] for i in files['test_full.json']['images']) == [11, 12, 2000001, 2500001]


def test_val_task_files_never_hold_val_b():
    files, _ = build(**inputs())
    t1 = files['val_task_1.json']
    assert sorted(i['id'] for i in t1['images']) == [1, 2, 3, 2000000]
    assert {a['category_id'] for a in t1['annotations']} == {0, 1}
    assert sorted(i['id'] for i in files['val_task_2.json']['images']) == [1, 2, 3, 2500000]


@pytest.mark.parametrize('change, message', [
    (lambda d: d['vn_split']['test'].append('checkout/2000000.jpg'), 'both'),
    (lambda d: d['vn_split']['val'].pop(), 'not in the split'),
    (lambda d: d.update(val_b=[2]), 'overlap'),
    (lambda d: d['rpc_test']['images'][0].update(id=2000001), 'collide'),  # an RPC test id = a VN test id
])
def test_build_refuses_broken_inputs(change, message):
    data = inputs()
    change(data)
    with pytest.raises(ValueError, match=message):
        build(**data)
