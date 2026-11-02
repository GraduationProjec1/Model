"""autocheckout/feedback_fix.py: data of the improve mode (update-model plan, I2-I7)."""

import json

import numpy as np

from autocheckout import feedback_fix as ff
from autocheckout.predictions import Predictions


def rpc_val(n=12):
    """Val images with RPC names: pairs of images share a suffix (same scene), three levels, one SKU per image
    (scene groups also join images with the same SKU multiset, so every image gets its own SKU)."""
    images, annotations = [], []
    for i in range(1, n + 1):
        images.append({'id': i, 'file_name': f'20180824-13-43-{i:02d}-{(i + 1) // 2}.jpg',
                       'level': ['easy', 'medium', 'hard'][(i - 1) // 4]})
        annotations.append({'id': i, 'image_id': i, 'category_id': i, 'bbox': [0, 0, 10, 10]})
    return {'images': images, 'annotations': annotations, 'categories': [{'id': c} for c in range(n + 1)]}


def test_split_keeps_scene_groups_whole_and_balances_levels():
    val = rpc_val()
    a, b = ff.split_val(val, seed=1)
    ids_a, ids_b = {i['id'] for i in a['images']}, {i['id'] for i in b['images']}
    assert ids_a | ids_b == set(range(1, 13)) and not ids_a & ids_b
    keys = ff.scene_groups(val)
    assert not {keys[i] for i in ids_a} & {keys[i] for i in ids_b}
    for level in ('easy', 'medium', 'hard'):
        counts = [sum(1 for i in half['images'] if i['level'] == level) for half in (a, b)]
        assert abs(counts[0] - counts[1]) <= 2  # the largest scene group holds two images
    assert {x['image_id'] for x in a['annotations']} <= ids_a
    assert ff.split_val(val, seed=1) == (a, b)


def test_fixture_names_fall_back_to_one_group_per_image():
    val = {'images': [{'id': 1, 'file_name': 'val_1.jpg'}, {'id': 2, 'file_name': 'val_2.jpg'}], 'annotations': []}
    a, b = ff.split_val(val, seed=0)
    assert len(a['images']) == len(b['images']) == 1


def counted(rows):
    return Predictions(image_id=np.array([r[0] for r in rows], dtype=np.int64),
                       query=np.arange(len(rows), dtype=np.int32),
                       label=np.array([r[1] for r in rows], dtype=np.int32),
                       score=np.ones(len(rows), dtype=np.float32),
                       boxes=np.array([[0, 0, 10, 10]] * len(rows), dtype=np.float32).reshape(-1, 4))


def test_wrong_images_pairs_and_rows():
    val = rpc_val(4)  # image i holds one object of label i
    preds = counted([(1, 2), (2, 1), (3, 4), (4, 4)])  # 1 -> 2, 2 -> 1, 3 -> 4
    assert ff.wrong_images(val, preds, range(5)) == [1, 2, 3]
    pairs = ff.confusion_pairs(val, preds, range(5))
    assert pairs == [(1, 2, 2), (3, 4, 1)]
    assert ff.pair_rows(pairs, top=1) == [1, 2]
    assert ff.pair_rows(pairs, min_count=2, extra=[0]) == [0, 1, 2]


def test_aliases_go_back_to_their_original_label_and_images_with_rows():
    coco = {'images': [{'id': 1}, {'id': 2}], 'annotations': [
        {'id': 1, 'image_id': 1, 'category_id': 200}, {'id': 2, 'image_id': 2, 'category_id': 7}]}
    resolved = ff.resolve_aliases(coco, {200: 19})
    assert [a['category_id'] for a in resolved['annotations']] == [19, 7]
    assert [i['id'] for i in ff.images_with(resolved, [19])['images']] == [1]


def test_replay_covers_every_class_first_then_fills():
    images = [{'id': i} for i in range(1, 21)]
    # label 0 everywhere but images 18-20, which hold the rare labels 3, 4, 5; 5 only in image 20 (excluded)
    anns = [{'id': i, 'image_id': i, 'category_id': 0 if i <= 17 else i - 15} for i in range(1, 21)]
    coco = {'images': images, 'annotations': anns}
    replay = ff.select_replay(coco, exclude=[20], n_images=6, min_per_class=1, labels=range(6), seed=0)
    labels = {a['category_id'] for a in replay['annotations']}
    assert len(replay['images']) == 6 and {0, 3, 4} <= labels and 5 not in labels
    assert 20 not in {i['id'] for i in replay['images']}


def test_release_prefixes_names_renumbers_and_repeats(tmp_path):
    (tmp_path / 'synth').mkdir()
    (tmp_path / 'raw').mkdir()
    synth = {'images': [{'id': 5000000, 'file_name': 'images/task_5/a.jpg'}],
             'annotations': [{'id': 9, 'image_id': 5000000, 'category_id': 1}]}
    real = {'images': [{'id': 1, 'file_name': 'val2019/b.jpg'}],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 2}, {'id': 2, 'image_id': 1, 'category_id': 0}]}
    train, manifest = ff.build_release(tmp_path / 'release', {'v2': tmp_path / 'synth', 'real': tmp_path / 'raw'},
                                       [('v2', synth, 1), ('real', real, 3)], [{'id': c} for c in range(3)])
    assert [i['file_name'] for i in train['images']] == ['v2/images/task_5/a.jpg'] + ['real/val2019/b.jpg'] * 3
    assert [i['id'] for i in train['images']] == [1, 2, 3, 4]
    assert len({a['id'] for a in train['annotations']}) == len(train['annotations']) == 7
    assert (tmp_path / 'release/real').resolve() == (tmp_path / 'raw').resolve()
    assert manifest['images_per_source'] == {'v2': 1, 'real': 3}
    target = ff.build_task_dir(tmp_path / 'tasks', train, real, real, tmp_path / 'raw')
    assert json.loads((target / 'train_task_5_fix.json').read_text())['images'][0]['source'] == 'v2'
    assert sorted(p.name for p in target.iterdir()) == ['test_full.json', 'train_task_5_fix.json', 'val_full.json',
                                                         'val_task_5.json']


def test_quota_caps_objects_and_takes_images_with_both_sides_of_a_pair_first():
    images, annotations = [], []
    layout = {1: [1, 2], 2: [1, 9], 3: [2, 9], 4: [1, 2], 5: [9, 9], 6: [1], 7: [3]}  # image -> labels
    for image_id, labels in layout.items():
        images.append({'id': image_id})
        for label in labels:
            annotations.append({'id': len(annotations) + 1, 'image_id': image_id, 'category_id': label})
    coco = {'images': images, 'annotations': annotations}
    chosen = ff.select_by_quota(coco, {1: 2, 2: 2}, seed=0)
    ids = {i['id'] for i in chosen['images']}
    assert ids == {1, 4}  # two images with both 1 and 2 reach both quotas; nothing else adds a needed object
    assert {a['image_id'] for a in chosen['annotations']} == ids
    assert ff.select_by_quota(coco, {1: 2, 2: 2}, seed=0) == chosen
    assert {i['id'] for i in ff.select_by_quota(coco, {1: 2, 2: 2}, seed=0, exclude=[1])['images']} >= {4}
    assert {i['id'] for i in ff.select_by_quota(coco, {3: 5}, seed=0)['images']} == {7}  # short of quota: all
