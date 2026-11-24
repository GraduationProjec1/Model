"""tools/merge_225.py: train file of one task from several synthetic parts (A7, AN/cyclegan/03 section 1.6)."""

import pytest

from tools.merge_225 import merge_parts

CATS = [{'id': 0, 'name': 'r3'}, {'id': 1, 'name': 'VN01'}]


def part(image_ids, size=1840, label=0, bbox=(184.0, 368.0, 92.0, 46.0)):
    images = [{'id': i, 'file_name': f'images/task_1/{i}.jpg', 'width': size, 'height': size,
               'generation_task': 1} for i in image_ids]
    anns = [{'id': 10 * i, 'image_id': i, 'category_id': label, 'bbox': list(bbox), 'area': bbox[2] * bbox[3],
             'iscrowd': 0, 'segmentation': {'size': [size, size], 'counts': 'x'}} for i in image_ids]
    return {'images': images, 'annotations': anns, 'categories': [CATS[label]]}


def test_render_part_is_rescaled_and_moved_to_its_own_folder_and_ids():
    syn = part([1000000, 1000001])
    out = merge_parts([('rpc_syn', syn, 1.0, 0), ('rpc_render', syn, 800 / 1840, 100_000_000)])
    images = {i['id']: i for i in out['images']}
    assert sorted(images) == [1000000, 1000001, 101000000, 101000001]
    render = images[101000000]
    assert render['file_name'] == 'rpc_render/images/task_1/1000000.jpg' and render['width'] == render['height'] == 800
    assert images[1000000]['file_name'] == 'rpc_syn/images/task_1/1000000.jpg' and images[1000000]['width'] == 1840
    ann = next(a for a in out['annotations'] if a['image_id'] == 101000000)
    assert ann['bbox'] == pytest.approx([80.0, 160.0, 40.0, 20.0]) and ann['area'] == pytest.approx(800.0)
    assert ann['id'] == 100_000_000 + 10_000_000 and 'segmentation' not in ann
    assert 'segmentation' in next(a for a in out['annotations'] if a['image_id'] == 1000000)
    assert {i['part'] for i in out['images']} == {'rpc_syn', 'rpc_render'}


def test_categories_are_the_union_of_the_parts():
    out = merge_parts([('rpc_syn', part([1]), 1.0, 0), ('vn_syn', part([1], label=1), 1.0, 200_000_000)])
    assert out['categories'] == CATS


@pytest.mark.parametrize('bad, message', [
    (lambda: [('a', part([1]), 1.0, 0), ('b', part([1]), 1.0, 0)], 'collide'),
    (lambda: [('a', part([100_000_001]), 1.0, 0), ('b', part([1]), 1.0, 100_000_000)], 'offset'),
    (lambda: [('a', part([1], bbox=(1800.0, 0.0, 60.0, 10.0)), 1.0, 0)], 'outside'),
    (lambda: [('a', part([1]), 1.0, 0), ('b', {**part([2], label=1), 'categories': [{'id': 0, 'name': 'VN01'}]},
                                          1.0, 200_000_000)], 'category'),
])
def test_merge_refuses_broken_inputs(bad, message):
    with pytest.raises(ValueError, match=message):
        merge_parts(bad())
