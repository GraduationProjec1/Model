"""tools/al_build_variant.py: AL training data keeps the fixed part and refuses Val-B / calib / test images (A4)."""

import json

import pytest

from autocheckout.io import load_json
from tools import al_build_variant as bv

CATS = [{'id': c, 'name': f'c{c}'} for c in range(4)]


def _coco(images, cats=CATS):
    return {'images': images, 'annotations': [{'id': i['id'], 'image_id': i['id'], 'category_id': i['id'] % 4,
                                               'bbox': [0, 0, 5, 5], 'area': 25, 'iscrowd': 0} for i in images],
            'categories': cats}


def _w(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


@pytest.fixture
def release(tmp_path):
    rel = tmp_path / 'rel'
    val_a = _coco([{'id': i, 'file_name': f'val2019/a{i}.jpg', 'width': 10, 'height': 10, 'level': 'easy'}
                   for i in range(1, 9)])
    _w(rel / 'meta/val_a.json', val_a)
    _w(rel / 'meta/val_b.json', _coco([{'id': 20, 'file_name': 'val2019/b20.jpg', 'width': 10, 'height': 10}]))
    _w(rel / 'eval/test_full.json', _coco([{'id': 30, 'file_name': 'test2019/a8.jpg', 'width': 10, 'height': 10}]))
    for i in range(1, 9):
        (rel / 'archive/val2019').mkdir(parents=True, exist_ok=True)
        (rel / f'archive/val2019/a{i}.jpg').write_bytes(b'x')
    base = _coco([{'id': 1, 'file_name': 'real/val2019/a1.jpg', 'width': 10, 'height': 10, 'orig_id': 1,
                   'source': 'real'},
                  {'id': 2, 'file_name': 'v2/task_1/9.jpg', 'width': 10, 'height': 10, 'orig_id': 9, 'source': 'v2'},
                  {'id': 3, 'file_name': 'relearn/task_6/7.jpg', 'width': 10, 'height': 10, 'orig_id': 7,
                   'source': 'relearn'}])
    _w(rel / 'variants/pilot/train_task_5_fix.json', base)
    _w(rel / 'variants/pilot/train_rows.json', [1, 2])
    _w(rel / 'variants/pilot/manifest.json', {'params': {'feedback_repeat': 3}})
    _w(rel / 'active/splits/val_a_split.json', {'pool': [1, 2, 3, 4, 5, 8], 'calib': [6, 7]})
    for name, ids in {'s1': [2, 3], 's2': [4, 5], 'bad_calib': [2, 6], 'bad_test': [8]}.items():
        _w(rel / f'active/selections/{name}.json', {'strategy': 'x', 'budget': len(ids), 'seed': 0, 'image_ids': ids})
    return rel


def test_fixed_part_is_kept_and_real_images_repeat(release):
    m1, m2 = bv.build(release, 's1'), bv.build(release, 's2')
    assert m1['fixed_part_md5'] == m2['fixed_part_md5']
    assert m1['images_per_source'] == {'real': 6, 'v2': 1, 'relearn': 1}
    train = load_json(release / 'variants/al_s1/train_task_5_fix.json')
    real = [i for i in train['images'] if i['source'] == 'real']
    assert sorted(i['orig_id'] for i in real) == [2, 2, 2, 3, 3, 3]
    assert len({i['id'] for i in train['images']}) == len(train['images'])
    assert {a['image_id'] for a in train['annotations']} == {i['id'] for i in train['images']}
    assert (release / 'images/real/val2019/a2.jpg').is_file() and m1['images_copied'] == 2
    assert load_json(release / 'variants/al_s1/train_rows.json') == [1, 2]


@pytest.mark.parametrize('name', ['bad_calib', 'bad_test'])
def test_leaks_are_refused(release, name):
    with pytest.raises(ValueError):
        bv.build(release, name)


def test_report_table_and_curve(tmp_path):
    from tools import al_report

    rel, campaign = tmp_path / 'rel', tmp_path / 'camp'
    _w(rel / 'active/scored/baseline_r1.json', {'name': 'baseline_r1', 'cAcc_m2': 0.40})
    for name, strategy, budget, seed, cacc in [('al_random_B10_s0', 'random', 10, 0, 0.41),
                                                ('al_random_B10_s1', 'random', 10, 1, 0.43),
                                                ('al_margin_B10', 'margin', 10, None, 0.45),
                                                ('al_margin_B20', 'margin', 20, None, 0.47)]:
        _w(rel / f'active/scored/{name}.json', {'name': name, 'cAcc_m2': cacc})
        _w(rel / f'variants/{name}/manifest.json', {'strategy': strategy, 'budget': budget, 'seed': seed,
                                                    'labelled_images': budget, 'labelled_boxes': budget * 12})
    _w(campaign / 'stages/al_margin_B10.json', {'cAcc_m2': 0.46, 'AP50_other': {'base': 0.9, 'new': 0.89},
                                               'comparison': {'pair_confusions_in_rows': {'base': 50, 'new': 20}},
                                               'gate': {'decision': 'go'}})
    rows = al_report.main(['--release', str(rel), '--campaign', str(campaign)])
    assert len(rows) == 5
    md = (rel / 'active/report/al_results.md').read_text()
    assert '| al_margin_B10 | margin | 1 | 10 |' in md and '+0.0500' in md and '50→20' in md and '| go |' in md
    assert (rel / 'active/report/al_curve.png').stat().st_size > 1000


def test_round2_adds_new_pool_images_to_the_labelled_set(release, tmp_path):
    import numpy as np

    from autocheckout.predictions import Predictions, save_predictions
    from tools import al_round2

    _w(release / 'active/selections/m.json', {'strategy': 'margin', 'budget': 2, 'seed': None, 'opts': {},
                                              'pool_md5': 'x', 'image_ids': [2, 3]})
    bv.build(release, 'm')
    campaign = tmp_path / 'camp'
    ids = [1, 2, 3, 4, 5, 6, 7, 8]  # predictions on all of val; calib images (6, 7) must never be picked
    margins = {1: 0.2, 2: 0.0, 3: 0.0, 4: 0.1, 5: 0.9, 6: 0.0, 7: 0.0, 8: 0.95}  # 8 is a test file
    rows = [(i, 0, 0, 0.9, [0, 0, 5, 5]) for i in ids] + [(i, 0, 1, 0.9 - margins[i], [0, 0, 5, 5]) for i in ids]
    image, query, label, score, boxes = zip(*rows, strict=True)
    save_predictions(campaign / 'runs/EXP-FB-FT-al_m/task_5/pred_val.npz',
                     Predictions(np.array(image), np.array(query), np.array(label), np.array(score), np.array(boxes)))
    manifest = al_round2.round2(release, campaign, 'al_m')
    chosen = load_json(release / 'active/selections/m_r2.json')
    assert chosen['image_ids'] == [2, 3, 4, 1] and chosen['new_image_ids'] == [4, 1]
    assert manifest['variant'] == 'al_m_r2' and manifest['round'] == 2 and manifest['labelled_images'] == 4
    with pytest.raises(ValueError):
        al_round2.round2(release, campaign, 'al_m_r2')


def test_update_model_al_recipes_and_start(tmp_path):
    from autocheckout import update_model as um

    assert um._kind('al_margin_B500') == 'pilot' and um._kind('al_margin_B500_r2') == 'round2'
    assert um._kind('e1') == 'e1'
    plan = {'seed_task5': 'seed.pth', 'root': str(tmp_path), 'runs': str(tmp_path / 'runs')}
    assert um._al_start(plan, 'al_x') == ('seed.pth', 'baseline')
    with pytest.raises(RuntimeError, match='train al_x first'):
        um._al_start(plan, 'al_x_r2')
    _w(tmp_path / 'stages/al_x.json', {'cAcc_m2': 0.5})
    prev, reference = um._al_start(plan, 'al_x_r2')
    assert reference == 'al_x' and prev.endswith('runs/EXP-FB-FT-al_x/task_5/task_final.pth')
