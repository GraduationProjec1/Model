"""autocheckout/new_skus.py: update_model --mode new (Task 9 of docs/project/feedback-finetune-plan-2026-11-01.md)."""

import json
from pathlib import Path

import numpy as np
import pytest

from autocheckout import new_skus
from autocheckout.predictions import Predictions, load_predictions
from autocheckout.taskcfg import TaskConfig
from tools.make_new_task_config import new_task_config

from pdp_helpers import task_config_dict

BASE = TaskConfig.from_dict(task_config_dict(sizes=(1, 1), reserved=0))  # old labels 0 | 1
CFG = new_task_config(BASE, [{'pdp_class_id': 2, 'name': 'new'}], reserve_after=1, name='toy+new')  # task 3: label 2
A, B, C, D = [10, 10, 30, 30], [50, 10, 70, 30], [10, 50, 30, 70], [50, 50, 70, 70]  # xyxy


def xywh(box):
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def val_full():
    """RPC images 1-4 (calib 1, 2; Val-B 3, 4) with labels 0 at A and 1 at B; photos 101, 102 of the new product
    (label 2 at C)."""
    images = [{'id': i, 'file_name': f'{i}.jpg', 'width': 100, 'height': 100} for i in (1, 2, 3, 4, 101, 102)]
    anns = [(i, 0, A) for i in (1, 2, 3, 4)] + [(i, 1, B) for i in (1, 2, 3, 4)] + [(i, 2, C) for i in (101, 102)]
    return {'images': images, 'categories': [{'id': c, 'name': str(c)} for c in range(3)],
            'annotations': [{'id': k, 'image_id': i, 'category_id': c, 'bbox': xywh(b), 'area': 400, 'iscrowd': 0}
                            for k, (i, c, b) in enumerate(anns, start=1)]}


def predictions(with_new: bool) -> Predictions:
    rows = [(i, 0, 0, .8, A) for i in (1, 2, 3, 4)] + [(i, 1, 1, .8, B) for i in (1, 2, 3, 4)]
    rows += [(101, 1, 0, .6, C), (102, 0, 0, .8, C)]  # the old label 0 on the new product, in both photos
    if with_new:
        rows += [(4, 2, 2, .7, D),     # the new label on an RPC image: a counting error of Val-B
                 (101, 0, 2, .9, C)]   # the new label wins the NMS against the old one in photo 101
    image_id, query, label, score, boxes = zip(*rows, strict=False)
    return Predictions(image_id=image_id, query=query, label=label, score=score, boxes=boxes)


def scored(with_new: bool) -> dict:
    return new_skus.score_stage(val_full(), CFG, predictions(with_new), stage=3 if with_new else 2, old_stage=2,
                                calib=[1, 2], val_b=[3, 4], new_val=[101, 102])


def test_baseline_scores_old_products_and_old_labels_on_the_new_photos():
    r = scored(False)
    assert r['thresholds'] == {'task_1': 0.1, 'task_2': 0.1}  # every value works on calib: ties to the smallest
    assert (r['old']['cAcc_m2'], r['old']['AP50'], r['old']['right_images']) == (1.0, 1.0, [3, 4])
    assert r['new']['claimed_by_old_label'] == 2 and r['new']['cross_rate'] == 1.0
    assert 'AP50' not in r['new']  # nothing of the new task is learned yet


def test_new_task_threshold_on_the_new_photos_and_every_label_counted():
    r = scored(True)
    assert r['thresholds'] == {'task_1': 0.1, 'task_2': 0.1, 'task_3': 0.1}
    # Val-B: the new label on RPC image 4 counts; old labels only, as before
    assert (r['old']['cAcc_m2'], r['old']['cAcc_m2_old_labels_only'], r['old']['right_images']) == (0.5, 1.0, [3])
    assert r['old']['AP50'] == 1.0
    # photo 101 right (new label kept by the NMS), 102 wrong (only the old label): cAcc 0.5, AP50 at recall 0.5
    assert r['new']['cAcc_m2'] == 0.5 and r['new']['cAcc_m2_new_labels_only'] == 0.5
    assert r['new']['AP50'] == pytest.approx(51 / 101, abs=1e-3) and r['new']['objects_per_sku'] == {'2': 2}
    assert (r['new']['claimed_by_old_label'], r['new']['cross_rate']) == (1, 0.5)


def test_verdict_go_stop_and_filter():
    base = {'old': {'AP50': 0.98, 'cAcc_m2': 0.68}, 'new': {'cross_rate': 0.72}}
    good = {'old': {'AP50': 0.975, 'cAcc_m2': 0.675}, 'new': {'AP50': 0.93, 'cross_rate': 0.02}}
    assert new_skus.verdict(good, base)[0::2] == ('go', False)
    decision, reason, must_filter = new_skus.verdict({**good, 'new': {'AP50': 0.93, 'cross_rate': 0.08}}, base)
    assert (decision, must_filter) == ('go', True) and 'filter' in reason
    for change, text in (({'new': {'AP50': 0.85, 'cross_rate': 0}}, 'new products'),
                         ({'new': {'AP50': float('nan'), 'cross_rate': 0}}, 'new products'),
                         ({'old': {'AP50': 0.96, 'cAcc_m2': 0.68}}, 'AP50 of the old'),
                         ({'old': {'AP50': 0.98, 'cAcc_m2': 0.66}}, 'cAcc-M2 of Val-B')):
        decision, reason, _ = new_skus.verdict({**good, **change}, base)
        assert decision == 'stop' and text in reason


def test_sign_test_counts_fixed_and_broken_images():
    r = new_skus.sign_test([1, 2, 3], [2, 3, 4, 5])
    assert (r['fixed'], r['broken']) == (2, 1) and 0 < r['sign_test_p'] <= 1
    assert new_skus.sign_test([1], [1])['sign_test_p'] == 1.0


def test_merge_and_links_refuse_collisions(tmp_path):
    base = val_full()
    with pytest.raises(ValueError, match='collide'):
        new_skus._merge(base, {'images': base['images'][:1], 'annotations': [], 'categories': []})
    (tmp_path / 'a/x').mkdir(parents=True)
    (tmp_path / 'b/x').mkdir(parents=True)
    coco = {'images': [{'file_name': 'x/1.jpg'}]}
    with pytest.raises(ValueError, match='both'):
        new_skus._link_images(tmp_path / 'images', [(coco, tmp_path / 'a'), (coco, tmp_path / 'b')])
    new_skus._link_images(tmp_path / 'images', [(coco, tmp_path / 'a')])
    assert (tmp_path / 'images/x').resolve() == (tmp_path / 'a/x').resolve()


def test_update_model_routes_mode_new_stages():
    from autocheckout import update_model

    with pytest.raises(SystemExit):
        update_model.main(['--mode', 'new', '--settings', 'x.json', '--stage', 'e1'])
    with pytest.raises(SystemExit):
        update_model.main(['--mode', 'improve', '--settings', 'x.json', '--stage', 'expand'])


@pytest.mark.slow
def test_mode_new_dry_run_expands_trains_gates_and_tests_once(tmp_path):
    import torch

    from autocheckout import full_syn, update_model
    base = dict(round=1, artifact_root=str(tmp_path), dry_run=True, batch_size=1, effective_batch=2,
                resolution=640, fsa_epochs=8, epochs=16, audit_images=4)
    seed = full_syn.prepare({**base, 'campaign_id': 'seed'})
    for stage in (full_syn.stage_fsa, full_syn.stage_task1, full_syn.stage_tasks):
        stage(seed)
    settings = dict(campaign_id='new', round=1, artifact_root=str(tmp_path), dry_run=True,
                    seed_campaign=str(tmp_path / 'dry_run/seed'))
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(settings))
    plan = update_model.main(['--mode', 'new', '--settings', str(path), '--stage', 'data'])
    assert (plan['old_stage'], plan['new_task'], plan['new_labels'], plan['n_classes']) == (5, 6, [6], 9)
    task_dir = Path(plan['task_dir'])
    assert sorted(p.name for p in task_dir.iterdir()) == [
        'test_full.json', 'train_task_6.json', 'train_task_6_pilot.json', 'val_full.json', 'val_task_5.json',
        'val_task_6.json']
    splits = json.loads((Path(plan['root']) / 'inputs/splits.json').read_text())
    assert splits['val_b'] and not set(splits['val_b']) & set(splits['calib'])
    run = Path(plan['runs']) / new_skus.EXP

    expand = new_skus.stage_expand(plan)
    assert expand['old_rows_identical'] and torch.load(run / 'task_5/task_final.pth', map_location='cpu')[
        'model']['class_embed.0.weight'].shape[0] == 9
    baseline = new_skus.stage_baseline(plan)
    assert load_predictions(run / 'task_5/pred_val.npz').label.max() < 6  # still a task-5 model
    assert baseline['seed_val_b_cAcc_m2'] == baseline['old']['cAcc_m2']  # same weights for the learned classes
    pilot = update_model.main(['--mode', 'new', '--settings', str(path), '--stage', 'pilot'])
    assert pilot['gate'] is None and pilot['train_images'] == plan['pilot_images']
    task = update_model.main(['--mode', 'new', '--settings', str(path), '--stage', 'task'])
    assert task['gate']['decision'] in {'go', 'stop'} and task['gate']['enforced'] is False
    assert set(task['thresholds']) == {f'task_{t}' for t in range(1, 7)} and 'AP50' in task['new']
    assert set(task['comparison']) == {'fixed', 'broken', 'sign_test_p'}
    assert load_predictions(run / 'task_6/pred_val.npz').label.max() < 7
    assert new_skus.stage_train(plan, 'task')['finished_at'] == task['finished_at']  # never trained again

    with pytest.raises(RuntimeError, match='confirm'):
        new_skus.stage_final_test(plan)
    final = update_model.main(['--mode', 'new', '--settings', str(path), '--stage', 'final_test',
                               '--confirm-test'])
    assert final['images'] == len(splits['new_test']) and final['thresholds'] == task['thresholds']
    assert new_skus.stage_final_test(plan, confirm=True)['finished_at'] == final['finished_at']
    assert new_skus.prepare(settings) == plan
    assert np.isfinite(final['cAcc_m2'])


# --- option A of AN/task6/08 (replay RPC images in the new task) --------------------------------------------------
def replay_source():
    """r4's training file in small: image 1 twice (oversampled), images under real/ and v2/, labels 0 and 1."""
    images = [{'id': 1, 'file_name': 'real/val2019/a.jpg'}, {'id': 2, 'file_name': 'real/val2019/a.jpg'},
              {'id': 3, 'file_name': 'real/val2019/b.jpg'}, {'id': 4, 'file_name': 'v2/task_1/c.jpg'},
              {'id': 5, 'file_name': 'v2/task_1/d.jpg'}]
    anns = [(1, 0), (2, 0), (3, 1), (4, 0), (4, 1), (5, 1)]
    return {'images': images, 'categories': [{'id': 0, 'name': '0'}, {'id': 1, 'name': '1'}],
            'annotations': [{'id': k, 'image_id': i, 'category_id': c, 'bbox': xywh(A), 'area': 400, 'iscrowd': 0}
                            for k, (i, c) in enumerate(anns, start=1)]}


def test_replay_images_dedups_takes_each_part_and_refuses_leaks():
    chosen = new_skus.replay_images(replay_source(), {'real': 2, 'v2': 1}, min_per_class=1, learned=2,
                                    exclude=set(), seed=0)
    names = sorted(i['file_name'] for i in chosen['images'])
    assert len(names) == 3 and len(set(names)) == 3  # the oversampled image once
    assert sum(n.startswith('real/') for n in names) == 2 and sum(n.startswith('v2/') for n in names) == 1
    with pytest.raises(ValueError, match='Val-B'):
        new_skus.replay_images(replay_source(), {'real': 2}, min_per_class=1, learned=2, exclude={'b.jpg'}, seed=0)
    with pytest.raises(ValueError, match='learned'):
        new_skus.replay_images(replay_source(), {'real': 2}, min_per_class=1, learned=1, exclude=set(), seed=0)
    with pytest.raises(ValueError, match='only 2'):
        new_skus.replay_images(replay_source(), {'v2': 3}, min_per_class=1, learned=2, exclude=set(), seed=0)


def test_with_replay_renumbers_after_the_new_products():
    train = {'images': [{'id': 7, 'file_name': 'synth_new/x.jpg'}], 'categories': [{'id': 2, 'name': 'new'}],
             'annotations': [{'id': 9, 'image_id': 7, 'category_id': 2, 'bbox': xywh(C), 'area': 400, 'iscrowd': 0}]}
    merged = new_skus._with_replay(train, replay_source())
    assert [i['id'] for i in merged['images']] == [7, 8, 9, 10, 11, 12]
    assert len({a['id'] for a in merged['annotations']}) == 7 and min(a['id'] for a in merged['annotations'][1:]) > 9
    replayed = [i for i in merged['images'] if i.get('replay')]
    assert len(replayed) == 5 and replayed[0]['orig_id'] == 1
    assert {c['id'] for c in merged['categories']} == {0, 1, 2}


def test_new_label_taking_old_objects():
    # Val-B image 4: the new label D does not cover an old object; add one on B above the old label's score
    preds = predictions(True)
    assert new_skus.taken_by_new_label(val_full(), preds, [3, 4], old_seen=2)['taken'] == 0
    more = Predictions(image_id=[*preds.image_id, 3], query=[*preds.query, 5], label=[*preds.label, 2],
                       score=[*preds.score, .9], boxes=[*preds.boxes, B])
    r = new_skus.taken_by_new_label(val_full(), more, [3, 4], old_seen=2)
    assert (r['objects'], r['taken'], r['rate']) == (4, 1, 0.25)


@pytest.mark.slow
def test_mode_new_dry_run_with_replay_trains_on_old_and_new_labels(tmp_path):
    from autocheckout import full_syn, update_model
    base = dict(round=1, artifact_root=str(tmp_path), dry_run=True, batch_size=1, effective_batch=2,
                resolution=640, fsa_epochs=8, epochs=16, audit_images=4)
    seed = full_syn.prepare({**base, 'campaign_id': 'seed'})
    for stage in (full_syn.stage_fsa, full_syn.stage_task1, full_syn.stage_tasks):
        stage(seed)
    settings = dict(campaign_id='new-replay', round=1, artifact_root=str(tmp_path), dry_run=True,
                    seed_campaign=str(tmp_path / 'dry_run/seed'),
                    replay={'parts': {'': 2}, 'min_per_class': 1, 'seed': 0}, hyperparameters={'lr_old': 0})
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(settings))
    plan = update_model.main(['--mode', 'new', '--settings', str(path), '--stage', 'data'])
    train = json.loads((Path(plan['task_dir']) / 'train_task_6.json').read_text())
    labels = {a['category_id'] for a in train['annotations']}
    assert labels & set(range(6)) and 6 in labels  # old labels from the replay, the new label from the package
    assert plan['replay']['images'] == 2 and plan['train_images'] == len(train['images'])
    assert plan['synth_root'].endswith('inputs/train_images')
    assert all((Path(plan['synth_root']) / i['file_name']).is_file() for i in train['images'])
    pilot = update_model.main(['--mode', 'new', '--settings', str(path), '--stage', 'pilot'])
    assert pilot['train_images'] == plan['pilot_images'] and 'taken_by_new_label' in pilot['old']
    assert plan['hyperparameters']['task']['lr_old'] == 0 and pilot['hyperparameters']['lr_old'] == 0
    cli = json.loads((Path(plan['runs']) / 'EXP-NEW-pilot/resolved_cli.json').read_text())['arguments']
    assert cli[len(cli) - 1 - cli[::-1].index('--lr_old') + 1] == '0'  # the last --lr_old wins
    with pytest.raises(ValueError, match='unknown'):
        new_skus.prepare({**settings, 'campaign_id': 'bad', 'hyperparameters': {'lr_olds': 0}})
    assert new_skus.prepare(settings) == plan


def test_link_images_named_folders(tmp_path):
    (tmp_path / 'rpc/val2019').mkdir(parents=True)
    (tmp_path / 'pkg/synth_new').mkdir(parents=True)
    coco = {'images': [{'file_name': 'synth_new/1.jpg'}]}
    new_skus._link_images(tmp_path / 'train', [(coco, tmp_path / 'pkg')], named={'real': tmp_path / 'rpc'})
    assert (tmp_path / 'train/real/val2019').resolve() == (tmp_path / 'rpc/val2019').resolve()
    assert (tmp_path / 'train/synth_new').resolve() == (tmp_path / 'pkg/synth_new').resolve()
    with pytest.raises(ValueError, match='both'):
        new_skus._link_images(tmp_path / 'x', [(coco, tmp_path / 'pkg')], named={'synth_new': tmp_path / 'rpc'})
