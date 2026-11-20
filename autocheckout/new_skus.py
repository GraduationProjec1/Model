"""Update model, mode new (docs/project/feedback-finetune-plan-2026-11-01.md, Phần 2;
docs/project/mode-new-2026-11-20.md): learn products the model does not know in one new task with new class
slots, the PDP way.

The seed is a task-5 model (the best improve run, e.g. AL round 4) and the products come as a package of the
Checkout-system data pipeline (synthetic training images, real counter photos for val and test). Stages, each written
once to ``stages/<stage>.json`` and skipped when done:

``expand`` (pdp/expand.py to the slots of the new config) -> ``baseline`` (the expanded model, still a task-5 model,
predicts val: the mark of the gate) -> ``pilot`` (1 epoch on a subset: checks the chain on the GPU) -> ``task`` (the
new task, gate N7) -> ``final_test`` (the test photos of the new products, once, with confirm, after a go).

Validation (``val_full.json`` of the campaign) = RPC val (old products) + the val photos of the new products, one
prediction pass for both. Old products are judged on Val-B (val minus the Val-A pool and calib of the active-learning
split; never trained on), the thresholds of their task groups chosen on Val-A-calib, as tools/al_score.py does. The
threshold of the new task is chosen on the val photos of the new products (decision DL-07 of Checkout-system), the
old ones fixed. Every label counts: a new product reported on an RPC image, or an old product reported on a new
product, is a counting error.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from autocheckout.full_syn import (
    DRY_MODEL_ARGS,
    FSA,
    PDP,
    _attempt,
    _environment,
    _require,
    _run,
    _toy_releases,
    sha256_file,
)
from autocheckout.io import load_json, md5_file, save_json
from autocheckout.predictions import Predictions, load_predictions
from autocheckout.relearn import _finish
from autocheckout.taskcfg import TaskConfig

EXP = 'EXP-NEW'
RECIPE = ('resolution', 'batch_size', 'effective_batch', 'tf32', 'num_workers', 'checkpoint_every_minutes')
# N5 (lr 1e-4 / 1e-5, step down at 75 % of the epochs, 12 epochs) with the I2 augmentation of rounds 3-4. The pilot is
# one epoch on PILOT_IMAGES images: it checks data, training, predictions and scores on the GPU before the long run.
HYPERPARAMETERS = {'pilot': dict(epochs=1, lr=1e-4, lr_old=1e-5, lr_drop=100, augment=1),
                   'task': dict(epochs=12, lr=1e-4, lr_old=1e-5, lr_drop=9, augment=1)}
PILOT_IMAGES = 300
# Gate of the new task (N7, docs/project/mode-new-2026-11-20.md §4), written before any run:
# new_ap50: least AP50 of the new products on their val photos; old_ap50_drop / old_cacc_drop: largest fall of the
# AP50 / cAcc-M2 of the old products on Val-B against the baseline; cross_rate: largest share of new-product objects
# that an old label also counts. A cross_rate above it does not stop the run: the shop then filters the labels it
# does not sell (Checkout-system, task_skus), which the report says.
GATE = dict(new_ap50=0.90, old_ap50_drop=0.01, old_cacc_drop=0.01, cross_rate=0.05)
NMS, MATCH_IOU = 0.45, 0.5
# settings 'data_root' (or env AC_DATA): the data folder of autocheckout/datapaths.py. Keys set in the settings win.
DATA_ROOT_LAYOUT = {'eval_root': 'eval', 'raw_root': 'rpc', 'seed_campaign': 'models/full-syn-r1',
                    'pretrained_root': 'pretrained/hf-cache',
                    'calib_split': 'relearn/active/splits/val_a_split.json'}
DRY_NEW_LABELS = 1  # dry run: one new product (objects of fixture label 0 under the new label)


# --- scores (pure: unit-tested in tests/test_new_skus.py) ---------------------------------------------------------
def _groups(cfg: TaskConfig, stage: int) -> tuple[list[list[int]], list[str]]:
    return [list(cfg.task(t).labels) for t in range(1, stage + 1)], [f'task_{t}' for t in range(1, stage + 1)]


def _vector(groups: list[list[int]], thresholds: list[float], size: int) -> np.ndarray:
    """Per-label thresholds; labels outside the groups (not learned) never count."""
    vector = np.full(size, np.inf)
    for group, threshold in zip(groups, thresholds, strict=True):
        vector[group] = threshold
    return vector


def _on(preds: Predictions, ids: list[int]) -> Predictions:
    """The rows of ``ids`` only (pycocotools refuses results on images outside its set)."""
    return preds.subset(np.isin(preds.image_id, ids))


def cross_counts(coco_new: dict, preds: Predictions, vector: np.ndarray, old_seen: int) -> dict[str, Any]:
    """How the counted detections (class-agnostic NMS, per-label thresholds) of the new-product photos fall: objects
    of a new product that an old label also counts (IoU >= MATCH_IOU), and old-label detections per photo."""
    from autocheckout.cl_metrics import box_iou
    from autocheckout.counting import dedup_detections

    ids = [i['id'] for i in coco_new['images']]
    rows = dedup_detections(preds.subset(np.isin(preds.image_id, ids)), NMS)
    labels = rows.label.astype(np.int64)
    keep = rows.score >= np.where(labels < len(vector), vector[np.minimum(labels, len(vector) - 1)], np.inf)
    rows = rows.subset(keep)
    claimed = objects = 0
    for image_id in ids:
        gt = [a['bbox'] for a in coco_new['annotations'] if a['image_id'] == image_id]
        if not gt:
            continue
        gt = np.array([[x, y, x + w, y + h] for x, y, w, h in gt], dtype=np.float64)
        mine = rows.image_id == image_id
        old = mine & (rows.label < old_seen)
        iou = box_iou(gt, rows.boxes[old])
        claimed += int((iou >= MATCH_IOU).any(axis=1).sum()) if iou.size else 0
        objects += len(gt)
    n_old = int((rows.label < old_seen).sum())
    return {'objects': objects, 'claimed_by_old_label': claimed, 'cross_rate': claimed / objects if objects else 0.0,
            'old_label_detections': n_old, 'old_label_detections_per_image': n_old / len(ids) if ids else 0.0}


def taken_by_new_label(coco_full: dict, preds: Predictions, ids: list[int], old_seen: int) -> dict[str, Any]:
    """Old objects of ``ids`` whose best box (highest score among the top-1-per-query rows with IoU >= MATCH_IOU,
    before the NMS) has a new label: the new label then wins the class-agnostic NMS and the old object goes uncounted
    (run 1 of task 6: 19.9 % of Val-B objects, AN/task6/08). Diagnostic only, not a gate."""
    from autocheckout.cl_metrics import box_iou

    wanted = set(ids)
    rows = preds.subset(np.isin(preds.image_id, ids)).top1_per_query()
    gt: dict[int, list[list[float]]] = {}
    for a in coco_full['annotations']:
        if a['image_id'] in wanted and a['category_id'] < old_seen:
            x, y, w, h = a['bbox']
            gt.setdefault(a['image_id'], []).append([x, y, x + w, y + h])
    objects = taken = 0
    for image_id, boxes in gt.items():
        mine = rows.image_id == image_id
        objects += len(boxes)
        if not mine.any():
            continue
        iou = box_iou(np.array(boxes, dtype=np.float64), rows.boxes[mine])
        scores = np.where(iou >= MATCH_IOU, rows.score[mine][None, :], -np.inf)
        covered = np.isfinite(scores).any(axis=1)
        taken += int((covered & (rows.label[mine][scores.argmax(axis=1)] >= old_seen)).sum())
    return {'objects': objects, 'taken': taken, 'rate': taken / objects if objects else 0.0}


def score_stage(val_full: dict, cfg: TaskConfig, preds: Predictions, *, stage: int, old_stage: int,
                calib: list[int], val_b: list[int], new_val: list[int]) -> dict[str, Any]:
    """Scores of a model that has learned up to ``stage`` (``old_stage``: the seed; ``old_stage + 1``: with the new
    task). Old-group thresholds on ``calib`` (cAcc-M2 counting the old labels), the new-task threshold on ``new_val``
    with the old ones fixed; then cAcc-M2 of Val-B and of the new photos with every learned label counted."""
    from autocheckout.cl_metrics import group_ap, load_coco
    from autocheckout.feedback_fix import subset
    from tools.al_score import count_on, score
    from tools.eval_count import CALIBRATION_GRID

    old_groups, old_names = _groups(cfg, old_stage)
    old = score(val_full, cfg, preds, calib, val_b, groups=(old_groups, old_names))
    thresholds, groups, names = [old['thresholds'][n] for n in old_names], old_groups, old_names
    old_seen = cfg.seen_classes(old_stage)
    if stage > old_stage:
        groups, names = _groups(cfg, stage)
        candidates = [(count_on(val_full, cfg, preds, new_val, [*thresholds, t], groups=groups)['cAcc_m2'], -t)
                      for t in CALIBRATION_GRID]
        thresholds = [*thresholds, -max(candidates)[1]]  # best cAcc, ties to the smaller threshold
    valb = count_on(val_full, cfg, preds, val_b, thresholds, groups=groups)
    new_coco = subset(val_full, new_val)
    new_labels = list(cfg.task(old_stage + 1).labels)
    coco_valb = load_coco(subset(val_full, val_b))
    report = {'stage': stage, 'thresholds': dict(zip(names, thresholds, strict=True)), 'nms_iou': NMS,
              'calib_images': old['calib_images'], 'calib_cAcc': old['calib_cAcc'],
              'old': {'images': valb['target_images'], 'cAcc_m2': valb['cAcc_m2'],
                      'cAcc_m2_old_labels_only': old['cAcc_m2'],
                      **{k: valb[k] for k in ('ACD', 'mCCD', 'mCIoU')},
                      'AP50': group_ap(coco_valb, _on(preds, val_b), list(range(old_seen)))['AP50'],
                      'right_images': valb['right_images']},
              'new': {'images': len(new_val),
                      **cross_counts(new_coco, preds, _vector(groups, thresholds, cfg.num_classes), old_seen)}}
    if stage > old_stage:
        report['old']['taken_by_new_label'] = taken_by_new_label(val_full, preds, val_b, old_seen)
        coco_new = load_coco(new_coco)
        all_labels = count_on(val_full, cfg, preds, new_val, thresholds, groups=groups)
        filtered = count_on(val_full, cfg, preds, new_val, thresholds[-1:], groups=groups[-1:])
        on_new = _on(preds, new_val)
        per_sku = {label: group_ap(coco_new, on_new, [label])['AP50'] for label in new_labels}
        report['new'].update(AP50=group_ap(coco_new, on_new, new_labels)['AP50'], cAcc_m2=all_labels['cAcc_m2'],
                             cAcc_m2_new_labels_only=filtered['cAcc_m2'],
                             **{k: all_labels[k] for k in ('ACD', 'mCCD', 'mCIoU')},
                             AP50_per_sku={str(k): v for k, v in per_sku.items()},
                             objects_per_sku={str(k): sum(1 for a in new_coco['annotations'] if a['category_id'] == k)
                                              for k in new_labels})
    return report


def sign_test(base_right: list[int], new_right: list[int]) -> dict[str, Any]:
    """Val-B images counted right by one model and wrong by the other; two-sided sign test."""
    from scipy.stats import binomtest

    base, new = set(base_right), set(new_right)
    fixed, broken = len(new - base), len(base - new)
    p = binomtest(broken, fixed + broken, 0.5).pvalue if fixed + broken else 1.0
    return {'fixed': fixed, 'broken': broken, 'sign_test_p': float(p)}


def verdict(report: dict, baseline: dict, gate: dict = GATE) -> tuple[str, str, bool]:
    """(go / stop, reason, the shop must filter labels) of the new task against the baseline (N7)."""
    failed = []
    ap_new = report['new']['AP50']
    if not ap_new >= gate['new_ap50']:  # NaN fails too
        failed.append(f"AP50 of the new products {ap_new:.4f} (needs >= {gate['new_ap50']})")
    ap_base, ap_old = baseline['old']['AP50'], report['old']['AP50']
    if ap_base - ap_old > gate['old_ap50_drop']:
        failed.append(f"AP50 of the old products {ap_base:.4f} -> {ap_old:.4f} (may fall {gate['old_ap50_drop']})")
    c_base, c_old = baseline['old']['cAcc_m2'], report['old']['cAcc_m2']
    if c_base - c_old > gate['old_cacc_drop']:
        failed.append(f'cAcc-M2 of Val-B {c_base:.4f} -> {c_old:.4f} (may fall {gate["old_cacc_drop"]})')
    cross = report['new']['cross_rate']
    must_filter = cross > gate['cross_rate']
    summary = (f"new AP50 {ap_new:.4f}; old AP50 {ap_base:.4f} -> {ap_old:.4f}; Val-B cAcc-M2 {c_base:.4f} -> "
               f"{c_old:.4f}; new products also counted as old {cross:.1%} (baseline "
               f"{baseline['new']['cross_rate']:.1%})" + ('; the shop must filter the labels it does not sell'
                                                          if must_filter else ''))
    return ('stop', f'{summary}: failed ' + '; '.join(failed), must_filter) if failed else ('go', summary, must_filter)


# --- plan -----------------------------------------------------------------------------------------------------------
def _new_fixture(root: Path, eval_root: Path) -> tuple[Path, str]:
    """Dry run: a package of one new product (the fixture's label-0 objects under the next label), its images being
    the fixture's. Returns (package root, folder of the task files)."""
    package = root / 'new_products'
    if not (package / 'products.json').is_file():
        cfg = TaskConfig.load(eval_root / 'task_config.json')
        label = cfg.seen_classes(len(cfg.data_tasks))
        toy = root / 'toy/tasks'
        for split, source, shift in (('train', toy / 'train_full.json', 0), ('val', toy / 'val_full.json', 1000),
                                     ('test', toy / 'test_full.json', 1000)):
            full = load_json(source)
            save_json(package / f'tasks/{split}_task_{len(cfg.data_tasks) + 1}.json', {
                'images': [{**i, 'id': i['id'] + shift} for i in full['images']],
                'annotations': [{**a, 'id': a['id'] + 10_000, 'image_id': a['image_id'] + shift, 'category_id': label}
                                for a in full['annotations'] if a['category_id'] == 0],
                'categories': [{'id': label, 'name': 'new0'}]})
        save_json(package / 'products.json', [{'pdp_class_id': label + i, 'name': f'new{i}'}
                                              for i in range(DRY_NEW_LABELS)])
        val_ids = [i['id'] for i in load_json(eval_root / 'val_full.json')['images']]
        save_json(root / 'val_a_split.json', {'calib': val_ids[:1], 'pool': []})
    return package, 'tasks'


def _md5_ids(ids) -> str:
    import hashlib
    import json

    return hashlib.md5(json.dumps(sorted(int(i) for i in ids)).encode()).hexdigest()


def _merge(base: dict, extra: dict) -> dict:
    """val_full of the campaign: RPC val and the val photos of the new products (ids must not collide)."""
    if {i['id'] for i in base['images']} & {i['id'] for i in extra['images']} or \
            {a['id'] for a in base['annotations']} & {a['id'] for a in extra['annotations']}:
        raise ValueError('image or object ids of the new products collide with RPC val')
    known = {c['id'] for c in base['categories']}
    return {**base, 'images': base['images'] + extra['images'],
            'annotations': base['annotations'] + extra['annotations'],
            'categories': base['categories'] + [c for c in extra['categories'] if c['id'] not in known]}


def replay_images(source: dict, parts: dict[str, int], *, min_per_class: int, learned: int, exclude: set[str],
                  seed: int) -> dict:
    """Option A of AN/task6/08: RPC images replayed in the new task so that the new rows see old products as
    negatives. From ``source`` (the seed's training file, its oversampled images once), ``parts[prefix]`` images whose
    file name starts with ``prefix`` (feedback_fix.select_replay: each old class first up to ``min_per_class``
    objects). Refused: a file name (last part) in ``exclude`` (Val-B, calib, the new product photos) or a label the
    seed has not learned."""
    from autocheckout.feedback_fix import select_replay, subset

    first: dict[str, int] = {}
    for image in source['images']:
        first.setdefault(image['file_name'], image['id'])
    leaked = sorted({name.rsplit('/', 1)[-1] for name in first} & exclude)
    if leaked:
        raise ValueError(f'{len(leaked)} replay images are Val-B, calib or new-product photos, e.g. {leaked[:3]}')
    unique = subset(source, sorted(first.values()))
    if any(a['category_id'] >= learned for a in unique['annotations']):
        raise ValueError(f'replay labels must be classes the seed learned (< {learned})')
    chosen: list[int] = []
    for k, (prefix, n) in enumerate(sorted(parts.items())):
        part = subset(unique, [i['id'] for i in unique['images'] if i['file_name'].startswith(prefix)])
        if len(part['images']) < n:
            raise ValueError(f'replay part {prefix!r}: only {len(part["images"])} images, {n} wanted')
        chosen += [i['id'] for i in select_replay(part, [], n, min_per_class, range(learned), seed + k)['images']]
    return subset(unique, chosen)


def _with_replay(train: dict, replay: dict) -> dict:
    """The new task's training file followed by the replay images, renumbered after its largest ids (``orig_id``
    keeps the seed's id, ``replay`` marks them)."""
    image_base = max(i['id'] for i in train['images'])
    ann_base = max((a['id'] for a in train['annotations']), default=0)
    new_id = {image['id']: image_base + k for k, image in enumerate(replay['images'], start=1)}
    known = {c['id'] for c in train['categories']}
    return {**train,
            'images': train['images'] + [{**i, 'id': new_id[i['id']], 'orig_id': i['id'], 'replay': True}
                                         for i in replay['images']],
            'annotations': train['annotations'] + [{**a, 'id': ann_base + k, 'image_id': new_id[a['image_id']]}
                                                   for k, a in enumerate(replay['annotations'], start=1)],
            'categories': sorted([*train['categories'], *(c for c in replay['categories'] if c['id'] not in known)],
                                 key=lambda c: c['id'])}


def _link_images(target: Path, sources: list[tuple[dict, Path]], named: dict[str, Path] | None = None) -> None:
    """One image folder for the prediction pass: the first path part of every file name linked to its folder
    (``named``: first path parts linked to given folders, e.g. the replay's real/ -> rpc)."""
    links: dict[str, Path] = {name: Path(folder).resolve() for name, folder in (named or {}).items()}
    for coco, root in sources:
        for image in coco['images']:
            name = image['file_name'].split('/')[0]
            source = (root / name).resolve()
            if links.setdefault(name, source) != source:
                raise ValueError(f'{name} of the image file names is in both {links[name]} and {source}')
    target.mkdir(parents=True, exist_ok=True)
    for name, source in links.items():
        if not source.exists():
            raise FileNotFoundError(source)
        link = target / name
        if link.is_symlink() and link.resolve() != source:
            raise ValueError(f'{link} points to {link.resolve()}, expected {source}')
        if not link.is_symlink():
            link.symlink_to(source)


def prepare(settings: dict) -> dict:
    """Plan of a mode-new campaign: the seed (task-5 model of an improve run) and the recipe of its round, the new
    config, the campaign's annotation and image folders, the splits. A changed setting or input in an existing
    campaign is refused."""
    from autocheckout.training_workflow import BACKBONE_REVISION, DETECTOR_REVISION, campaign_root, resolve
    from tools.make_new_task_config import new_task_config

    root = campaign_root(settings)
    dry = bool(settings.get('dry_run', True))
    if dry:
        _, eval_root, images, model = _toy_releases(root / 'fixture')
        package, tasks = _new_fixture(root / 'fixture', eval_root)
        raw, calib_path = images, root / 'fixture/val_a_split.json'
        seed = resolve(settings['seed_campaign'])
        seed_run = seed / 'runs' / PDP
        synth_root = photos = images
        detector = backbone = ''
    else:
        data = Path(os.environ.get('AC_DATA') or resolve(settings['data_root']))
        paths = {k: data / v for k, v in DATA_ROOT_LAYOUT.items()}
        paths.update({k: data / settings[k] for k in DATA_ROOT_LAYOUT if settings.get(k)})
        eval_root, raw, seed, calib_path = (paths[k] for k in ('eval_root', 'raw_root', 'seed_campaign', 'calib_split'))
        seed_run, package, tasks = data / settings['seed_run'], data / settings['new_root'], settings['new_tasks']
        synth_root, photos, model = package, package, ''
        cache = paths['pretrained_root']
        detector = str(cache / f'models--SenseTime--deformable-detr/snapshots/{DETECTOR_REVISION}')
        backbone = str(cache / f'models--timm--convnextv2_base.fcmae_ft_in22k_in1k/snapshots/{BACKBONE_REVISION}'
                       / 'model.safetensors')
        for path in (Path(detector) / 'model.safetensors', Path(backbone)):
            if not path.is_file():
                raise FileNotFoundError(f'pinned pretrained weights missing: {path}')
    seed_plan = load_json(seed / 'plan.json')
    if seed_plan['dry_run'] != dry:
        raise ValueError('a dry run and a real run cannot seed each other')
    seed_config = eval_root / 'task_config.json'
    if sha256_file(seed_config) != seed_plan['task_config_sha256']:
        raise ValueError(f'{seed_config} is not the task config of the seed campaign')
    base = TaskConfig.load(seed_config)
    old_stage = len(base.data_tasks)
    seed5 = seed_run / f'task_{old_stage}/task_final.pth'
    seed_md5 = md5_file(seed5)
    if settings.get('seed_task5_md5') and settings['seed_task5_md5'] != seed_md5:
        raise ValueError(f'{seed5} has md5 {seed_md5}, the settings expect {settings["seed_task5_md5"]}')
    cfg = new_task_config(base, load_json(package / 'products.json'),
                          reserve_after=int(settings.get('reserve_after', 1 if dry else 25)),
                          name=settings.get('config_name', 'dry-run+new' if dry else f'{base.name}+new'))
    new_task = old_stage + 1
    new_labels = set(cfg.task(new_task).labels)
    files = {split: load_json(package / tasks / f'{split}_task_{new_task}.json') for split in ('train', 'val', 'test')}
    for split, coco in files.items():
        wrong = {a['category_id'] for a in coco['annotations']} - new_labels
        if wrong:
            raise ValueError(f'{split}_task_{new_task}.json has labels {sorted(wrong)} outside the new task')
        missing = [i['file_name'] for i in coco['images']
                   if not ((synth_root if split == 'train' else photos) / i['file_name']).is_file()]
        if missing:
            raise FileNotFoundError(f'{split}: {len(missing)} images missing, e.g. {missing[:3]}')
    val_full = load_json(eval_root / 'val_full.json')
    split = load_json(calib_path)
    if not dry and split.get('val_a_md5') != _md5_ids([*split['pool'], *split['calib']]):
        raise ValueError(f'{calib_path}: pool + calib is not the Val-A of its val_a_md5')
    val_b = sorted({i['id'] for i in val_full['images']} - set(split['pool']) - set(split['calib']))

    inputs = root / 'inputs'
    task_dir = inputs / 'tasks'
    cfg.save(inputs / 'task_config.json')
    train = files['train']
    pick = sorted(np.random.default_rng(0).choice(len(train['images']), min(PILOT_IMAGES, len(train['images'])),
                                                  replace=False).tolist())
    pilot_ids = {train['images'][k]['id'] for k in pick}
    # run 3 of task 6 (AN/task6/09): settings 'hyperparameters' override keys of both variants (e.g. lr_old 0)
    override = settings.get('hyperparameters') or {}
    if set(override) - set(HYPERPARAMETERS['task']):
        raise ValueError(f"unknown hyperparameters {sorted(set(override) - set(HYPERPARAMETERS['task']))}")
    hyperparameters = {v: {**hp, **override} for v, hp in HYPERPARAMETERS.items()}
    replay, replay_plan = settings.get('replay'), {}
    if replay:  # option A of AN/task6/08: RPC images of the seed's training file in the new task
        source = root / 'fixture/toy/tasks/train_full.json' if dry else data / replay['source']
        names = {i['id']: i['file_name'].rsplit('/', 1)[-1] for i in val_full['images']}
        exclude = {names[i] for i in [*val_b, *split['calib']]} | {
            i['file_name'].rsplit('/', 1)[-1] for s in ('val', 'test') for i in files[s]['images']}
        chosen = replay_images(load_json(source), replay['parts'], min_per_class=int(replay['min_per_class']),
                               learned=base.seen_classes(old_stage), exclude=exclude, seed=int(replay.get('seed', 0)))
        train_root, synth_root = synth_root, inputs / 'train_images'
        if replay.get('roots'):  # first path part -> folder under the data root (real/ -> rpc, v2/ -> synth_v2/images)
            _link_images(synth_root, [(files['train'], train_root)],
                         named={name: data / folder for name, folder in replay['roots'].items()})
        else:
            _link_images(synth_root, [(files['train'], train_root), (chosen, raw)])
        missing = [i['file_name'] for i in chosen['images'] if not (synth_root / i['file_name']).is_file()]
        if missing:
            raise FileNotFoundError(f'replay: {len(missing)} images missing, e.g. {missing[:3]}')
        train = _with_replay(train, chosen)
        replayed = [i['id'] for i in train['images'] if i.get('replay')]
        pick = np.random.default_rng(1).choice(len(replayed), min(PILOT_IMAGES, len(replayed)), replace=False)
        pilot_ids |= {replayed[k] for k in pick.tolist()}
        per_class = np.bincount([a['category_id'] for a in chosen['annotations']],
                                minlength=base.seen_classes(old_stage))
        replay_plan = {'replay': {**replay, 'source_sha256': sha256_file(source), 'images': len(chosen['images']),
                                  'objects': len(chosen['annotations']), 'least_objects_per_class': int(per_class.min())}}
    written = {
        f'train_task_{new_task}.json': train,
        f'train_task_{new_task}_pilot.json': {
            **train, 'images': [i for i in train['images'] if i['id'] in pilot_ids],
            'annotations': [a for a in train['annotations'] if a['image_id'] in pilot_ids]},
        f'val_task_{new_task}.json': files['val'],
        # read by PDP's validation loader when the expanded task-5 model only predicts (baseline); never scored
        f'val_task_{old_stage}.json': files['val'],
        'val_full.json': _merge(val_full, files['val']),
        'test_full.json': files['test'],
    }
    for name, coco in written.items():
        save_json(task_dir / name, coco)
    splits = {'calib': sorted(split['calib']), 'val_b': val_b,
              'new_val': sorted(i['id'] for i in files['val']['images']),
              'new_test': sorted(i['id'] for i in files['test']['images'])}
    save_json(inputs / 'splits.json', splits)
    _link_images(inputs / 'images', [(val_full, raw), (files['val'], photos), (files['test'], photos)])
    plan = dict(
        campaign_id=settings['campaign_id'], round=int(settings['round']), dry_run=dry, mode='new', root=str(root),
        runs=str(root / 'runs'), raw_root=str(inputs / 'images'), task_dir=str(task_dir),
        task_config=str(inputs / 'task_config.json'), synth_root=str(synth_root), model_config=str(model),
        detector_dir=detector, backbone_weights=backbone, eval_root=str(eval_root), package=str(package),
        **{k: seed_plan[k] for k in RECIPE},
        seed_root=str(seed), seed_run=str(seed_run), seed_task5=str(seed5), seed_task_config=str(seed_config),
        seed=dict(campaign_id=seed_plan['campaign_id'], round=seed_plan['round'], task5_md5=seed_md5),
        old_stage=old_stage, new_task=new_task, learned=base.seen_classes(old_stage), n_classes=cfg.num_classes,
        new_labels=sorted(new_labels), hyperparameters=hyperparameters, gate=GATE, pilot_images=len(pilot_ids),
        train_images=len(train['images']), task_config_sha256=sha256_file(inputs / 'task_config.json'),
        release_config_sha256=None, **replay_plan,
        input_sha256={**{name: sha256_file(task_dir / name) for name in sorted(written)},
                      'splits.json': sha256_file(inputs / 'splits.json'),
                      'products.json': sha256_file(package / 'products.json')})
    path = root / 'plan.json'
    locations = {'root', 'runs', 'raw_root', 'task_dir', 'task_config', 'synth_root', 'model_config', 'detector_dir',
                 'backbone_weights', 'eval_root', 'package', 'seed_root', 'seed_run', 'seed_task5', 'seed_task_config'}
    if path.is_file():
        previous = load_json(path)
        if {k: v for k, v in previous.items() if k not in locations} != {k: v for k, v in plan.items()
                                                                         if k not in locations}:
            raise ValueError('settings or inputs changed in an existing campaign; start a new campaign_id')
    save_json(path, plan, indent=2)
    return plan


# --- runs -----------------------------------------------------------------------------------------------------------
def _train(plan: dict, record: dict, attempt: Path, experiment: str, task: int, splits: str, hp: dict | None = None,
           suffix: str = '') -> int:
    """Train task ``task`` of ``experiment`` (or only predict ``splits`` when its task_final.pth exists) through
    scripts/run_exp.sh with configs/exp/native/EXP-NEW.sh."""
    from autocheckout.training_workflow import REPO

    hp = hp or HYPERPARAMETERS['task']
    runs = Path(plan['runs'])
    output = runs / experiment
    output.mkdir(parents=True, exist_ok=True)
    fsa_hf = str(runs / FSA / 'task_1/hf_model')
    lines = [f'source "$REPO/configs/exp/native/{EXP}.sh"']
    environment = _environment(plan, experiment, runs, FSA_HF=fsa_hf, START_TASK=task, N_TASKS=task,
                               EPOCHS=hp['epochs'], PRED_SPLITS=splits, STOP_AFTER_STEPS=0, LR=hp['lr'],
                               LR_OLD=hp['lr_old'], LR_DROP=hp['lr_drop'], AUGMENT=hp['augment'], TRAIN_SUFFIX=suffix)
    if plan['dry_run']:
        lines.append(f'ARGS+=({shlex.join(DRY_MODEL_ARGS + ["--model_config", fsa_hf])})')
        environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    lines.append(f"ARGS+=(--n_classes {plan['n_classes']})")
    config = output / 'resolved_config.sh'
    config.write_text('\n'.join(lines) + '\n')
    resolved = subprocess.run(['bash', '-ec', 'source "$CONFIG"; printf "%s\\0" "${ARGS[@]}"'],
                              env={**os.environ, **environment, 'CONFIG': str(config)},
                              capture_output=True, text=True, check=True)
    save_json(output / 'resolved_cli.json', dict(arguments=resolved.stdout.split('\0')[:-1],
                                                environment=environment), indent=2)
    return _run(record, attempt, f'{experiment}_task{task}_{splits}', ['bash', str(REPO / 'scripts/run_exp.sh'),
                str(config)], environment)


def _score(plan: dict, run: Path, stage: int, split: str = 'val') -> dict:
    task_dir = Path(plan['task_dir'])
    splits = load_json(Path(plan['root']) / 'inputs/splits.json')
    preds = load_predictions(run / f'task_{stage}/pred_{split}.npz')
    cfg = TaskConfig.load(plan['task_config'])
    if split == 'val':
        return score_stage(load_json(task_dir / 'val_full.json'), cfg, preds, stage=stage,
                           old_stage=plan['old_stage'], calib=splits['calib'], val_b=splits['val_b'],
                           new_val=splits['new_val'])
    raise ValueError(split)


def _hours(run: Path, stage: int) -> float:
    info = load_json(run / f'task_{stage}/run_info.json')
    return round(info.get('total_seconds', 0) / 3600, 3)


# --- stages ---------------------------------------------------------------------------------------------------------
def stage_expand(plan: dict) -> dict:
    """The seed's task-5 model with the class slots of the new config (pdp/expand.py, N1); the FSA run of the seed
    campaign is linked (its hf_model builds the processor). The classes the seed learned must stay identical."""
    import torch

    from autocheckout.training_workflow import REPO

    done = Path(plan['root']) / 'stages/expand.json'
    if done.is_file():
        return load_json(done)
    if md5_file(plan['seed_task5']) != plan['seed']['task5_md5']:
        raise ValueError(f"{plan['seed_task5']} changed since this campaign was prepared")
    runs = Path(plan['runs'])
    runs.mkdir(parents=True, exist_ok=True)
    link, fsa = runs / FSA, Path(plan['seed_root']) / 'runs' / FSA
    if not link.is_symlink():
        link.symlink_to(fsa.resolve())
    elif link.resolve() != fsa.resolve():
        raise ValueError(f'{link} points to {link.resolve()}, expected {fsa}')
    out = runs / EXP / f"task_{plan['old_stage']}/task_final.pth"
    with _attempt(plan, 'expand') as (record, attempt):
        if not out.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            _run(record, attempt, 'expand', [sys.executable, str(REPO / 'pdp/expand.py'), '--checkpoint',
                 plan['seed_task5'], '--old-config', plan['seed_task_config'], '--new-config', plan['task_config'],
                 '--learned', str(plan['learned']), '--out', str(out)], {})
    seed, new = (torch.load(p, map_location='cpu')['model'] for p in (plan['seed_task5'], out))
    learned = plan['learned']
    same = all(torch.equal(v[:learned], new[k][:learned]) and torch.equal(v[-1:], new[k][-1:])
               for k, v in seed.items() if 'class_embed' in k)
    if not same or new['class_embed.0.bias'].shape[0] != plan['n_classes']:
        raise ValueError(f'{out}: the classifier rows of the learned classes changed')
    report = {'stage': 'expand', 'learned': learned, 'n_classes': plan['n_classes'], 'checkpoint_md5': md5_file(out),
              'old_rows_identical': same}
    return _finish(plan, 'expand', report, None)


def stage_baseline(plan: dict) -> dict:
    """The expanded model, a task-5 model, predicts val (RPC val + the new photos): the mark of the gate. Its scores
    on Val-B match the seed's (same weights for the learned classes); on the new photos it reports old labels."""
    done = Path(plan['root']) / 'stages/baseline.json'
    if done.is_file():
        return load_json(done)
    _require(plan, 'expand')
    run, stage = Path(plan['runs']) / EXP, plan['old_stage']
    with _attempt(plan, 'baseline') as (record, attempt):
        _train(plan, record, attempt, EXP, stage, 'val')
    report = {'stage': 'baseline', **_score(plan, run, stage)}
    seed_pred = Path(plan['seed_run']) / f'task_{stage}/pred_val.npz'
    if seed_pred.is_file():  # the seed's own predictions on RPC val: the same Val-B cAcc-M2 is expected
        from tools.al_score import score

        splits = load_json(Path(plan['root']) / 'inputs/splits.json')
        cfg = TaskConfig.load(plan['task_config'])
        seed_score = score(load_json(Path(plan['eval_root']) / 'val_full.json'), cfg, load_predictions(seed_pred),
                           splits['calib'], splits['val_b'], groups=_groups(cfg, stage))
        report['seed_val_b_cAcc_m2'] = seed_score['cAcc_m2']
    return _finish(plan, 'baseline', report, None)


def stage_train(plan: dict, variant: str) -> dict:
    """``pilot`` (EXP-NEW-pilot: one epoch on the pilot subset, from the same expanded model, no gate) or ``task``
    (EXP-NEW, the new task, gate N7 against the baseline). A finished run is never trained again."""
    if variant not in HYPERPARAMETERS:
        raise ValueError(f'variant must be one of {sorted(HYPERPARAMETERS)}, got {variant}')
    done = Path(plan['root']) / 'stages' / f'{variant}.json'
    if done.is_file():
        return load_json(done)
    baseline = _require(plan, 'baseline')
    if variant == 'task':
        _require(plan, 'pilot')
    experiment = EXP if variant == 'task' else f'{EXP}-pilot'
    run, task = Path(plan['runs']) / experiment, plan['new_task']
    if variant == 'pilot':
        link, source = run / f"task_{plan['old_stage']}", Path(plan['runs']) / EXP / f"task_{plan['old_stage']}"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(source.resolve())
    with _attempt(plan, variant) as (record, attempt):
        _train(plan, record, attempt, experiment, task, 'val', plan['hyperparameters'][variant],
               suffix='_pilot' if variant == 'pilot' else '')
    scores = _score(plan, run, task)
    comparison = sign_test(baseline['old']['right_images'], scores['old']['right_images'])
    decision, reason, must_filter = verdict(scores, baseline)
    report = {'stage': variant, 'hyperparameters': plan['hyperparameters'][variant], 'train_hours': _hours(run, task),
              'train_images': plan['pilot_images'] if variant == 'pilot' else plan['train_images'], **scores,
              'comparison': comparison, 'shop_must_filter_labels': must_filter}
    gate = None
    if variant == 'task':
        gate = dict(name='new task', decision=decision, reason=reason, enforced=False, **GATE)
        print(f'GATE new task: {decision.upper()} - {reason}')
    else:
        print(f'pilot (no gate): {reason}')
    return _finish(plan, variant, report, gate)


def stage_final_test(plan: dict, confirm: bool = False) -> dict:
    """The test photos of the new products, scored once with the thresholds of the task stage, after its go and only
    with ``confirm=True`` (a dry run tests whatever its gate said). An existing result is never recomputed."""
    from autocheckout.cl_metrics import group_ap, load_coco
    from tools.al_score import count_on

    root = Path(plan['root'])
    done = root / 'stages/final_test.json'
    if done.is_file():
        print(f'test already scored on {load_json(done)["finished_at"]}; not recomputed')
        return load_json(done)
    task = _require(plan, 'task')
    if task['gate']['decision'] != 'go' and not plan['dry_run']:
        raise RuntimeError(f"the new task stopped at its gate ({task['gate']['reason']}): nothing to test")
    if not confirm:
        raise RuntimeError('test is scored once: call stage_final_test(plan, confirm=True) to score it')
    run, stage = Path(plan['runs']) / EXP, plan['new_task']
    with _attempt(plan, 'final_test') as (record, attempt):
        _train(plan, record, attempt, EXP, stage, 'test')
    test = load_json(Path(plan['task_dir']) / 'test_full.json')
    cfg = TaskConfig.load(plan['task_config'])
    preds = load_predictions(run / f'task_{stage}/pred_test.npz')
    preds = _on(preds, [i['id'] for i in test['images']])
    groups, names = _groups(cfg, stage)
    thresholds = [task['thresholds'][n] for n in names]
    ids = [i['id'] for i in test['images']]
    new_labels = list(cfg.task(stage).labels)
    coco = load_coco(test)
    all_labels = count_on(test, cfg, preds, ids, thresholds, groups=groups)
    filtered = count_on(test, cfg, preds, ids, thresholds[-1:], groups=groups[-1:])
    levels = {}
    for level in ('easy', 'medium', 'hard'):
        chosen = [i['id'] for i in test['images'] if i.get('level') == level]
        if chosen:
            levels[level] = {'images': len(chosen),
                             'cAcc_m2': count_on(test, cfg, preds, chosen, thresholds, groups=groups)['cAcc_m2'],
                             'cAcc_m2_new_labels_only': count_on(test, cfg, preds, chosen, thresholds[-1:],
                                                                 groups=groups[-1:])['cAcc_m2']}
    report = {'stage': 'final_test', 'thresholds': task['thresholds'], 'images': len(ids),
              'AP50': group_ap(coco, preds, new_labels)['AP50'],
              'AP50_per_sku': {str(k): group_ap(coco, preds, [k])['AP50'] for k in new_labels},
              'cAcc_m2': all_labels['cAcc_m2'], 'cAcc_m2_new_labels_only': filtered['cAcc_m2'],
              **{k: all_labels[k] for k in ('ACD', 'mCCD', 'mCIoU')}, 'by_level': levels,
              **cross_counts(test, preds, _vector(groups, thresholds, cfg.num_classes), plan['learned'])}
    return _finish(plan, 'final_test', report, None)


ORDER = ('expand', 'baseline', 'pilot', 'task')
STAGES = {'expand': stage_expand, 'baseline': stage_baseline, 'pilot': lambda p: stage_train(p, 'pilot'),
          'task': lambda p: stage_train(p, 'task')}


def run(settings: dict, stage: str, confirm: bool = False) -> dict:
    """``update_model --mode new --stage <stage>``: the stages before ``stage`` that are not done, then ``stage``;
    ``data`` only prepares the campaign (no GPU) and prints what it holds; ``final_test`` needs ``confirm``."""
    plan = prepare(settings)
    if stage == 'data':
        print(f"{plan['campaign_id']}: new task {plan['new_task']} = labels {plan['new_labels'][0]}.."
              f"{plan['new_labels'][-1]}, n_classes {plan['n_classes']}; train {plan['train_images']} images (pilot "
              f"{plan['pilot_images']}); seed md5 {plan['seed']['task5_md5']}; files in {plan['task_dir']}")
        return plan
    if stage == 'final_test':
        return stage_final_test(plan, confirm=confirm)
    if stage not in ORDER:
        raise ValueError(f'mode new stages are {list(ORDER)}, data and final_test; got {stage}')
    result = {}
    for name in ORDER[:ORDER.index(stage) + 1]:
        result = STAGES[name](plan)
    return result
