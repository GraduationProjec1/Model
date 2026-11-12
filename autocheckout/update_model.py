"""Update a trained model (docs/project/feedback-finetune-plan-2026-11-01.md): one entry point, two modes.

``improve`` retrains the classifier rows of the confused SKU pairs of the round-1 task-5 model on images with every
object labelled (feedback images of Val-A, synthetic images of those SKUs, replay), as task 5 again: no alias, no new
slot, no teacher. Stages, each written once to ``stages/<stage>.json`` and skipped when done:

``split`` (Val-A / Val-B) -> ``baseline`` (round 1 scored on Val-B, feedback images and pairs from Val-A) ->
``pilot`` -> ``e1`` (-> ``e2`` only after an e1 stop) -> ``round2`` -> ``final_test`` (once, with confirm).

``new`` (learn SKUs the model does not know) is Part 2 of the plan and not written yet.

    python -m autocheckout.update_model --mode improve --settings configs/training/update_model_improve.json \\
        --stage pilot
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from autocheckout import feedback_fix as ff
from autocheckout.full_syn import (
    DRY_MODEL_ARGS,
    FSA,
    N_TASKS,
    PDP,
    _attempt,
    _environment,
    _require,
    _run,
    _toy_releases,
    score_val,
    sha256_file,
)
from autocheckout.io import load_json, md5_file, save_json
from autocheckout.predictions import load_predictions, save_predictions
from autocheckout.relearn import _finish
from autocheckout.taskcfg import TaskConfig
from tools import sku_regression as sr

EXP = 'EXP-FB-FT'
RECIPE = ('resolution', 'batch_size', 'effective_batch', 'tf32', 'num_workers', 'checkpoint_every_minutes')
# I8: hyperparameters of each improve run
VARIANTS = {
    'pilot': dict(epochs=4, lr=2e-5, lr_old=5e-6, lr_drop=3),
    'e1': dict(epochs=6, lr=2e-5, lr_old=5e-6, lr_drop=4),
    'e2': dict(epochs=6, lr=2e-5, lr_old=5e-6, lr_drop=4),
    'round2': dict(epochs=2, lr=1e-5, lr_old=2.5e-6, lr_drop=100),
}
# Round 3 (docs/design/2026-11-07-vong-3, 6.5): al_<selection>_r3_<tag>, from the round-2 model. augment (I2) without
# flips; tune_old_private: the private prompts of tasks 1-4 train too; lr_old 0 freezes every shared part.
VARIANTS['r3_base'] = dict(VARIANTS['round2'])
VARIANTS['r3_strong'] = dict(epochs=6, lr=2e-5, lr_old=5e-6, lr_drop=4, augment=1)
VARIANTS['r3_prompt'] = {**VARIANTS['r3_strong'], 'tune_old_private': 1}
VARIANTS['r3_freeze'] = {**VARIANTS['r3_strong'], 'lr_old': 0.0}
R3_TAGS = {'rows': 'r3_base', 'train': 'r3_strong', 'real': 'r3_strong', 'all': 'r3_strong', 'prompt': 'r3_prompt',
           'freeze': 'r3_freeze'}
# Đợt GPU 1 sau vòng 3 (docs/design/2026-11-12-dot-gpu-1): al_<selection>_r3_<tag>_s<N> lặp lại run <tag> với seed
# train N (B4, đo nhiễu seed; dữ liệu của <tag>); al_<selection>_r4 là vòng 4 AL từ model _r3_real (B1, công thức
# r3_strong, tập chọn <selection>_r3 của tools/al_round2.py).
SEED_TAGS = {'real_s1': ('real', 1)}
VARIANTS.update({f'{R3_TAGS[data]}_s{seed}': {**VARIANTS[R3_TAGS[data]], 'seed': seed}
                 for data, seed in SEED_TAGS.values()})
VARIANTS['r4'] = dict(VARIANTS['r3_strong'])
R4_FROM = 'real'  # the round-3 run a round-4 run starts from (decision of 12/11/2026, §6 of the summary)
TRAIN_SEED = 0  # --seed of configs/exp/native/common.sh: every run before B4 trained with it
# Gates of the plan, written before any run. gain: least cAcc-M2 gain on Val-B (strict: must be > 0);
# pair_drop: least relative fall of the confusions inside the retrained pairs; other_drop: largest AP50 fall of the
# SKUs not retrained; max_falls: SKUs (>= 30 objects) that fell by tools.sku_regression.fell (recall -0.03 and
# >= 5 right objects lost; before 12/11/2026: share of right objects -0.02).
# 12/11/2026 (docs/02-cac-lan-train/08-2026-11-12-tong-ket-va-ke-hoach.md, 2.4 and 6): pair_drop 0.30 -> 0.25 and
# max_falls -> 10 with the new fall rule. 18 of 20 variants of rounds 1-3 stopped on 1-9 SKUs losing 3-6 objects while
# cAcc-M2 rose 10-23 points; with these rules 16 of 18 pass and al_margin_k_B500_r3_train (17 falls) still stops.
GATES = {
    'pilot': dict(gain=0.0, strict=True, sign_p=0.05, pair_drop=0.25, other_drop=0.01, max_falls=10),
    'e1': dict(gain=0.0165, strict=False, sign_p=0.05, pair_drop=0.25, other_drop=0.01, max_falls=10),
    'e2': dict(gain=0.0165, strict=False, sign_p=0.05, pair_drop=0.25, other_drop=0.01, max_falls=10),
    'round2': dict(gain=0.0, strict=False, sign_p=None, pair_drop=0.0, other_drop=0.005, max_falls=10),
}
# Round 3: the pilot gate is only reported; the decision follows the criteria of the design (11-C, 11-D).
GATES.update({kind: dict(GATES['pilot']) for kind in ('r3_base', 'r3_strong', 'r3_prompt', 'r3_freeze', 'r4')})
GATES.update({f'{R3_TAGS[data]}_s{seed}': dict(GATES['pilot']) for data, seed in SEED_TAGS.values()})
# Quota of the synthetic images (settings, objects per retrained SKU): from the full-face relearn release, from v2
# for an SKU the relearn release covers, from v2 for an SKU it does not. Without these keys every synthetic image
# holding a retrained SKU is taken (the first version: 29,539 v2 images for the pilot rows, 62,061 for e1).
QUOTA_KEYS = {'quota_relearn': 600, 'quota_v2': 300, 'quota_v2_only': 600}
# settings 'data_root' (or env AC_DATA / RELEARN_DATA_ROOT): the data folder. Its layout is the one data folder of
# autocheckout/datapaths.py (../autocheckout-data: eval/, rpc/, models/full-syn-r1/, relearn/, ...), or a legacy
# self-contained folder made by build_relearn_data.py --self-contained (relearn_fix_v1). Keys set in the settings win.
DATA_ROOT_LAYOUT = {'eval_root': 'eval', 'raw_root': 'rpc', 'seed_campaign': 'models/full-syn-r1',
                    'pretrained_root': 'pretrained/hf-cache', 'prebuilt_release': 'relearn',
                    'synth_v2_root': 'synth_v2'}
LEGACY_LAYOUT = {'eval_root': 'eval', 'raw_root': 'archive', 'seed_campaign': 'seed/full-syn-r1',
                 'pretrained_root': 'pretrained/hf-cache', 'prebuilt_release': '.', 'synth_v2_root': 'images/v2'}
STAGE = N_TASKS  # the improved model is scored as a stage-5 model of the round-1 config




def _kind(name: str) -> str:
    """Hyperparameters and gate of a run: an active-learning variant (``al_<selection>``, built by
    tools/al_build_variant.py) is trained and gated as the pilot, so only its real images differ; its second round
    (``al_<selection>_r2``, tools/al_round2.py) as round2, from the first round's model."""
    if name.startswith('al_') and '_r3_' in name:
        tag = name.rsplit('_r3_', 1)[1]
        if tag in SEED_TAGS:
            data, seed = SEED_TAGS[tag]
            return f'{R3_TAGS[data]}_s{seed}'
        if tag not in R3_TAGS:
            raise ValueError(f'{name}: round-3 tag must be one of {sorted(R3_TAGS) + sorted(SEED_TAGS)}')
        return R3_TAGS[tag]
    if name.startswith('al_'):
        if name.endswith('_r4'):
            return 'r4'
        return 'round2' if name.endswith('_r2') else 'pilot'
    return name


def r3_data_tag(name: str) -> str:
    """The round-3 tag whose training data a run uses: its own, or for a seed repeat (``_r3_real_s1``) the repeated
    run's (``real``)."""
    tag = name.rsplit('_r3_', 1)[1]
    return SEED_TAGS[tag][0] if tag in SEED_TAGS else tag


def _train_seed(hp: dict) -> int:
    return int(hp.get('seed', TRAIN_SEED))


def _al_start(plan: dict, name: str) -> tuple[str, str]:
    """(model to start from, run to compare with) of an active-learning variant."""
    if '_r3_' in name:  # round 3 (and its seed repeats): from the round-2 model of the same selection
        second = name.rsplit('_r3_', 1)[0] + '_r2'
        if _done(plan, second) is None:
            raise RuntimeError(f'{name} starts from {second}: train {second} first')
        return str(Path(plan['runs']) / f'{EXP}-{second}' / f'task_{STAGE}/task_final.pth'), second
    if name.endswith('_r4'):  # round 4: from the round-3 model R4_FROM of the same selection
        third = f'{name.removesuffix("_r4")}_r3_{R4_FROM}'
        if _done(plan, third) is None:
            raise RuntimeError(f'{name} starts from {third}: train {third} first')
        return str(Path(plan['runs']) / f'{EXP}-{third}' / f'task_{STAGE}/task_final.pth'), third
    if not name.endswith('_r2'):
        return plan['seed_task5'], 'baseline'
    first = name.removesuffix('_r2')
    if _done(plan, first) is None:
        raise RuntimeError(f'{name} starts from {first}: train {first} first')
    return str(Path(plan['runs']) / f'{EXP}-{first}' / f'task_{STAGE}/task_final.pth'), first


# --- plan ---------------------------------------------------------------------------------------------------------
def with_data_root(settings: dict) -> dict:
    """The paths of the data folder (``DATA_ROOT_LAYOUT``, else ``LEGACY_LAYOUT``) for the keys the settings leave
    out."""
    from autocheckout.training_workflow import resolve

    root = os.environ.get('AC_DATA') or os.environ.get('RELEARN_DATA_ROOT') or settings.get('data_root')
    if not root:
        return settings
    layout = DATA_ROOT_LAYOUT if (resolve(root) / 'relearn').is_dir() else LEGACY_LAYOUT
    paths = {k: str(Path(root) / v) for k, v in layout.items()}
    return {**paths, **{k: v for k, v in settings.items() if not (k in paths and v is None)}}


def prepare(settings: dict) -> dict:
    """Plan of an improve campaign: the seed (round-1 task-5 model) and its training recipe, the inputs and their
    hashes, the data choices of I2-I7. A changed setting or input in an existing campaign is refused."""
    settings = with_data_root(settings)
    from autocheckout.training_workflow import BACKBONE_REVISION, DETECTOR_REVISION, campaign_root, resolve

    root = campaign_root(settings)
    dry = bool(settings.get('dry_run', True))
    seed = resolve(settings['seed_campaign'])
    seed_plan = load_json(seed / 'plan.json')
    if seed_plan['dry_run'] != dry:
        raise ValueError('a dry run and a real run cannot seed each other')
    if seed_plan.get('pdp_experiment', PDP) != PDP:
        raise ValueError('the seed campaign must be a plain FULL-SYN run')
    relearn_release = relearn_config = None
    if dry:
        v2, eval_root, images, model = _toy_releases(root / 'fixture')
        v2_images = raw = images
        detector = backbone = ''
    else:
        v2, eval_root = resolve(settings['synth_v2_root']), resolve(settings['eval_root'])
        v2_images, raw, model = v2, resolve(settings.get('raw_root', 'data/archive')), ''
        if settings.get('relearn_release'):
            relearn_release = str(resolve(settings['relearn_release']))
            relearn_config = str(resolve(settings['relearn_task_config']))
        cache = resolve(settings.get('pretrained_root', 'data/pretrained/hf-cache'))
        detector = str(cache / f'models--SenseTime--deformable-detr/snapshots/{DETECTOR_REVISION}')
        backbone = str(cache / f'models--timm--convnextv2_base.fcmae_ft_in22k_in1k/snapshots/{BACKBONE_REVISION}'
                       / 'model.safetensors')
        for path in (Path(detector) / 'model.safetensors', Path(backbone)):
            if not path.is_file():
                raise FileNotFoundError(f'pinned pretrained weights missing: {path}')
    task_config = eval_root / 'task_config.json'
    if sha256_file(task_config) != seed_plan['task_config_sha256']:
        raise ValueError(f'{task_config} is not the task config of the seed campaign')
    seed5 = seed / 'runs' / PDP / f'task_{STAGE}/task_final.pth'
    seed_md5 = md5_file(seed5)
    if settings.get('seed_task5_md5') and settings['seed_task5_md5'] != seed_md5:
        raise ValueError(f'{seed5} has md5 {seed_md5}, the settings expect {settings["seed_task5_md5"]}')
    selection = []
    if settings.get('selection'):
        chosen = load_json(resolve(settings['selection']))
        selection = sorted(set(chosen['low']) | set(chosen['partners']) | set(chosen['faces']))
    plan = dict(
        campaign_id=settings['campaign_id'], round=int(settings['round']), dry_run=dry, mode='improve',
        root=str(root), runs=str(root / 'runs'), eval_root=str(eval_root), raw_root=str(raw),
        synth_v2_root=str(v2), synth_v2_images=str(v2_images), relearn_release=relearn_release,
        relearn_task_config=relearn_config, task_config=str(task_config), model_config=str(model),
        detector_dir=detector, backbone_weights=backbone,
        release_root=str(resolve(settings['release_out']) if settings.get('release_out') else root / 'release'),
        **{k: seed_plan[k] for k in RECIPE},
        seed_root=str(seed), seed_task5=str(seed5),
        seed=dict(campaign_id=seed_plan['campaign_id'], round=seed_plan['round'], task5_md5=seed_md5),
        selection=selection, split_seed=int(settings.get('split_seed', 20261101)),
        pilot_pairs=int(settings.get('pilot_pairs', 10)), e1_min_count=int(settings.get('e1_min_count', 10)),
        feedback_repeat=int(settings.get('feedback_repeat', 3)),
        replay_share=float(settings.get('replay_share', 0.3)),
        min_per_class=int(settings.get('min_per_class', 100)),
        task_config_sha256=sha256_file(task_config), release_config_sha256=None,
        input_sha256={name: sha256_file(eval_root / name) for name in ('val_full.json', 'test_full.json')})
    if any(k in settings for k in QUOTA_KEYS):  # synthetic images by a quota of objects per retrained SKU
        plan['quota'] = {k.removeprefix('quota_'): int(settings.get(k, v)) for k, v in QUOTA_KEYS.items()}
    if settings.get('prebuilt_release'):  # a release built ahead (build_relearn_data.py)
        plan['prebuilt_release'] = str(resolve(settings['prebuilt_release']))
        plan['prebuilt_sha256'] = sha256_file(Path(plan['prebuilt_release']) / 'SHA256SUMS')
    path = root / 'plan.json'
    locations = {'root', 'runs', 'eval_root', 'raw_root', 'synth_v2_root', 'synth_v2_images', 'relearn_release',
                 'relearn_task_config', 'task_config', 'model_config', 'detector_dir', 'backbone_weights',
                 'release_root', 'seed_root', 'seed_task5', 'prebuilt_release'}
    if path.is_file():
        previous = load_json(path)
        if {k: v for k, v in previous.items() if k not in locations} != {k: v for k, v in plan.items()
                                                                         if k not in locations}:
            raise ValueError('settings or inputs changed in an existing campaign; start a new campaign_id')
    root.mkdir(parents=True, exist_ok=True)
    save_json(path, plan, indent=2)
    return plan


def _done(plan: dict, stage: str) -> dict | None:
    path = Path(plan['root']) / 'stages' / f'{stage}.json'
    return load_json(path) if path.is_file() else None


def _inputs(plan: dict) -> Path:
    return Path(plan['root']) / 'inputs'


def _cfg(plan: dict) -> TaskConfig:
    return TaskConfig.load(plan['task_config'])


def _labels(plan: dict) -> range:
    return range(_cfg(plan).seen_classes(STAGE))


# --- scoring ------------------------------------------------------------------------------------------------------
def _score(plan: dict, run: Path, name: str) -> dict:
    """Task 5 of ``run`` on Val-B, as ``full_syn.score_val`` scores a stage (policies calibrated on Val-B)."""
    val_b = _inputs(plan) / 'val_b/val_full.json'
    target = Path(plan['root']) / 'scored' / name
    out = target / f'task_{STAGE}'
    if not (out / 'pred_val.npz').is_file():
        preds = load_predictions(run / f'task_{STAGE}/pred_val.npz')
        ids = np.array(sorted(i['id'] for i in load_json(val_b)['images']))
        if not np.isin(ids, preds.image_id).all():
            raise ValueError(f'{run} has no predictions for every image of Val-B')
        subset = preds.subset(np.isin(preds.image_id, ids))
        subset.meta.update(ann_md5=md5_file(val_b))
        save_predictions(out / 'pred_val.npz', subset)
        for name_ in ('task_final.pth', 'run_info.json'):
            link = out / name_
            if not link.is_symlink():
                link.symlink_to((run / f'task_{STAGE}' / name_).resolve())
    report = score_val({'task_config': plan['task_config'], 'task_dir': str(val_b.parent)}, target, STAGE)
    return {k: report[k] for k in ('cAcc_m2', 'cAcc_m1_groups', 'thresholds_groups', 'AP50', 'cAcc_by_level',
                                   'train_hours')}


def _counted(plan: dict, scored: Path, run: Path) -> Any:
    """Boxes of ``run`` (task 5, all of val) counted with the Val-B policy saved in ``scored``."""
    thresholds, nms = sr.policy_vector(load_json(scored / 'calibration_count_m2.json'), _cfg(plan), STAGE)
    return sr.counted(load_predictions(run / f'task_{STAGE}/pred_val.npz'), thresholds, nms)


def _ap50(plan: dict, scored: Path, labels: list[int]) -> float:
    from autocheckout.cl_metrics import group_ap, load_coco

    coco = load_coco(load_json(_inputs(plan) / 'val_b/val_full.json'))
    return group_ap(coco, load_predictions(scored / f'task_{STAGE}/pred_val.npz'), labels)['AP50']


def verdict(gate: dict, cacc_base: float, cacc_new: float, comparison: dict, ap_base: float,
            ap_new: float) -> tuple[str, str]:
    """go / stop of one improve run and the reason, every condition of its gate checked."""
    failed, gain = [], cacc_new - cacc_base
    if (gain <= gate['gain']) if gate['strict'] else (gain < gate['gain']):
        failed.append(f"cAcc-M2 gain {gain:+.4f} (needs {'>' if gate['strict'] else '>='} {gate['gain']})")
    if gate['sign_p'] is not None and comparison['sign_test_p'] >= gate['sign_p']:
        failed.append(f"fixed {comparison['fixed']} vs broken {comparison['broken']}, "
                      f"p {comparison['sign_test_p']:.3g}")
    pairs = comparison['pair_confusions_in_rows']
    if pairs['new'] > pairs['base'] * (1 - gate['pair_drop']):
        failed.append(f"pair confusions {pairs['base']} -> {pairs['new']} (needs a {gate['pair_drop']:.0%} fall)")
    # Every row retrained (r3_all): no "other SKU", AP50 is NaN and the condition cannot be judged. Say so instead of
    # letting the NaN comparison pass silently.
    ap_measured = not (math.isnan(ap_base) or math.isnan(ap_new))
    if ap_measured and ap_base - ap_new > gate['other_drop']:
        failed.append(f"AP50 of the other SKUs {ap_base:.4f} -> {ap_new:.4f} (may fall {gate['other_drop']})")
    if len(comparison['falls']) > gate['max_falls']:
        failed.append(f"{len(comparison['falls'])} SKUs fell (at most {gate['max_falls']})")
    ap_text = f'AP50 other {ap_base:.4f} -> {ap_new:.4f}' if ap_measured else 'AP50 other n/a (every row retrained)'
    summary = (f"cAcc-M2 {cacc_base:.4f} -> {cacc_new:.4f}; fixed {comparison['fixed']}, broken "
               f"{comparison['broken']}; pairs {pairs['base']} -> {pairs['new']}; {ap_text}; "
               f"falls {len(comparison['falls'])}")
    return ('stop', f"{summary}: failed " + '; '.join(failed)) if failed else ('go', summary)


def _criteria(name: str, decision: str, reason: str, **values) -> dict:
    """The gate of an improve run as a label: same ``decision`` / ``reason`` as before, never enforced. The run is
    trained and scored before its gate, so a stop would save no GPU, and scripts/run_al.sh went on past it anyway
    (12/11/2026, docs/02-cac-lan-train/08-2026-11-12-tong-ket-va-ke-hoach.md, 3). The gates that come before an
    expensive stage (FULL-SYN task 1, pseudo-labels, relearn tasks, split12) keep stopping."""
    gate = dict(name=name, decision=decision, reason=reason, enforced=False, **values)
    print(f'GATE {name}: {decision.upper()} - {reason} (label: not enforced)')
    return gate


# --- data and training --------------------------------------------------------------------------------------------
def _union(cocos: list[dict]) -> dict:
    """Images of several COCO files, each image id once (the dry-run fixture repeats the same images)."""
    images, annotations, seen = [], [], set()
    for coco in cocos:
        new = [i for i in coco['images'] if i['id'] not in seen]
        ids = {i['id'] for i in new}
        seen |= ids
        images += new
        annotations += [a for a in coco['annotations'] if a['image_id'] in ids]
    return {'images': images, 'annotations': annotations}


def _prebuilt(plan: dict, name: str) -> Path | None:
    """The variant folder of a release built ahead by build_relearn_data.py (settings ``prebuilt_release``), if any."""
    if not plan.get('prebuilt_release'):
        return None
    variant = Path(plan['prebuilt_release']) / 'variants' / name
    return variant if (variant / 'train_task_5_fix.json').is_file() else None


def _prebuilt_data(plan: dict, name: str, rows: list[int], variant: Path) -> tuple[Path, Path, dict]:
    """A prebuilt variant: images copied in the release (quota per weak SKU), its own train_rows.json. Val-B must be
    the split of this campaign and no Val-B image may be in the training file."""
    release = variant.parent.parent
    eval_root, inputs = Path(plan['eval_root']), _inputs(plan) / name
    val_b = load_json(_inputs(plan) / 'val_b/val_full.json')
    val_b_ids = {i['id'] for i in val_b['images']}
    if {i['id'] for i in load_json(release / 'meta/val_b.json')['images']} != val_b_ids:
        raise ValueError(f'{release}: Val-B differs from the split of this campaign')
    train = load_json(variant / 'train_task_5_fix.json')
    if {i['orig_id'] for i in train['images'] if i.get('source') == 'real'} & val_b_ids:
        raise ValueError(f'{variant}: Val-B images in the training file')
    trained = load_json(variant / 'train_rows.json')
    ff.build_task_dir(inputs, train, load_json(eval_root / 'val_full.json'), val_b, eval_root / 'test_full.json')
    save_json(inputs / 'train_rows.json', trained)
    manifest = {**load_json(variant / 'manifest.json'), 'prebuilt_release': str(release), 'baseline_rows': rows,
                'rows_differ_from_baseline': sorted(set(rows) ^ set(trained))}
    return inputs, release / 'images', manifest


def _build_data(plan: dict, name: str, rows: list[int], feedback_ids: list[int]) -> tuple[Path, Path, dict]:
    """I3-I7: the training file and image folder of one run, its annotation folder and train_rows.json."""
    variant = _prebuilt(plan, name)
    if variant is not None:
        return _prebuilt_data(plan, name, rows, variant)
    inputs, release = _inputs(plan) / name, Path(plan['release_root']) / name
    eval_root = Path(plan['eval_root'])
    val_full = load_json(eval_root / 'val_full.json')
    val_a, val_b = load_json(_inputs(plan) / 'val_a.json'), load_json(_inputs(plan) / 'val_b/val_full.json')
    feedback = ff.images_with(ff.subset(val_a, feedback_ids), rows)
    v2 = [load_json(Path(plan['synth_v2_root']) / f'tasks/synth_v2/train_task_{t}_gt_full.json')
          for t in range(1, STAGE + 1)]
    pool = _union(v2)
    del v2
    parts = [('real', feedback, plan['feedback_repeat'])]
    sources = {'real': Path(plan['raw_root']), 'v2': Path(plan['synth_v2_images'])}
    quota, relearn, covered = plan.get('quota'), None, set()
    if plan['relearn_release']:
        aliases = TaskConfig.load(plan['relearn_task_config']).alias_to_label()
        relearn = _union([ff.resolve_aliases(load_json(Path(plan['relearn_release']) /
                                                       f'tasks/synth_v2/train_task_{t}_gt_full.json'), aliases)
                          for t in (6, 7)])
        covered = set(aliases.values())
    if quota:
        synthetic = ff.select_by_quota(pool, {c: quota['v2'] if c in covered else quota['v2_only'] for c in rows},
                                       seed=plan['split_seed'])
    else:
        synthetic = ff.images_with(pool, rows)
    parts.append(('v2', synthetic, 1))
    n_synth = len(synthetic['images'])
    if relearn is not None:
        relearn = (ff.select_by_quota(relearn, {c: quota['relearn'] for c in rows if c in covered},
                                      seed=plan['split_seed']) if quota else ff.images_with(relearn, rows))
        parts.append(('relearn', relearn, 1))
        sources['relearn'] = Path(plan['relearn_release'])
        n_synth += len(relearn['images'])
    share = plan['replay_share']
    n_replay = round(share / (1 - share) * (len(feedback['images']) * plan['feedback_repeat'] + n_synth))
    replay = ff.select_replay(pool, {i['id'] for i in synthetic['images']}, n_replay, plan['min_per_class'],
                              _labels(plan), seed=plan['split_seed'])
    parts.append(('v2', replay, 1))
    train, manifest = ff.build_release(release, sources, parts, val_full['categories'])
    ff.build_task_dir(inputs, train, val_full, val_b, eval_root / 'test_full.json')
    save_json(inputs / 'train_rows.json', rows)
    objects = manifest['objects_per_label']
    manifest = {**manifest, 'rows': rows, 'feedback_images': len(feedback['images']),
                'synthetic_images': n_synth, 'replay_images': len(replay['images']), 'quota': quota,
                'rows_without_full_faces': [r for r in rows if r not in covered],
                'objects_per_row': {r: objects.get(r, 0) for r in rows},
                'least_objects_any_class': min(objects.get(c, 0) for c in _labels(plan))}
    save_json(release / 'manifest.json', manifest, indent=1)
    return inputs, release, manifest


def _train(plan: dict, record: dict, attempt: Path, name: str, task_dir: Path, release: Path, prev: str,
           hp: dict, splits: str) -> int:
    """Task 5 again through scripts/run_exp.sh with configs/exp/native/EXP-FB-FT.sh, run dir runs/EXP-FB-FT-<name>."""
    from autocheckout.training_workflow import REPO

    runs, experiment = Path(plan['runs']), f'{EXP}-{name}'
    fsa_hf = str(runs / FSA / 'task_1/hf_model')
    view = {**plan, 'task_dir': str(task_dir), 'synth_root': str(release)}
    environment = _environment(view, experiment, runs, START_TASK=STAGE, N_TASKS=STAGE, EPOCHS=hp['epochs'],
                               PRED_SPLITS=splits, STOP_AFTER_STEPS=0, FSA_HF=fsa_hf, PREV_CKPT=prev,
                               TRAIN_ROWS=task_dir / 'train_rows.json', LR=hp['lr'], LR_OLD=hp['lr_old'],
                               LR_DROP=hp['lr_drop'], AUGMENT=hp.get('augment', 0),
                               TUNE_OLD_PRIVATE=hp.get('tune_old_private', 0), SEED=_train_seed(hp))
    output = runs / experiment
    output.mkdir(parents=True, exist_ok=True)
    lines = [f'source "$REPO/configs/exp/native/{EXP}.sh"']
    if plan['dry_run']:
        import shlex
        lines.append(f'ARGS+=({shlex.join(DRY_MODEL_ARGS + ["--model_config", fsa_hf])})')
        environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    config = output / 'resolved_config.sh'
    config.write_text('\n'.join(lines) + '\n')
    resolved = subprocess.run(['bash', '-ec', 'source "$CONFIG"; printf "%s\\0" "${ARGS[@]}"'],
                              env={**os.environ, **environment, 'CONFIG': str(config)},
                              capture_output=True, text=True, check=True)
    save_json(output / 'resolved_cli.json', dict(arguments=resolved.stdout.split('\0')[:-1],
                                                environment=environment), indent=2)
    return _run(record, attempt, f'{experiment}_{splits}', ['bash', str(REPO / 'scripts/run_exp.sh'), str(config)],
                environment)


def _link_fsa(plan: dict) -> None:
    link, source = Path(plan['runs']) / FSA, Path(plan['seed_root']) / 'runs' / FSA
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.is_symlink():
        link.symlink_to(source.resolve())
    elif link.resolve() != source.resolve():
        raise ValueError(f'{link} points to {link.resolve()}, expected {source}')


# --- stages -------------------------------------------------------------------------------------------------------
def stage_split(plan: dict) -> dict:
    """I2: Val-A (feedback) and Val-B (scoring)."""
    done = _done(plan, 'split')
    if done:
        return done
    val_a, val_b = ff.split_val(load_json(Path(plan['eval_root']) / 'val_full.json'), plan['split_seed'])
    save_json(_inputs(plan) / 'val_a.json', val_a)
    save_json(_inputs(plan) / 'val_b/val_full.json', val_b)
    _link_fsa(plan)
    levels = lambda coco: {lv: sum(1 for i in coco['images'] if i.get('level') == lv)  # noqa: E731
                           for lv in ('easy', 'medium', 'hard')}
    report = {'stage': 'split', 'val_a': {'images': len(val_a['images']), 'levels': levels(val_a),
                                          'sha256': sha256_file(_inputs(plan) / 'val_a.json')},
              'val_b': {'images': len(val_b['images']), 'levels': levels(val_b),
                        'sha256': sha256_file(_inputs(plan) / 'val_b/val_full.json')}}
    return _finish(plan, 'split', report, None)


def stage_baseline(plan: dict) -> dict:
    """The round-1 model on Val-B (the mark of every gate); its wrong images and confusion pairs on Val-A (I3, I4)."""
    done = _done(plan, 'baseline')
    if done:
        return done
    _require(plan, 'split')
    seed_run = Path(plan['seed_root']) / 'runs' / PDP
    score = _score(plan, seed_run, 'baseline')
    counted = _counted(plan, Path(plan['root']) / 'scored/baseline', seed_run)
    val_a = load_json(_inputs(plan) / 'val_a.json')
    wrong = ff.wrong_images(val_a, counted, _labels(plan))
    pairs = ff.confusion_pairs(val_a, counted, _labels(plan))
    rows = {'pilot': ff.pair_rows(pairs, top=plan['pilot_pairs']),
            'e1': ff.pair_rows(pairs, min_count=plan['e1_min_count'], extra=plan['selection'])}
    if not rows['pilot'] and plan['dry_run']:  # the six-label fixture may have no confusion: still run the chain
        rows = {'pilot': [0, 1], 'e1': [0, 1, 2]}
    if not rows['pilot']:
        raise ValueError('the round-1 model confuses no pair of SKUs on Val-A: nothing to improve')
    report = {'stage': 'baseline', **score, 'val_a_wrong_images': len(wrong), 'wrong_images': wrong,
              'pairs_top50': [list(p) for p in pairs[:50]], 'rows': rows}
    return _finish(plan, 'baseline', report, None)


def _fit(plan: dict, run: Path, data: dict) -> dict:
    """Round 3 (docs/design/2026-11-07-vong-3, §6.7): cAcc of the Val-A images an AL run trained on and of the pool
    images it did not (tools/fit_check.py, Val-A-calib thresholds); nothing for a run without an AL selection."""
    if not data.get('selection') or not data.get('prebuilt_release'):
        return {}
    from tools.fit_check import fit_scores

    active = Path(data['prebuilt_release']) / 'active'
    return fit_scores(load_json(Path(plan['eval_root']) / 'val_full.json'), _cfg(plan),
                      load_predictions(run / f'task_{STAGE}/pred_val.npz'),
                      load_json(active / 'splits/val_a_split.json'),
                      load_json(active / 'selections' / f"{data['selection']}.json")['image_ids'])


def _run_variant(plan: dict, name: str, rows: list[int], feedback: list[int], prev: str, reference: str) -> dict:
    """Data, training, Val-B scores and gate of one improve run, against the scored run ``reference``."""
    with _attempt(plan, name) as (record, attempt):
        task_dir, release, data = _build_data(plan, name, rows, feedback)
        _train(plan, record, attempt, name, task_dir, release, prev, VARIANTS[_kind(name)], splits='val')
    rows = load_json(task_dir / 'train_rows.json')  # a prebuilt release brings its own rows
    run, root = Path(plan['runs']) / f'{EXP}-{name}', Path(plan['root'])
    score = _score(plan, run, name)
    ref_run = Path(plan['seed_root']) / 'runs' / PDP if reference == 'baseline' else \
        Path(plan['runs']) / f'{EXP}-{reference}'
    val_b = load_json(_inputs(plan) / 'val_b/val_full.json')
    ids = np.array([i['id'] for i in val_b['images']])
    base = _counted(plan, root / 'scored' / reference, ref_run)
    new = _counted(plan, root / 'scored' / name, run)
    comparison = sr.compare(val_b, base.subset(np.isin(base.image_id, ids)), new.subset(np.isin(new.image_id, ids)),
                            _labels(plan), rows)
    save_json(root / 'scored' / name / 'sku_regression.json', comparison, indent=1)
    others = [label for label in _labels(plan) if label not in set(rows)]
    ap_base = _ap50(plan, root / 'scored' / reference, others) if others else float('nan')
    ap_new = _ap50(plan, root / 'scored' / name, others) if others else float('nan')
    cacc_base = load_json(root / 'stages' / f'{reference}.json')['cAcc_m2']
    decision, reason = verdict(GATES[_kind(name)], cacc_base, score['cAcc_m2'], comparison, ap_base, ap_new)
    gate = _criteria(f'improve {name}', decision, reason, **GATES[_kind(name)])
    report = {'stage': name, 'reference': reference, 'rows': rows, 'data': data,
              'hyperparameters': VARIANTS[_kind(name)],
              **score, **_fit(plan, run, data), 'AP50_other': {'base': ap_base, 'new': ap_new},
              'comparison': {k: v for k, v in comparison.items() if k != 'per_sku'}}
    return _finish(plan, name, report, gate)


def stage_data(plan: dict, variant: str) -> dict:
    """Only the training data of one run (no GPU): to check the numbers before renting a machine. Training builds
    the same data again (same seed), so nothing here has to be copied."""
    baseline = _require(plan, 'baseline')
    key = 'pilot' if variant.startswith('al_') else variant
    rows = list(_labels(plan)) if variant == 'e2' else baseline['rows'][key]
    task_dir, _, manifest = _build_data(plan, variant, rows, baseline['wrong_images'])
    per_row = manifest.get('objects_per_row') or manifest.get('objects_per_weak_sku') or {None: 0}
    trained = load_json(task_dir / 'train_rows.json')
    print(f"{variant}: {manifest['images']} images ({manifest['images_per_source']}), {manifest['objects']} objects; "
          f"{len(trained)} rows trained ({len(rows)} from the baseline, differ: "
          f"{manifest.get('rows_differ_from_baseline', [])}), objects per row {min(per_row.values())}.."
          f"{max(per_row.values())}; train file {task_dir / 'train_task_5_fix.json'}")
    return manifest


def stage_train(plan: dict, variant: str) -> dict:
    """pilot, e1 (after a pilot go), e2 (every row; only after an e1 stop) or al_<selection> (an active-learning
    variant, trained as the pilot): one improve run from the round-1 model."""
    if variant not in ('pilot', 'e1', 'e2') and not variant.startswith('al_'):
        raise ValueError(f'variant must be pilot, e1, e2 or al_<selection>, got {variant}')
    if variant.startswith('al_') and _prebuilt(plan, variant) is None:
        raise FileNotFoundError(f'{variant}: no prebuilt variant (python -m tools.al_build_variant)')
    done = _done(plan, variant)
    if done:
        return done
    baseline = _require(plan, 'baseline')
    if variant == 'e1':
        _require(plan, 'pilot')
    if variant == 'e2':
        e1 = _done(plan, 'e1')
        if not e1 or e1['gate']['decision'] != 'stop':
            raise RuntimeError('e2 runs only after the e1 gate said stop (and the user agreed)')
    if variant.startswith('al_'):
        prev, reference = _al_start(plan, variant)
        return _run_variant(plan, variant, baseline['rows']['pilot'], baseline['wrong_images'], prev, reference)
    rows = list(_labels(plan)) if variant == 'e2' else baseline['rows'][variant]
    return _run_variant(plan, variant, rows, baseline['wrong_images'], plan['seed_task5'], 'baseline')


def stage_round2(plan: dict) -> dict:
    """A second feedback round: the Val-A images the e1 model still counts wrong, 2 epochs from the e1 model."""
    done = _done(plan, 'round2')
    if done:
        return done
    e1 = _require(plan, 'e1')
    if _prebuilt(plan, 'round2') is None and \
            not (Path(plan['synth_v2_root']) / 'tasks/synth_v2/train_task_1_gt_full.json').is_file():
        raise RuntimeError(f"round2 builds its data from the synthetic v2 release, not found at {plan['synth_v2_root']}: "
                           'extract full_syn_rpc_synth_v2_v1_<date>.tar and set synth_v2_root in the settings')
    e1_run = Path(plan['runs']) / f'{EXP}-e1'
    counted = _counted(plan, Path(plan['root']) / 'scored/e1', e1_run)
    wrong = ff.wrong_images(load_json(_inputs(plan) / 'val_a.json'), counted, _labels(plan))
    return _run_variant(plan, 'round2', e1['rows'], wrong, str(e1_run / f'task_{STAGE}/task_final.pth'), 'e1')


def stage_final_test(plan: dict, confirm: bool = False, variant: str | None = None) -> dict:
    """Test2019, scored once with the Val-B policies of the last improve run with a go, only with ``confirm=True``.
    With ``variant`` (an ``al_*`` run, whatever its gate), that run instead, in its own ``final_test_<variant>``."""
    from tools import eval_cl, eval_count

    stage = 'final_test' if variant is None else f'final_test_{variant}'
    done = _done(plan, stage)
    if done:
        print(f'test already scored on {done["finished_at"]}; not recomputed')
        return done
    if variant is not None:
        if not variant.startswith('al_') or _done(plan, variant) is None:
            raise RuntimeError(f'--variant of final_test must be a trained al_* run, got {variant}')
        chosen = variant
    else:
        chosen = next((v for v in ('round2', 'e1', 'e2') if (_done(plan, v) or {}).get('gate', {}).get('decision')
                       == 'go'), None)
    if chosen is None and plan['dry_run']:  # gates are not enforced in a dry run: still exercise the test stage
        chosen = next((v for v in ('round2', 'e1', 'e2') if _done(plan, v)), None)
    if chosen is None:
        raise RuntimeError('no improve run passed its gate: nothing to test')
    if not confirm:
        raise RuntimeError('test is scored once: call stage_final_test(plan, confirm=True) to score it')
    root, report = Path(plan['root']), load_json(Path(plan['root']) / 'stages' / f'{chosen}.json')
    e1_final = Path(plan['runs']) / f'{EXP}-e1/task_{STAGE}/task_final.pth'
    prev = str(e1_final) if chosen == 'round2' else plan['seed_task5']
    if chosen.startswith('al_'):
        prev = _al_start(plan, chosen)[0]
    with _attempt(plan, stage) as (record, attempt):
        variant = _prebuilt(plan, chosen)
        images = variant.parent.parent / 'images' if variant is not None else Path(plan['release_root']) / chosen
        _train(plan, record, attempt, chosen, _inputs(plan) / chosen, images, prev, VARIANTS[_kind(chosen)],
               splits='test')
    run, target = Path(plan['runs']) / f'{EXP}-{chosen}', root / 'scored' / f'{chosen}_test'
    out = target / f'task_{STAGE}'
    out.mkdir(parents=True, exist_ok=True)
    for name in ('pred_test.npz', 'task_final.pth', 'run_info.json'):
        if not (out / name).is_symlink():
            (out / name).symlink_to((run / f'task_{STAGE}' / name).resolve())
    policies = ('calibration_count_groups.json', 'calibration_count_m2.json')
    for policy in policies:
        save_json(target / policy, load_json(root / 'scored' / chosen / policy), indent=1)
    cfg, test = _cfg(plan), Path(plan['eval_root']) / 'test_full.json'
    detection = eval_cl.evaluate_run(target, 'test', test, cfg)
    save_json(target / 'metrics_cl_test.json', detection, indent=1)
    counting = {}
    for policy in policies:
        result = eval_count.evaluate_policy(target, test, cfg, load_json(target / policy))
        counting[policy] = {s: r['test']['overall'] for s, r in result['stages'].items()}
    result = {'stage': 'final_test', 'model': chosen, 'val_b_cAcc_m2': report['cAcc_m2'],
              'test2019': {'AP50': {s: v['m1']['mAP_A']['AP50'] for s, v in detection['stages'].items()},
                           'counting': counting}}
    return _finish(plan, stage, result, None)


STAGES = {'split': stage_split, 'baseline': stage_baseline, 'pilot': lambda p: stage_train(p, 'pilot'),
          'e1': lambda p: stage_train(p, 'e1'), 'e2': lambda p: stage_train(p, 'e2'), 'round2': stage_round2}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('improve', 'new'), required=True)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--stage', choices=(*STAGES, 'data', 'al', 'final_test'), required=True,
                        help='runs the stages before it that are not done yet, then this one')
    parser.add_argument('--confirm-test', action='store_true', help='with --stage final_test: score test2019 once')
    parser.add_argument('--variant', default='pilot',
                        help='with --stage data: the run whose training data to build (split and baseline run first); '
                             'with --stage al: the al_<selection> variant to train; with --stage final_test: an al_* '
                             'run to score on test2019 (once)')
    args = parser.parse_args(argv)
    if args.mode == 'new':
        raise NotImplementedError('mode new is Part 2 of the plan (Tasks 8-10)')
    from autocheckout.training_workflow import resolve

    plan = prepare(load_json(resolve(args.settings)))
    order = ['split', 'baseline', 'pilot', 'e1', 'round2']
    if args.stage == 'final_test':
        return stage_final_test(plan, confirm=args.confirm_test,
                                variant=args.variant if args.variant.startswith('al_') else None)
    if args.stage == 'data':
        stage_split(plan)
        stage_baseline(plan)
        return stage_data(plan, args.variant)
    if args.stage == 'e2':
        return stage_train(plan, 'e2')
    if args.stage == 'al':
        if not args.variant.startswith('al_'):
            parser.error('--stage al needs --variant al_<selection>')
        stage_split(plan)
        stage_baseline(plan)
        return stage_train(plan, args.variant)
    result = {}
    for stage in order[:order.index(args.stage) + 1]:
        result = STAGES[stage](plan)
    return result


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
