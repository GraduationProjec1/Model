"""Expand a trained checkpoint to the class slots of a larger task config
(docs/project/task6-relearn-plan-2026-10-30.md, Task 4, decision Q4).

The first ``--learned`` classes keep their classifier rows, private prompts and prototype memory, so
the expanded model predicts them exactly as before; the last classifier row is kept too. New
classifier rows start like a fresh focal-loss head (weight ~ N(0, init_std), bias at the 0.01 prior),
with the same values in every head. New prompt rows start at zero, as the slots of a task not yet
trained: init_task_prompts() initialises them when their task starts.

    python pdp/expand.py --checkpoint task_5/task_final.pth \\
        --old-config data/processed/rpc_eval_official_v1/task_config.json \\
        --new-config configs/relearn/task_config_relearn.json --learned 200 --out expanded.pth

With ``--repartition`` the slots stay the same and only the task boundaries after the learned classes change
(docs/design/2026-11-09-chia-12-13-sku, section 3.2): every tensor is kept as it is and only
``runtime.model_config.task_num_classes`` is rewritten, so tasks after ``--learned`` start exactly where they would
have started under the old config.

    python pdp/expand.py --repartition --checkpoint task_1/task_final.pth \\
        --old-config data/processed/rpc_eval_official_v1/task_config.json \\
        --new-config configs/tasks_100-8x12-13_seed0.json --learned 100 --out task_1_12x.pth
"""

import argparse
import math
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autocheckout.taskcfg import TaskConfig

PRIOR_PROB = 0.01
PRIVATE_PROMPT = re.compile(r'^model\.prompts\.private_[pka]_\d+$')


def _check_rows(key, value, expected, config):
    if value.shape[0] != expected:
        raise ValueError(f'{key} has {value.shape[0]} rows, {config.name} needs {expected}')


def expand_checkpoint(checkpoint, old_cfg, new_cfg, learned, seed=0):
    """A task_final.pth of old_cfg as a checkpoint of new_cfg; classes below `learned` unchanged."""
    def below(config):
        return [c for t in config.tasks for c in t.classes if c.label < learned]

    if len(below(old_cfg)) != learned or below(old_cfg) != below(new_cfg):
        raise ValueError(f'{new_cfg.name} differs from {old_cfg.name} at labels below {learned}')
    old_classes, new_classes = old_cfg.num_classes, new_cfg.num_classes
    runtime = deepcopy(checkpoint['runtime'])
    config = runtime['model_config']
    generator = torch.Generator().manual_seed(seed)
    n_new = new_classes - 1 - learned
    new_rows = {2: torch.randn(n_new, config['d_model'], generator=generator) * config['init_std'],
                1: torch.full((n_new,), -math.log((1 - PRIOR_PROB) / PRIOR_PROB))}

    model = {}
    for key, value in checkpoint['model'].items():
        if 'class_embed' in key:
            _check_rows(key, value, old_classes, old_cfg)
            model[key] = torch.cat([value[:learned], new_rows[value.ndim].to(value.dtype), value[-1:]])
        elif PRIVATE_PROMPT.match(key):
            _check_rows(key, value, old_cfg.num_slots, old_cfg)
            if value[learned:].any():
                raise ValueError(f'rows {learned}.. of {key} are not zero: a later task was trained')
            model[key] = torch.cat([value[:learned], value.new_zeros((new_cfg.num_slots - learned, *value.shape[1:]))])
        elif value.ndim and value.shape[0] in (old_classes, old_cfg.num_slots):
            raise ValueError(f'{key} has {value.shape[0]} rows, a class count of {old_cfg.name}; '
                             'expand_checkpoint does not know how to expand it')
        else:
            model[key] = value

    memory = {name: {i: checkpoint[name][i] for i in range(learned)}
              for name in ('class_query_cache', 'class_prototypes', 'class_cache_count')}
    for i in range(learned, new_cfg.num_slots):
        memory['class_query_cache'][i] = []
        memory['class_prototypes'][i] = torch.zeros(config['d_model'])
        memory['class_cache_count'][i] = 0

    config['id2label'] = {i: f'LABEL_{i}' for i in range(new_classes)}
    config['label2id'] = {f'LABEL_{i}': i for i in range(new_classes)}
    config['task_num_classes'] = [t.num_classes for t in new_cfg.tasks]
    return {'runtime': runtime, 'model': model, 'task_id': checkpoint['task_id'], **memory,
            'expanded_from': {'old_classes': old_classes, 'new_classes': new_classes, 'learned': learned,
                              'seed': seed}}


def repartition_checkpoint(checkpoint, old_cfg, new_cfg, learned):
    """A task_final.pth of old_cfg as a checkpoint of new_cfg, which has the same slots and the same classes below
    `learned` but other task boundaries above it. Only task_num_classes changes; the input is not modified."""
    def below(config):
        return [c for t in config.tasks for c in t.classes if c.label < learned]

    if len(below(old_cfg)) != learned or below(old_cfg) != below(new_cfg):
        raise ValueError(f'{new_cfg.name} differs from {old_cfg.name} at labels below {learned}')
    if (old_cfg.num_slots, old_cfg.num_classes) != (new_cfg.num_slots, new_cfg.num_classes):
        raise ValueError(f'{new_cfg.name} has {new_cfg.num_slots} slots, {old_cfg.name} {old_cfg.num_slots}: '
                         'use expand_checkpoint to change the number of slots')
    if not any(t.offset == learned for t in new_cfg.tasks) or not any(t.offset == learned for t in old_cfg.tasks):
        raise ValueError(f'label {learned} does not start a task in both configs')
    config = checkpoint['runtime']['model_config']
    if config['task_num_classes'] != [t.num_classes for t in old_cfg.tasks]:
        raise ValueError(f"checkpoint task_num_classes {config['task_num_classes']} is not {old_cfg.name}")
    for key, value in checkpoint['model'].items():
        if PRIVATE_PROMPT.match(key):
            _check_rows(key, value, old_cfg.num_slots, old_cfg)
            if value[learned:].any():
                raise ValueError(f'rows {learned}.. of {key} are not zero: a later task was trained')
    for i in range(learned, new_cfg.num_slots):
        if checkpoint['class_query_cache'].get(i) or checkpoint['class_cache_count'].get(i, 0) or \
                checkpoint['class_prototypes'][i].any():
            raise ValueError(f'prototype memory of class {i} is not empty: a later task was trained')
    runtime = deepcopy(checkpoint['runtime'])
    runtime['model_config']['task_num_classes'] = [t.num_classes for t in new_cfg.tasks]
    return {**checkpoint, 'runtime': runtime,
            'repartitioned_from': {'old_config': old_cfg.name, 'new_config': new_cfg.name, 'learned': learned,
                                   'old_task_num_classes': config['task_num_classes']}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='task_final.pth of the last learned task')
    parser.add_argument('--old-config', required=True, help='task config the checkpoint was trained with')
    parser.add_argument('--new-config', required=True, help='larger task config, equal to the old one below --learned')
    parser.add_argument('--learned', type=int, required=True, help='classes learned so far (labels 0 .. N-1)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--repartition', action='store_true',
                        help='same slots, other task boundaries above --learned: only task_num_classes changes')
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    old_cfg, new_cfg = TaskConfig.load(args.old_config), TaskConfig.load(args.new_config)
    if args.repartition:
        result = repartition_checkpoint(checkpoint, old_cfg, new_cfg, args.learned)
        message = (f"task_num_classes {result['repartitioned_from']['old_task_num_classes']} -> "
                   f"{result['runtime']['model_config']['task_num_classes']}")
    else:
        result = expand_checkpoint(checkpoint, old_cfg, new_cfg, args.learned)
        message = f'{old_cfg.num_classes} -> {new_cfg.num_classes} classes'
    tmp_path = args.out + '.tmp'
    torch.save(result, tmp_path)
    os.replace(tmp_path, args.out)
    print(f'{message}, {args.learned} learned; wrote {args.out}')


if __name__ == '__main__':
    main()
