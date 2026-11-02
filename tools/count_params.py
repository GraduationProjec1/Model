"""B4: parameters of a PDP checkpoint by part, read without torch (tensor shapes only), and what one update trains.
FULL-SYN r1 freezes backbone, encoder and decoder (``--freeze backbone,encoder,decoder``) and the private prompts of
the earlier tasks; of the rest, the new parameters train at ``lr`` and the shared ones (``query_tf``, ``input_proj``,
shared prompts, ``bbox_embed``, ...) at ``lr_old`` (10x lower):

- a new task: at ``lr`` its private prompts (``model.prompts.private_*_<t-1>``) + its rows of ``class_embed``;
- improve mode (active learning): at ``lr`` the ``train_rows`` rows of ``class_embed`` + the task-5 private prompts.
``class_embed.<i>`` are the per-decoder-layer copies of the classifier (one row per label: weight + bias).

    python -m tools.count_params ../relearn_fix_v2/seed/full-syn-r1/runs/EXP-FULL-SYN/task_5/task_final.pth \\
        --task 5 --task-classes 25 --train-rows 17
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import zipfile
from collections import OrderedDict, defaultdict


class _Tensor:
    """Stands for torch._utils._rebuild_tensor_v2(storage, offset, size, stride, ...): keeps the size."""

    def __init__(self, *args):
        self.shape = tuple(args[2]) if len(args) > 2 else ()


class _Any:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Any()

    def __setstate__(self, state):
        pass


class _Reader(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'OrderedDict':
            return OrderedDict
        if module.startswith('torch') and name.startswith('_rebuild_tensor'):
            return _Tensor
        if module.startswith('torch') and name == '_rebuild_parameter':
            return lambda tensor, *a: tensor
        try:
            return super().find_class(module, name)
        except Exception:
            return _Any

    def persistent_load(self, pid):
        return None


def shapes(path: str) -> dict[str, tuple]:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith('data.pkl'))
        state = _Reader(z.open(name)).load()
    model = state.get('model', state)
    return {k: v.shape for k, v in model.items() if isinstance(v, _Tensor)}


def summary(path: str, task: int, task_classes: int, train_rows: int) -> dict:
    sizes = {k: math.prod(v) for k, v in shapes(path).items()}
    parts = defaultdict(int)
    for key, n in sizes.items():
        part = ('backbone' if '.backbone.' in key else 'encoder' if '.encoder.' in key else
                'decoder' if '.decoder.' in key else 'prompts.private' if 'prompts.private' in key else
                'prompts.shared' if 'prompts.shared' in key else 'prompts.query_tf' if 'prompts.' in key else
                'class_embed' if key.startswith('class_embed') else 'bbox_embed' if key.startswith('bbox_embed') else
                'input_proj, queries, ...')
        parts[part] += n
    total = sum(sizes.values())
    weight = [s for k, s in shapes(path).items() if re.fullmatch(r'class_embed\.\d+\.weight', k)]
    layers, (labels, width) = len(weight), weight[0]
    row = layers * (width + 1)
    private = defaultdict(int)
    for key, n in sizes.items():
        m = re.search(r'prompts\.private_[a-z]+_(\d+)$', key)
        if m:
            private[int(m.group(1))] += n
    own = private.get(task - 1, 0)
    frozen = parts['backbone'] + parts['encoder'] + parts['decoder'] + parts['prompts.private'] - own
    new_task, improve = own + task_classes * row, own + train_rows * row
    return {'checkpoint': path, 'total': total, 'by_part': dict(sorted(parts.items(), key=lambda x: -x[1])),
            'labels': labels, 'class_embed_layers': layers, 'params_per_label_row': row,
            'private_prompts_per_task': dict(sorted(private.items())),
            'frozen': frozen, 'frozen_share': frozen / total,
            'trained_at_lr_old': total - frozen - new_task,
            'trained_at_lr_old_share': (total - frozen - new_task) / total,
            'new_task_at_lr': new_task, 'new_task_at_lr_share': new_task / total,
            'improve_at_lr': improve, 'improve_at_lr_share': improve / total}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('checkpoint')
    ap.add_argument('--task', type=int, default=5, help='task whose private prompts are new')
    ap.add_argument('--task-classes', type=int, default=25)
    ap.add_argument('--train-rows', type=int, default=17)
    args = ap.parse_args(argv)
    return summary(args.checkpoint, args.task, args.task_classes, args.train_rows)


if __name__ == '__main__':
    print(json.dumps(main(), indent=1))
