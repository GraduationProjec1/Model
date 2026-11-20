"""Mode new (docs/project/feedback-finetune-plan-2026-11-01.md, Phần 2, N1): a task of products the model does not know.

- The data tasks of ``--base`` are kept as they are; its reserved task is dropped.
- ``--products`` (JSON list of ``{"pdp_class_id": label, "name": ...}``, as the Checkout-system packaging writes it)
  becomes one new data task, labels continuing after the last data task. Each product has its own id in the
  ``rpc_category_id`` field (``label + --id-offset``, default 1 like RPC: labels 200..224 -> ids 201..225, the
  category_id of the synthetic images); an id already used by an earlier class is refused (a product the model knows
  is relearned with ``--mode improve``, not here).
- ``--reserve-after`` slots form a last, reserved task (room for the next products without growing the model again).

The checkpoint of the base gets the slots of the new config with ``pdp/expand.py --learned <seen classes of the
base>``; the classes below that stay identical (checked here).

    python -m tools.make_new_task_config --base data/eval/task_config.json \\
        --products data/new_products/vn25/products.json --reserve-after 25 \\
        --name 100-4x25_seed0+vn25 --out configs/tasks_100-4x25_seed0+vn25.json
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from autocheckout.io import load_json
from autocheckout.taskcfg import ClassInfo, Task, TaskConfig


def new_task_config(base: TaskConfig, products: Sequence[dict[str, Any]], *, reserve_after: int, name: str,
                    id_offset: int = 1) -> TaskConfig:
    """``base`` with one data task of ``products`` after its data tasks, then ``reserve_after`` reserved slots."""
    if not products:
        raise ValueError('no products')
    if reserve_after < 0:
        raise ValueError('reserve_after must be >= 0')
    data = list(base.data_tasks)
    offset = base.seen_classes(data[-1].task_id)
    used = {c.rpc_category_id for t in data for c in t.classes if c.rpc_category_id is not None}
    classes = []
    for position, product in enumerate(products):
        label = offset + position
        if 'pdp_class_id' in product and int(product['pdp_class_id']) != label:
            raise ValueError(f"product {product['name']!r} has pdp_class_id {product['pdp_class_id']}, but its label "
                             f'in the new task is {label}: products must be listed by label, from {offset} on')
        rpc_id = int(product.get('rpc_category_id', label + id_offset))
        if rpc_id in used:
            raise ValueError(f"product {product['name']!r}: id {rpc_id} is already a class of {base.name}; "
                             'a known product is relearned with --mode improve')
        used.add(rpc_id)
        classes.append(ClassInfo(label=label, name=str(product['name']), rpc_category_id=rpc_id,
                                 supercategory=product.get('supercategory')))
    tasks = [*data, Task(task_id=len(data) + 1, offset=offset, classes=tuple(classes))]
    if reserve_after:
        start = offset + len(classes)
        tasks.append(Task(task_id=len(tasks) + 1, offset=start, reserved=True,
                          classes=tuple(ClassInfo(label=start + i, name=f'reserved_{start + i}')
                                        for i in range(reserve_after))))
    return TaskConfig(name=name, seed=base.seed, tasks=tuple(tasks))


def main(argv: list[str] | None = None) -> TaskConfig:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--base', type=Path, required=True, help='task config of the model to extend')
    parser.add_argument('--products', type=Path, required=True, help='JSON list of {pdp_class_id, name}')
    parser.add_argument('--reserve-after', type=int, default=25, help='reserved slots after the new task')
    parser.add_argument('--id-offset', type=int, default=1, help='id of a product = its label + this')
    parser.add_argument('--name', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    base = TaskConfig.load(args.base)
    cfg = new_task_config(base, load_json(args.products), reserve_after=args.reserve_after, name=args.name,
                          id_offset=args.id_offset)
    cfg.save(args.out)
    new = cfg.task(len(base.data_tasks) + 1)
    print(f'{cfg.name}: task {new.task_id} = labels {new.offset}..{new.offset + new.num_classes - 1}, '
          f'{cfg.num_slots} slots (base {base.num_slots}), n_classes {cfg.num_classes}; wrote {args.out}')
    return cfg


if __name__ == '__main__':
    main()
