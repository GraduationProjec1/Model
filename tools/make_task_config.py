"""DL4: split the RPC SKUs into class-incremental tasks (IMPLEMENTATION_PLAN.md, section 4.4).

- ``--sizes`` (default 100,25,25,25,25) must add up to the number of categories; ``--reserved``
  slots (default 24) form a last, reserved task (products outside RPC, no RPC id).
- Stratified by supercategory: task by task, the task size is shared among the supercategories
  in proportion to their SKUs not assigned yet (largest remainder, ties by supercategory name).
  Task totals are exact and each supercategory is spread over the tasks like the task sizes.
- Within a supercategory the SKUs are shuffled with ``--seed`` and dealt to the tasks in order;
  inside a task, classes are ordered by RPC category id. Labels follow docs/data_preprocessing/formats.md, section 3.

    python -m tools.make_task_config \\
        --categories /data/rpc/raw/retail_product_checkout/instances_test2019.json \\
        --name 100-4x25_seed0 --sizes 100,25,25,25,25 --reserved 24 --seed 0

``--split-from`` splits every task >= 2 of an existing config instead (docs/design/2026-11-09-chia-12-13-sku):
task 1 and the reserved task are kept as they are; inside each other task, ``--split`` sizes are shared among its
supercategories (largest remainder), each supercategory (by name) is shuffled with ``random.Random(seed)`` (a new
generator per task) and dealt in order; inside a new task, classes are ordered by RPC id; labels are contiguous.

    python -m tools.make_task_config --split-from data/processed/rpc_eval_official_v1/task_config.json \\
        --split 12,13 --name 100-8x12-13_seed0 --seed 0
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from autocheckout.io import load_json
from autocheckout.sampling import largest_remainder
from autocheckout.taskcfg import ClassInfo, Task, TaskConfig


def allocate(skus_per_group: dict[str, int], sizes: list[int]) -> dict[str, list[int]]:
    """SKUs of each supercategory in each task; rows sum to the SKU counts, columns to ``sizes``."""
    total = sum(skus_per_group.values())
    if sum(sizes) != total:
        raise ValueError(f"task sizes add up to {sum(sizes)}, but there are {total} SKUs")
    remaining = dict(skus_per_group)
    shares: dict[str, list[int]] = {group: [] for group in skus_per_group}
    for size in sizes:
        task_shares = largest_remainder(remaining, size)
        for group, n in task_shares.items():
            shares[group].append(n)
            remaining[group] -= n
    return shares


def build_task_config(categories: list[dict[str, Any]], sizes: list[int], reserved: int, seed: int,
                      name: str) -> TaskConfig:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for category in sorted(categories, key=lambda c: c["id"]):
        by_group[category["supercategory"]].append(category)
    shares = allocate({group: len(skus) for group, skus in by_group.items()}, sizes)

    rng = random.Random(seed)
    task_skus: list[list[dict[str, Any]]] = [[] for _ in sizes]
    for group in sorted(by_group):
        skus = list(by_group[group])
        rng.shuffle(skus)
        start = 0
        for task_index, n in enumerate(shares[group]):
            task_skus[task_index].extend(skus[start:start + n])
            start += n

    tasks, offset = [], 0
    for task_id, skus in enumerate(task_skus, start=1):
        skus.sort(key=lambda c: c["id"])
        classes = tuple(ClassInfo(label=offset + i, name=c["name"], rpc_category_id=c["id"],
                                  supercategory=c["supercategory"]) for i, c in enumerate(skus))
        tasks.append(Task(task_id=task_id, offset=offset, classes=classes))
        offset += len(classes)
    if reserved:
        classes = tuple(ClassInfo(label=offset + i, name=f"reserved_{offset + i}") for i in range(reserved))
        tasks.append(Task(task_id=len(tasks) + 1, offset=offset, classes=classes, reserved=True))
    return TaskConfig(name=name, seed=seed, tasks=tuple(tasks))


def split_task_config(config: TaskConfig, sizes: list[int], seed: int, name: str) -> TaskConfig:
    """Every data task >= 2 of ``config`` split into ``len(sizes)`` tasks of ``sizes`` classes (see module doc)."""
    groups_of_task = []
    for task in config.data_tasks:
        if task.task_id == 1:
            groups_of_task.append(list(task.classes))
            continue
        if any(c.alias_of is not None for c in task.classes):
            raise ValueError(f'task {task.task_id} has alias classes; only plain RPC tasks can be split')
        by_group: dict[str, list[ClassInfo]] = defaultdict(list)
        for info in sorted(task.classes, key=lambda c: c.rpc_category_id):
            by_group[info.supercategory].append(info)
        shares = allocate({group: len(infos) for group, infos in by_group.items()}, sizes)
        rng = random.Random(seed)
        parts: list[list[ClassInfo]] = [[] for _ in sizes]
        for group in sorted(by_group):
            infos = list(by_group[group])
            rng.shuffle(infos)
            start = 0
            for index, n in enumerate(shares[group]):
                parts[index].extend(infos[start:start + n])
                start += n
        groups_of_task += parts
    tasks, offset = [], 0
    for task_id, infos in enumerate(groups_of_task, start=1):
        infos = sorted(infos, key=lambda c: c.rpc_category_id)
        classes = tuple(ClassInfo(label=offset + i, name=c.name, rpc_category_id=c.rpc_category_id,
                                  supercategory=c.supercategory) for i, c in enumerate(infos))
        tasks.append(Task(task_id=task_id, offset=offset, classes=classes))
        offset += len(classes)
    for task in config.tasks:
        if task.reserved:
            classes = tuple(ClassInfo(label=offset + i, name=f'reserved_{offset + i}') for i in range(task.num_classes))
            tasks.append(Task(task_id=len(tasks) + 1, offset=offset, classes=classes, reserved=True))
            offset += task.num_classes
    return TaskConfig(name=name, seed=seed, tasks=tuple(tasks))


def distribution_table(config: TaskConfig) -> str:
    """Tasks x supercategory counts as a plain-text table."""
    data_tasks = config.data_tasks
    groups = sorted({c.supercategory for t in data_tasks for c in t.classes})
    lines = [f"{'supercategory':24s}" + "".join(f"{'T' + str(t.task_id):>6s}" for t in data_tasks)
             + f"{'total':>7s}"]
    for group in groups + ["total"]:
        counts = [sum(group in ("total", c.supercategory) for c in t.classes) for t in data_tasks]
        lines.append(f"{group:24s}" + "".join(f"{n:6d}" for n in counts) + f"{sum(counts):7d}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categories", type=Path, help="COCO JSON with the SKU categories")
    parser.add_argument("--split-from", type=Path, help="existing task config whose tasks >= 2 are split")
    parser.add_argument("--split", default="12,13", help="with --split-from: sizes each task >= 2 is split into")
    parser.add_argument("--name", required=True, help="config name, e.g. 100-4x25_seed0")
    parser.add_argument("--sizes", default="100,25,25,25,25", help="number of SKUs per data task")
    parser.add_argument("--reserved", type=int, default=24, help="reserved slots in a last task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="default: configs/tasks_<name>.json")
    args = parser.parse_args(argv)

    if (args.categories is None) == (args.split_from is None):
        parser.error("give exactly one of --categories and --split-from")
    if args.split_from:
        split = [int(size) for size in args.split.split(",")]
        config = split_task_config(TaskConfig.load(args.split_from), split, args.seed, args.name)
    else:
        categories = load_json(args.categories)["categories"]
        sizes = [int(size) for size in args.sizes.split(",")]
        config = build_task_config(categories, sizes, args.reserved, args.seed, args.name)
    out = args.out or Path("configs") / f"tasks_{args.name}.json"
    config.save(out)
    print(distribution_table(config))
    reserved = sum(t.num_classes for t in config.tasks if t.reserved)
    print(f"{config.num_slots} slots ({reserved} reserved), {config.num_classes} classifier outputs; "
          f"wrote {out}")


if __name__ == "__main__":
    main()
