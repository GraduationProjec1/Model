"""225-SKU config: RPC and VN products learned together in 6 tasks (AN/cyclegan/02_KIEM_TRA_CHIA_TASK_VA_VAL_TEST.md, §11).

Starting from the RPC config ``--base`` (task 1 = 100 SKUs, tasks 2..5 = 25 SKUs each):

- ``--move`` (default ``10,3,3,3,3``) RPC SKUs leave each base task for a new last task, so the RPC counts become
  90/22/22/22/22/22. Every other RPC SKU stays in its base task.
- The supercategory mix of the new task follows RPC as a whole: its share of each supercategory is the largest
  remainder of the 200 SKUs over the new task's size. Tasks 2..5 give theirs first (the supercategories with the
  largest unmet share, ties broken with ``--seed``); task 1 gives the rest.
- Inside each (base task, supercategory) cell, the SKUs that move start as a seeded random draw. With
  ``--confusions`` (``[[gt rpc id, predicted rpc id, count], ...]``, measured on Val-A only) a hill climb then swaps a
  moving SKU with a staying SKU of the same cell while that lowers the confusion weight between different tasks.
  Swaps inside a cell keep every count and every supercategory share unchanged.
- ``--vn-split`` puts VN products into the tasks: one ``;``-separated group per task, numbers ``01``..``25``. A VN
  product keeps the class name and id of ``--vn-config`` (VNxx -> rpc_category_id 200 + xx) and gets the supercategory
  ``vn``.
- Inside a task classes are ordered by id (RPC first, VN after); labels are contiguous. ``--reserved`` slots form a
  last, reserved task (default 0).

    python -m tools.make_merged_task_config --base configs/tasks_100-4x25_seed0.json \\
        --vn-config configs/tasks_100-4x25_seed0+vn25.json \\
        --vn-split "01,05,06,12,14,17,18,20,24,25;08,21,22;04,10,23;02,15,16;09,11,13;03,07,19" \\
        --confusions configs/data/confusions/r4_vala_swaps_v1.json --reserved 25 --name 225_vn10-3-res25_seed0
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from autocheckout.io import load_json
from autocheckout.sampling import largest_remainder
from autocheckout.taskcfg import ClassInfo, Task, TaskConfig

VN_ID_OFFSET = 200      # VNxx has rpc_category_id 200 + xx (labels 200..224 + 1 in the task-6 packaging)
VN_SUPERCATEGORY = 'vn'


def parse_vn_split(text: str) -> list[list[int]]:
    groups = [[int(x) for x in part.split(',') if x.strip()] for part in text.split(';')]
    flat = [v for g in groups for v in g]
    if len(set(flat)) != len(flat):
        raise ValueError(f'a VN product is listed twice in --vn-split: {sorted(v for v, n in Counter(flat).items() if n > 1)}')
    return groups


def task6_quota(base: TaskConfig, size: int) -> dict[str, int]:
    total = Counter(c.supercategory for t in base.data_tasks for c in t.classes)
    return largest_remainder(dict(sorted(total.items())), size)


def cell_counts(base: TaskConfig, move: Sequence[int], quota: dict[str, int], rng: random.Random) -> dict[tuple[int, str], int]:
    """How many SKUs of each supercategory leave each base task (rows sum to ``move``, columns to ``quota``)."""
    avail = {t.task_id: Counter(c.supercategory for c in t.classes) for t in base.data_tasks}
    need = dict(quota)
    cells: dict[tuple[int, str], int] = defaultdict(int)
    order = [t.task_id for t in base.data_tasks if t.task_id != 1] + [1]
    for task_id in order:
        for _ in range(move[task_id - 1]):
            options = [s for s in sorted(need) if need[s] > 0 and avail[task_id][s] - cells[task_id, s] > 0]
            if not options:
                raise ValueError(f'task {task_id}: no supercategory left to move (quota {quota}, cells {dict(cells)})')
            top = max(need[s] for s in options)
            pick = rng.choice([s for s in options if need[s] == top])
            cells[task_id, pick] += 1
            need[pick] -= 1
    if any(need.values()):
        raise ValueError(f'supercategory shares not met: {need}')
    return dict(cells)


def cross_weight(task_of: dict[int, int], weights: dict[tuple[int, int], int]) -> int:
    return sum(w for (a, b), w in weights.items() if a in task_of and b in task_of and task_of[a] != task_of[b])


def choose_movers(base: TaskConfig, cells: dict[tuple[int, str], int], weights: dict[tuple[int, int], int],
                  new_task: int, rng: random.Random) -> set[int]:
    by_cell = defaultdict(list)
    for t in base.data_tasks:
        for c in sorted(t.classes, key=lambda c: c.rpc_category_id):
            by_cell[t.task_id, c.supercategory].append(c.rpc_category_id)
    movers: set[int] = set()
    for key, n in sorted(cells.items()):
        movers.update(rng.sample(by_cell[key], n))
    home = {c.rpc_category_id: t.task_id for t in base.data_tasks for c in t.classes}
    if not weights:
        return movers
    task_of = {r: (new_task if r in movers else home[r]) for r in home}
    best = cross_weight(task_of, weights)
    improved = True
    while improved:
        improved = False
        for key in sorted(cells):
            for m in sorted(r for r in by_cell[key] if r in movers):
                for s in sorted(r for r in by_cell[key] if r not in movers):
                    task_of[m], task_of[s] = home[m], new_task
                    w = cross_weight(task_of, weights)
                    if w < best:
                        best, improved = w, True
                        movers.discard(m)
                        movers.add(s)
                        break
                    task_of[m], task_of[s] = new_task, home[s]
                if improved:
                    break
            if improved:
                break
    return movers


def merged_task_config(base: TaskConfig, vn_classes: dict[int, ClassInfo], vn_split: list[list[int]], *,
                       move: Sequence[int], seed: int, name: str, reserved: int = 0,
                       confusions: Sequence[Sequence[int]] = ()) -> TaskConfig:
    data = base.data_tasks
    if len(move) != len(data):
        raise ValueError(f'--move needs one number per base task ({len(data)}), got {list(move)}')
    if len(vn_split) != len(data) + 1:
        raise ValueError(f'--vn-split needs {len(data) + 1} groups (base tasks + the new one), got {len(vn_split)}')
    missing = sorted(set(vn_classes) - {v for g in vn_split for v in g})
    unknown = sorted({v for g in vn_split for v in g} - set(vn_classes))
    if missing or unknown:
        raise ValueError(f'--vn-split must list every VN product once: missing {missing}, unknown {unknown}')
    new_task = len(data) + 1
    rng = random.Random(seed)
    quota = task6_quota(base, sum(move))
    cells = cell_counts(base, move, quota, rng)
    weights: dict[tuple[int, int], int] = defaultdict(int)
    for a, b, k in confusions:
        if a != b:
            weights[min(a, b), max(a, b)] += int(k)
    movers = choose_movers(base, cells, dict(weights), new_task, rng)

    members: list[list[ClassInfo]] = [[c for c in t.classes if c.rpc_category_id not in movers] for t in data]
    members.append([c for t in data for c in t.classes if c.rpc_category_id in movers])
    for task_index, group in enumerate(vn_split):
        members[task_index] += [vn_classes[v] for v in group]

    tasks, offset = [], 0
    for task_id, infos in enumerate(members, start=1):
        infos = sorted(infos, key=lambda c: c.rpc_category_id)
        classes = tuple(ClassInfo(label=offset + i, name=c.name, rpc_category_id=c.rpc_category_id,
                                  supercategory=c.supercategory) for i, c in enumerate(infos))
        tasks.append(Task(task_id=task_id, offset=offset, classes=classes))
        offset += len(classes)
    if reserved:
        tasks.append(Task(task_id=len(tasks) + 1, offset=offset, reserved=True,
                          classes=tuple(ClassInfo(label=offset + i, name=f'reserved_{offset + i}') for i in range(reserved))))
    return TaskConfig(name=name, seed=seed, tasks=tuple(tasks))


def load_vn_classes(vn_config: TaskConfig) -> dict[int, ClassInfo]:
    out = {}
    for t in vn_config.data_tasks:
        for c in t.classes:
            if c.rpc_category_id is not None and c.rpc_category_id > VN_ID_OFFSET:
                number = c.rpc_category_id - VN_ID_OFFSET
                if not c.name.startswith(f'VN{number:02d}'):
                    raise ValueError(f'id {c.rpc_category_id} is {c.name!r}, expected VN{number:02d}')
                out[number] = ClassInfo(label=-1, name=c.name, rpc_category_id=c.rpc_category_id,
                                        supercategory=VN_SUPERCATEGORY)
    return out


def summary(config: TaskConfig, base: TaskConfig, confusions) -> str:
    groups = sorted({c.supercategory for t in config.data_tasks for c in t.classes})
    lines = [f"{'supercategory':18s}" + ''.join(f"{'T' + str(t.task_id):>5s}" for t in config.data_tasks) + f"{'total':>7s}"]
    for g in groups + ['total']:
        counts = [sum(g in ('total', c.supercategory) for c in t.classes) for t in config.data_tasks]
        lines.append(f'{g:18s}' + ''.join(f'{n:5d}' for n in counts) + f'{sum(counts):7d}')
    weights = defaultdict(int)
    for a, b, k in confusions:
        if a != b:
            weights[min(a, b), max(a, b)] += int(k)
    if weights:
        old = {c.rpc_category_id: t.task_id for t in base.data_tasks for c in t.classes}
        new = {c.rpc_category_id: t.task_id for t in config.data_tasks for c in t.classes}
        total = sum(weights.values())
        lines.append(f'RPC confusion weight across tasks: base {cross_weight(old, weights)}/{total}, '
                     f'new {cross_weight(new, weights)}/{total}')
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> TaskConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', type=Path, required=True, help='RPC task config (data tasks 1..5)')
    parser.add_argument('--vn-config', type=Path, required=True, help='config holding the 25 VN classes (ids 201..225)')
    parser.add_argument('--vn-split', required=True, help='VN numbers per task, e.g. "01,05;08,21;..."')
    parser.add_argument('--move', default='10,3,3,3,3', help='RPC SKUs leaving each base task for the new task')
    parser.add_argument('--confusions', type=Path, help='Val-A confusions [[gt rpc id, pred rpc id, count], ...]')
    parser.add_argument('--reserved', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', required=True)
    parser.add_argument('--out', type=Path, help='default: configs/tasks_<name>.json')
    args = parser.parse_args(argv)
    base = TaskConfig.load(args.base)
    confusions = load_json(args.confusions) if args.confusions else []
    config = merged_task_config(base, load_vn_classes(TaskConfig.load(args.vn_config)), parse_vn_split(args.vn_split),
                                move=[int(x) for x in args.move.split(',')], seed=args.seed, name=args.name,
                                reserved=args.reserved, confusions=confusions)
    out = args.out or Path('configs') / f'tasks_{args.name}.json'
    config.save(out)
    print(summary(config, base, confusions))
    print(f'{config.num_slots} slots, {config.num_classes} classifier outputs; wrote {out}')
    return config


if __name__ == '__main__':
    main()
