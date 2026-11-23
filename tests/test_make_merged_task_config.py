from collections import Counter

import pytest

from autocheckout.taskcfg import ClassInfo, TaskConfig
from tools.make_merged_task_config import load_vn_classes, merged_task_config, parse_vn_split
from tools.make_task_config import build_task_config

GROUP_SIZES = [22, 18, 16, 15, 14, 13, 12, 12, 11, 11, 10, 10, 9, 8, 7, 7, 5]   # like RPC, as in test_make_task_config


def rpc_like_categories():
    out = []
    for index, count in enumerate(GROUP_SIZES):
        for _ in range(count):
            cid = len(out) + 1
            out.append({'id': cid, 'name': f'{cid}_g{index:02d}', 'supercategory': f'g{index:02d}'})
    return out

SPLIT = '01,05,06,12,14,17,18,20,24,25;08,21,22;04,10,23;02,15,16;09,11,13;03,07,19'


def base():
    return build_task_config(rpc_like_categories(), [100, 25, 25, 25, 25], reserved=0, seed=0, name='b')


def vn():
    return {n: ClassInfo(label=-1, name=f'VN{n:02d} x', rpc_category_id=200 + n, supercategory='vn') for n in range(1, 26)}


def test_counts_labels_and_supercategory_mix():
    b = base()
    cfg = merged_task_config(b, vn(), parse_vn_split(SPLIT), move=[10, 3, 3, 3, 3], seed=0, name='m')
    assert [t.num_classes for t in cfg.tasks] == [100, 25, 25, 25, 25, 25]
    rpc = [[c for c in t.classes if c.supercategory != 'vn'] for t in cfg.tasks]
    assert [len(r) for r in rpc] == [90, 22, 22, 22, 22, 22]
    assert sorted(cfg.rpc_to_label()) == list(range(1, 201)) + list(range(201, 226))
    # RPC SKUs only stay in their base task or go to task 6
    home = {c.rpc_category_id: t.task_id for t in b.data_tasks for c in t.classes}
    for t in cfg.tasks:
        for c in t.classes:
            if c.supercategory != 'vn':
                assert t.task_id in (home[c.rpc_category_id], 6)
    # task 6 supercategory mix is proportional to all 200 SKUs (floor or ceil of 22/200 of each group)
    total = Counter(c.supercategory for t in b.data_tasks for c in t.classes)
    t6 = Counter(c.supercategory for c in rpc[5])
    for g, n in total.items():
        assert int(22 * n / 200) <= t6[g] <= int(22 * n / 200) + 1
    assert [sorted(c.rpc_category_id - 200 for c in t.classes if c.supercategory == 'vn') for t in cfg.tasks] \
        == [sorted(int(x) for x in g.split(',')) for g in SPLIT.split(';')]
    # same seed, same config
    again = merged_task_config(b, vn(), parse_vn_split(SPLIT), move=[10, 3, 3, 3, 3], seed=0, name='m')
    assert again.to_dict() == cfg.to_dict()


def test_confusions_keep_confused_pairs_together():
    b = base()
    plain = merged_task_config(b, vn(), parse_vn_split(SPLIT), move=[10, 3, 3, 3, 3], seed=0, name='m')
    t6 = [c for c in plain.task(6).classes if c.supercategory != 'vn']
    mover = t6[0]
    home = {c.rpc_category_id: t.task_id for t in b.data_tasks for c in t.classes}
    # a staying SKU of the same base task and supercategory, strongly confused with the mover's old task-mate
    stay = next(c for c in b.task(home[mover.rpc_category_id]).classes
                if c.supercategory == mover.supercategory and c.rpc_category_id != mover.rpc_category_id
                and plain.task_of_label(plain.rpc_to_label()[c.rpc_category_id]) != 6)
    mate = next(c for c in b.task(home[mover.rpc_category_id]).classes
                if c.rpc_category_id not in (mover.rpc_category_id, stay.rpc_category_id)
                and plain.task_of_label(plain.rpc_to_label()[c.rpc_category_id]) != 6)
    conf = [[mover.rpc_category_id, mate.rpc_category_id, 50]]
    cfg = merged_task_config(b, vn(), parse_vn_split(SPLIT), move=[10, 3, 3, 3, 3], seed=0, name='m', confusions=conf)
    task = {r: cfg.task_of_label(lb) for r, lb in cfg.rpc_to_label().items()}
    assert task[mover.rpc_category_id] == task[mate.rpc_category_id]
    assert sum(1 for c in cfg.task(6).classes if c.supercategory != 'vn') == 22


def test_bad_vn_split_is_refused():
    with pytest.raises(ValueError, match='twice'):
        parse_vn_split('01,02;02')
    with pytest.raises(ValueError, match='missing'):
        merged_task_config(base(), vn(), parse_vn_split('01;02;03;04;05;06'), move=[10, 3, 3, 3, 3], seed=0, name='m')


def test_load_vn_classes_from_real_config():
    out = load_vn_classes(TaskConfig.load('configs/tasks_100-4x25_seed0+vn25.json'))
    assert sorted(out) == list(range(1, 26)) and out[14].rpc_category_id == 214 and out[14].name.startswith('VN14')
