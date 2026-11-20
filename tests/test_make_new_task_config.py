"""tools/make_new_task_config.py: mode new, Task 8 of docs/project/feedback-finetune-plan-2026-11-01.md."""

from pathlib import Path

import pytest

from autocheckout.taskcfg import TaskConfig
from tools.make_new_task_config import main, new_task_config

REPO = Path(__file__).resolve().parent.parent
BASE = TaskConfig.load(REPO / 'configs/tasks_100-4x25_seed0.json')  # 200 classes in 5 tasks + 24 reserved slots
PRODUCTS = [{'pdp_class_id': 200 + i, 'name': f'VN{i + 1:02d} sp'} for i in range(25)]


def below(cfg, learned=200):
    return [c for t in cfg.tasks for c in t.classes if c.label < learned]


def test_new_task_after_the_data_tasks_then_reserved_slots():
    cfg = new_task_config(BASE, PRODUCTS, reserve_after=25, name='vn')
    assert [t.num_classes for t in cfg.tasks] == [100, 25, 25, 25, 25, 25, 25]
    assert (cfg.num_slots, cfg.num_classes) == (250, 251)
    new, reserved = cfg.task(6), cfg.task(7)
    assert not new.reserved and reserved.reserved
    assert [c.label for c in new.classes] == list(range(200, 225))
    assert [c.rpc_category_id for c in new.classes] == list(range(201, 226))  # = category_id of the synthetic images
    assert [c.name for c in new.classes] == [p['name'] for p in PRODUCTS]
    assert all(c.alias_of is None for c in new.classes)
    assert len(cfg.data_tasks) == 6 and cfg.seen_classes(6) == 225
    # the classes the base model learned stay identical: what pdp/expand.py checks
    assert below(cfg) == below(BASE) and cfg.data_tasks[:5] == BASE.data_tasks


def test_no_reserved_task_when_reserve_after_is_zero():
    cfg = new_task_config(BASE, PRODUCTS[:3], reserve_after=0, name='vn3')
    assert [t.num_classes for t in cfg.tasks] == [100, 25, 25, 25, 25, 3] and cfg.num_classes == 204


def test_known_id_is_refused():
    with pytest.raises(ValueError, match='mode improve'):
        new_task_config(BASE, [{'name': 'x', 'rpc_category_id': 3}], reserve_after=0, name='bad')
    with pytest.raises(ValueError, match='already'):
        new_task_config(BASE, [{'name': 'a', 'rpc_category_id': 300}, {'name': 'b', 'rpc_category_id': 300}],
                        reserve_after=0, name='dup')


def test_labels_must_follow_the_listed_order():
    shifted = [{**p, 'pdp_class_id': p['pdp_class_id'] + 1} for p in PRODUCTS]
    with pytest.raises(ValueError, match='listed by label'):
        new_task_config(BASE, shifted, reserve_after=25, name='bad')


def test_cli_writes_a_valid_config(tmp_path):
    import json

    products = tmp_path / 'products.json'
    products.write_text(json.dumps(PRODUCTS, ensure_ascii=False))
    out = tmp_path / 'cfg.json'
    main(['--base', str(REPO / 'configs/tasks_100-4x25_seed0.json'), '--products', str(products), '--name', 'vn',
          '--out', str(out)])
    assert TaskConfig.load(out) == new_task_config(BASE, PRODUCTS, reserve_after=25, name='vn')
