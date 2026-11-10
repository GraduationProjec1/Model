from collections import Counter
from pathlib import Path

import pytest

from autocheckout.io import save_json
from autocheckout.taskcfg import TaskConfig
from tools.make_task_config import allocate, build_task_config, distribution_table, main

# 200 synthetic SKUs in 17 supercategories of uneven size, like RPC.
GROUP_SIZES = [22, 18, 16, 15, 14, 13, 12, 12, 11, 11, 10, 10, 9, 8, 7, 7, 5]
SIZES = [100, 25, 25, 25, 25]


def rpc_like_categories():
    categories = []
    for index, count in enumerate(GROUP_SIZES):
        for _ in range(count):
            cid = len(categories) + 1
            categories.append({"id": cid, "name": f"{cid}_g{index:02d}", "supercategory": f"g{index:02d}"})
    return categories


def test_allocate_small_examples():
    assert allocate({"a": 4, "b": 2}, [3, 3]) == {"a": [2, 2], "b": [1, 1]}
    # Equal remainders go to the supercategory whose name sorts first.
    assert allocate({"c": 1, "b": 1, "a": 1}, [2, 1]) == {"c": [0, 1], "b": [1, 0], "a": [1, 0]}
    with pytest.raises(ValueError, match="add up"):
        allocate({"a": 3}, [2, 2])


def test_task_sizes_are_exact_and_supercategories_proportional():
    categories = rpc_like_categories()
    config = build_task_config(categories, SIZES, reserved=24, seed=0, name="100-4x25_seed0")
    assert [t.num_classes for t in config.tasks] == [*SIZES, 24]
    assert config.num_slots == 224 and config.num_classes == 225
    assert sorted(config.rpc_to_label()) == [c["id"] for c in categories]
    per_group = Counter(c["supercategory"] for c in categories)
    for task, size in zip(config.data_tasks, SIZES, strict=True):
        counts = Counter(c.supercategory for c in task.classes)
        for group, total in per_group.items():
            assert abs(counts[group] - total * size / 200) < 1  # floor or ceiling of the exact share
        rpc_ids = [c.rpc_category_id for c in task.classes]
        assert rpc_ids == sorted(rpc_ids)
    reserved = config.tasks[-1]
    assert reserved.reserved and reserved.offset == 200
    assert [c.name for c in reserved.classes] == [f"reserved_{label}" for label in range(200, 224)]
    assert all(c.rpc_category_id is None for c in reserved.classes)


def test_seed_changes_members_but_not_counts():
    categories = rpc_like_categories()
    first = build_task_config(categories, SIZES, 24, 0, "a")
    assert first == build_task_config(categories, SIZES, 24, 0, "a")
    other = build_task_config(categories, SIZES, 24, 1, "a")
    assert other.rpc_to_label() != first.rpc_to_label()
    assert distribution_table(other) == distribution_table(first)


def test_cli_writes_a_loadable_config(tmp_path, capsys):
    save_json(tmp_path / "instances.json", {"categories": rpc_like_categories()})
    out = tmp_path / "tasks.json"
    main(["--categories", str(tmp_path / "instances.json"), "--name", "100-4x25_seed0", "--out", str(out)])
    loaded = TaskConfig.load(out)
    assert loaded == build_task_config(rpc_like_categories(), SIZES, 24, 0, "100-4x25_seed0")
    printed = capsys.readouterr().out
    assert "total                      100    25    25    25    25    200" in printed
    assert "224 slots" in printed


# --- split of tasks >= 2 (docs/design/2026-11-09-chia-12-13-sku) -------------------------------------------------
def test_split_keeps_task_1_and_reserved_and_splits_every_later_task():
    from tools.make_task_config import split_task_config

    old = build_task_config(rpc_like_categories(), SIZES, reserved=24, seed=0, name="100-4x25_seed0")
    new = split_task_config(old, [12, 13], seed=0, name="100-8x12-13_seed0")
    assert [t.num_classes for t in new.tasks] == [100, 12, 13, 12, 13, 12, 13, 12, 13, 24]
    assert new.num_slots == old.num_slots and new.num_classes == old.num_classes
    assert new.tasks[0].classes == old.tasks[0].classes  # labels, names, RPC ids of task 1 unchanged
    assert new.tasks[-1].reserved and new.tasks[-1].offset == 200
    for t in range(2, 6):
        halves = {c.rpc_category_id for c in new.task(2 * t - 2).classes} | {
            c.rpc_category_id for c in new.task(2 * t - 1).classes}
        assert halves == {c.rpc_category_id for c in old.task(t).classes}
    labels = [c.label for t in new.tasks for c in t.classes]
    assert labels == list(range(224))
    for task in new.data_tasks:
        assert [c.rpc_category_id for c in task.classes] == sorted(c.rpc_category_id for c in task.classes)
    assert split_task_config(old, [12, 13], 0, "x") == split_task_config(old, [12, 13], 0, "x")


def test_split_shares_supercategories_and_refuses_bad_sizes():
    from tools.make_task_config import split_task_config

    old = build_task_config(rpc_like_categories(), SIZES, reserved=24, seed=0, name="a")
    new = split_task_config(old, [12, 13], seed=0, name="b")
    for t in range(2, 6):
        per_group = Counter(c.supercategory for c in old.task(t).classes)
        a, b = (Counter(c.supercategory for c in new.task(n).classes) for n in (2 * t - 2, 2 * t - 1))
        assert all(a[g] + b[g] == n for g, n in per_group.items())
        assert all(abs(a[g] - n * 12 / 25) < 1 for g, n in per_group.items())
    with pytest.raises(ValueError, match="add up"):
        split_task_config(old, [12, 12], seed=0, name="c")


def test_cli_split_from_writes_the_split_config(tmp_path, capsys):
    from tools.make_task_config import split_task_config

    old = build_task_config(rpc_like_categories(), SIZES, reserved=24, seed=0, name="100-4x25_seed0")
    old.save(tmp_path / "old.json")
    out = tmp_path / "split.json"
    main(["--split-from", str(tmp_path / "old.json"), "--split", "12,13", "--name", "100-8x12-13_seed0",
          "--out", str(out)])
    assert TaskConfig.load(out) == split_task_config(old, [12, 13], 0, "100-8x12-13_seed0")
    assert "224 slots (24 reserved)" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--name", "x", "--out", str(out)])  # neither --categories nor --split-from


REAL_OLD = Path(__file__).resolve().parents[1] / "configs/tasks_100-4x25_seed0.json"
REAL_NEW = Path(__file__).resolve().parents[1] / "configs/tasks_100-8x12-13_seed0.json"


def test_committed_split_config_is_the_split_of_the_r1_config():
    """V11 of 03-verify-report.md: 2a is these 12 SKUs; 98_milk lands in 4b (task 7), 97_milk in 5b (task 9)."""
    from tools.make_task_config import split_task_config

    new = split_task_config(TaskConfig.load(REAL_OLD), [12, 13], 0, "100-8x12-13_seed0")
    assert new == TaskConfig.load(REAL_NEW)
    assert sorted(c.name for c in new.task(2).classes) == [
        "115_canned_food", "132_chocolate", "143_candy", "17_dried_fruit", "180_tissue", "24_dried_food",
        "32_instant_drink", "46_instant_noodles", "63_dessert", "73_drink", "93_alcohol", "99_milk"]
    where = {c.name: t.task_id for t in new.data_tasks for c in t.classes}
    assert where["98_milk"] == 7 and where["97_milk"] == 9
