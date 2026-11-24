"""P2 (docs/design/2026-11-22-toi-uu-2-luot): the teacher's pseudo-labels of an image are cached by image_id.

Within a task the teacher, the old-class prototypes and (without augmentation) the images are fixed, so the
pseudo-labels of an image are the same every epoch. The cache only stores and serves batches without padding:
padding depends on the other images of the batch and could change the teacher's output at the image borders.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from pdp_helpers import make_batch, make_trainer, pdp_args, use_tiny_detr

pytestmark = pytest.mark.slow

LOW_THRESHOLD = ("--pseudo_thresh_high", "0.01")


def task2_trainer(tmp_path, *extra):
    torch.manual_seed(0)
    trainer = make_trainer(pdp_args(tmp_path, *LOW_THRESHOLD, *extra), task_id=2)
    trainer.set_teacher()
    trainer.train()
    return trainer


def sub(tmp_path, name):
    path = tmp_path / name
    path.mkdir(exist_ok=True)
    return path


def training_labels(trainer, batch):
    """The targets (with pseudo-labels) the training pass of common_step gets."""
    seen = []
    forward = trainer.model.forward

    def spy(*args, **kwargs):
        if kwargs.get("train"):
            seen.append(kwargs["labels"])
        return forward(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trainer.model, "forward", spy)
        trainer.common_step(batch, 0)
    return seen[-1]


def unique_batch(labels_per_image, first_id, seed=0):
    """make_batch numbers the images 1..B in every batch; the cache keys by image_id, so give unique ids."""
    batch = make_batch(labels_per_image, seed=seed)
    for offset, target in enumerate(batch["labels"]):
        target["image_id"] = torch.tensor([first_id + offset])
    return batch


def assert_same_labels(got, expected):
    for a, b in zip(got, expected, strict=True):
        assert torch.equal(a["class_labels"], b["class_labels"]) and torch.equal(a["boxes"], b["boxes"])


def padded(batch):
    batch["pixel_mask"][1, -20:, :] = 0
    return batch


@pytest.fixture(autouse=True)
def tiny(monkeypatch):
    use_tiny_detr(monkeypatch)


def test_cached_pseudo_labels_equal_recomputed(tmp_path):
    batch = make_batch([[3], [3, 4]])
    off = task2_trainer(sub(tmp_path, "off"), "--cache_pseudo", "0")
    expected = training_labels(off, batch)
    assert sum(len(t["class_labels"]) for t in expected) > 3
    on = task2_trainer(sub(tmp_path, "on"), "--cache_pseudo", "auto")
    first = training_labels(on, batch)
    second = training_labels(on, batch)
    assert_same_labels(first, expected)
    assert_same_labels(second, expected)


def test_teacher_not_called_when_batch_cached(tmp_path):
    trainer = task2_trainer(tmp_path, "--cache_pseudo", "auto")
    batch = make_batch([[3], [3, 4]])
    trainer.common_step(batch, 0)
    trainer.common_step(batch, 1)
    assert trainer.teacher_forward_calls == 1
    assert trainer.pseudo_cache_hits == 2


def test_padded_batch_never_uses_the_cache(tmp_path):
    trainer = task2_trainer(tmp_path, "--cache_pseudo", "auto")
    batch = padded(make_batch([[3], [3, 4]]))
    trainer.common_step(batch, 0)
    trainer.common_step(batch, 1)
    assert trainer.teacher_forward_calls == 2 and trainer.pseudo_cache_hits == 0


def test_cache_cleared_on_set_teacher(tmp_path):
    trainer = task2_trainer(tmp_path, "--cache_pseudo", "auto")
    batch = make_batch([[3], [3, 4]])
    trainer.common_step(batch, 0)
    trainer.set_teacher()
    trainer.common_step(batch, 1)
    assert trainer.teacher_forward_calls == 2


@pytest.mark.parametrize("blocker", [("--augment", "1"), ("--feature_kd", "1"), ("--ppg_legacy", "1")])
def test_cache_pseudo_1_rejects_augment_feature_kd_legacy(tmp_path, blocker):
    with pytest.raises(ValueError, match="cache_pseudo"):
        make_trainer(pdp_args(tmp_path, "--cache_pseudo", "1", *blocker), task_id=2)


def test_cache_pseudo_auto_disables_and_logs(tmp_path):
    trainer = task2_trainer(tmp_path, "--cache_pseudo", "auto", "--augment", "1")
    batch = make_batch([[3], [3, 4]])
    trainer.common_step(batch, 0)
    trainer.common_step(batch, 1)
    assert trainer.teacher_forward_calls == 2
    trainer.args.log_file.flush()
    log = (tmp_path / "log.txt").read_text()
    assert "P2: pseudo-label cache off" in log and "--augment" in log


def test_pseudo_label_total_unchanged_and_cache_hits_recorded(tmp_path):
    batches = [unique_batch([[3], [3, 4]], first_id=1), unique_batch([[4], [3]], first_id=3, seed=1)]
    totals = {}
    for mode in ("0", "auto"):
        trainer = task2_trainer(sub(tmp_path, mode), "--cache_pseudo", mode)
        for step, batch in enumerate(batches * 2):
            trainer.common_step(batch, step)
        totals[mode] = (trainer.ppg_pseudo_labels_total, trainer.pseudo_cache_hits, trainer.teacher_forward_calls)
    assert totals["0"][0] == totals["auto"][0] > 0
    assert totals["0"][1:] == (0, 4) and totals["auto"][1:] == (4, 2)


def test_training_unchanged_with_cache(tmp_path):
    """Two epochs of two batches with real optimizer steps: same losses and gradients with and without cache."""
    runs = {}
    batches = [unique_batch([[3], [3, 4]], first_id=1), unique_batch([[4], [3]], first_id=3, seed=1)]
    for mode in ("0", "auto"):
        trainer = task2_trainer(sub(tmp_path, mode), "--cache_pseudo", mode)
        optimizer = trainer.configure_optimizers()["optimizer"]
        torch.manual_seed(1)
        losses, grads = [], []
        for step, batch in enumerate(batches * 2):
            loss, _ = trainer.common_step(batch, step)
            loss.backward()
            losses.append(loss.detach().clone())
            grads.append({n: p.grad.clone() for n, p in trainer.named_parameters() if p.grad is not None})
            optimizer.step()
            optimizer.zero_grad()
        runs[mode] = (losses, grads)
    assert all(torch.equal(a, b) for a, b in zip(runs["0"][0], runs["auto"][0], strict=True))
    for old, new in zip(runs["0"][1], runs["auto"][1], strict=True):
        assert old.keys() == new.keys() and all(torch.equal(old[n], new[n]) for n in old)


def test_cache_not_changed_by_later_edits_of_the_targets(tmp_path):
    """The cache keeps its own copies: editing a step's targets in place later must not change what it serves."""
    trainer = task2_trainer(tmp_path, "--cache_pseudo", "auto")
    batch = unique_batch([[3], [3, 4]], first_id=1)
    trainer.common_step(batch, 0)
    stored = {i: tuple(t.clone() for t in trainer.pseudo_cache[i]) for i in (1, 2)}
    assert sum(len(classes) for _, classes in stored.values()) > 0
    for boxes, _ in trainer.pseudo_cache.values():
        for target in batch["labels"]:
            assert target["boxes"].data_ptr() != boxes.data_ptr()
    for target in batch["labels"]:
        target["boxes"].mul_(0)
    for i in (1, 2):
        assert all(torch.equal(a, b) for a, b in zip(trainer.pseudo_cache[i], stored[i], strict=True))


def test_training_contract_includes_new_flags_only_when_set(tmp_path):
    import runtime

    default = runtime.training_contract(pdp_args(sub(tmp_path, "a")))
    assert "reuse_features" not in default and "cache_pseudo" not in default
    flags = runtime.training_contract(pdp_args(sub(tmp_path, "b"), "--reuse_features", "1", "--cache_pseudo", "auto"))
    assert flags["reuse_features"] == 1 and flags["cache_pseudo"] == "auto"


def test_progress_records_pseudo_cache_hits(tmp_path):
    from autocheckout.io import load_json

    import checkpointing

    callback = checkpointing.TrainingProgress(tmp_path)
    teacher = torch.nn.Linear(2, 2)
    teacher.requires_grad_(False)
    module = SimpleNamespace(task_id=2, teacher=teacher, PREV_INTRODUCED_CLS=2, seen_classes=3,
                             teacher_forward_calls=1, ppg_pseudo_labels_total=5, pseudo_cache_hits=3,
                             missing_prototypes=lambda: [])
    trainer = SimpleNamespace(global_step=0, current_epoch=0, optimizers=[SimpleNamespace(param_groups=[{'lr': 1e-4}])],
                              callback_metrics={'tr': torch.tensor(1.0)}, is_global_zero=True)
    callback.on_train_start(trainer, module)
    trainer.global_step = 1
    callback.on_train_batch_end(trainer, module, {}, {}, 0)
    callback.on_train_end(trainer, module)
    rows = [json.loads(line) for line in (tmp_path / 'step_metrics.jsonl').read_text().splitlines()]
    assert rows[0]['pseudo_cache_hits'] == 3
    assert load_json(tmp_path / 'progress.json')['pseudo_cache_hits'] == 3
