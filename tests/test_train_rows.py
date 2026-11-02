"""--train_rows (update-model plan, F1/F2/I9): which classifier rows a task trains.

Without the flag a task trains the rows of its own classes, as before. With it, exactly the listed rows train
(the last row is never frozen, as before); every other row keeps its loaded values bit for bit, also under AdamW
weight decay, which shrinks a parameter even when its gradient is zero.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pdp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdp_helpers import make_batch, make_trainer, pdp_args, use_tiny_detr  # noqa: E402

# toy config (3, 2): task 2 has PREV_INTRODUCED_CLS 3, seen 5, n_classes 6 (row 5 = background)
BATCH = [[3, 4], [4]]


def trainer_for(tmp_path, monkeypatch, *extra):
    use_tiny_detr(monkeypatch)
    trainer = make_trainer(pdp_args(tmp_path, *extra), task_id=2)
    trainer.train()
    return trainer


def rows_file(tmp_path, rows):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows))
    return str(path)


def grad_rows(trainer):
    """Rows of the classifier with a non-zero gradient after one backward and on_after_backward. The last row
    (background) is never frozen but gets no gradient from the loss, with or without the flag."""
    loss, _ = trainer.common_step(make_batch(BATCH), 0)
    loss.backward()
    trainer.on_after_backward()
    rows = set()
    for head in trainer.model.class_embed:
        nonzero = head.weight.grad.abs().sum(dim=1) + head.bias.grad.abs() > 0
        rows |= set(torch.nonzero(nonzero).flatten().tolist())
    return rows


def test_without_the_flag_a_task_trains_its_own_rows(tmp_path, monkeypatch):
    trainer = trainer_for(tmp_path, monkeypatch)
    assert trainer.frozen_rows is None
    assert grad_rows(trainer) == {3, 4}


def test_only_the_listed_rows_train(tmp_path, monkeypatch):
    trainer = trainer_for(tmp_path, monkeypatch, "--train_rows", rows_file(tmp_path, [1, 3]))
    assert grad_rows(trainer) == {1, 3}


def test_all_trains_every_seen_row(tmp_path, monkeypatch):
    trainer = trainer_for(tmp_path, monkeypatch, "--train_rows", "all")
    assert grad_rows(trainer) == {0, 1, 2, 3, 4}


def test_a_row_of_a_class_not_seen_yet_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="train_rows"):
        trainer_for(tmp_path, monkeypatch, "--train_rows", rows_file(tmp_path, [1, 5]))


def test_frozen_rows_keep_their_values_under_weight_decay(tmp_path, monkeypatch):
    # lr and weight decay large enough that 1 - lr * wd is not rounded to 1.0 in float32
    trainer = trainer_for(tmp_path, monkeypatch, "--train_rows", rows_file(tmp_path, [1, 3]),
                          "--lr", "0.1", "--weight_decay", "0.1")
    before = [(h.weight.detach().clone(), h.bias.detach().clone()) for h in trainer.model.class_embed]
    optimizer = trainer.configure_optimizers()["optimizer"]
    trainer.on_fit_start()
    frozen = [0, 2, 4]
    for step in range(3):
        optimizer.zero_grad()
        loss, _ = trainer.common_step(make_batch(BATCH, seed=step), 0)
        loss.backward()
        trainer.on_after_backward()
        optimizer.step()
        if step == 0:  # weight decay alone moves the frozen rows: the restore below is needed
            assert not torch.equal(trainer.model.class_embed[0].weight[frozen], before[0][0][frozen])
        trainer.on_train_batch_end(None, None, step)
    for head, (weight, bias) in zip(trainer.model.class_embed, before, strict=True):
        assert torch.equal(head.weight[frozen], weight[frozen])
        assert torch.equal(head.bias[frozen], bias[frozen])
        assert not torch.equal(head.weight[[1, 3]], weight[[1, 3]])
