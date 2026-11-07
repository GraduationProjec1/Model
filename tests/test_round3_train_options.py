"""Round 3 (docs/design/2026-11-07-vong-3, Task 6): the CLI flag reaches the prompts, and lr_old=0 freezes
everything that is not a new parameter (the r3_freeze recipe)."""

import pytest
import torch

from pdp_helpers import make_batch, make_trainer, pdp_args, use_tiny_detr

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("flag", [0, 1])
def test_tune_old_private_flag_reaches_prompts(tmp_path, monkeypatch, flag):
    use_tiny_detr(monkeypatch)
    trainer = make_trainer(pdp_args(tmp_path, "--tune_old_private", str(flag)), task_id=2)
    assert trainer.model.model.prompts.tune_old_private is bool(flag)


def test_lr_old_zero_keeps_shared_parameters(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    trainer = make_trainer(pdp_args(tmp_path, "--lr_old", "0", "--optim_groups", "prompt"), task_id=2)
    opt = trainer.configure_optimizers()["optimizer"]
    assert opt.param_groups[1]["lr"] == 0
    old = [p for p in opt.param_groups[1]["params"]]
    assert old, "no shared parameter is trainable, so the test proves nothing"
    before = [p.detach().clone() for p in old]
    trainer.train()
    _, loss_dict = trainer.common_step(make_batch([[3, 4], [3]]), 0)
    loss = sum(v for k, v in loss_dict.items() if k.startswith("loss") and torch.is_tensor(v) and v.requires_grad)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert all(torch.equal(b, p.detach()) for b, p in zip(before, old, strict=True))
