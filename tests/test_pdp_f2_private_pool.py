"""F2: the private pool has one slot per class of every task, and each new task gets trainable prompts."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.prompt import Prompt
from pdp_helpers import RPC_RESERVED, RPC_SIZES

SIZES = list(RPC_SIZES) + [RPC_RESERVED]  # 100, 25, 25, 25, 25, 24 -> 224 slots


def make_prompt(sizes=SIZES):
    torch.manual_seed(0)
    return Prompt(emb_d=256, n_tasks=len(sizes), prompt_param=[100, 10, 0], key_dim=256,
                  args=SimpleNamespace(local_query=0), task_num_classes=sizes)


def task_slice(prompt, task_count):
    s = sum(prompt.pool_sizes[:task_count])
    return s, s + prompt.pool_sizes[task_count]


def private(prompt, name, layer=0):
    return getattr(prompt, f"private_{name}_{layer}")


def train_step(prompt, lr=1e-2):
    """One optimizer step on a loss that depends on every layer's prompts."""
    opt = torch.optim.AdamW([p for p in prompt.parameters()], lr=lr, weight_decay=0.0)
    query = torch.randn(4, 256)
    loss = 0
    for layer in prompt.e_layers:
        (ek, ev), _, _ = prompt(query, layer, None, train=True)
        loss = loss + ek.pow(2).sum() + ev.sum()
    opt.zero_grad()
    loss.backward()
    grads = {n: p.grad.clone() if p.grad is not None else None for n, p in prompt.named_parameters()}
    opt.step()
    return grads


@pytest.fixture(scope="module")
def prompt_after_task1():
    """Prompt module as it is when task 2 starts: task 1 trained, task 2 slots initialised."""
    prompt = make_prompt()
    train_step(prompt)  # stand-in for training task 1
    prompt.set_task_id(1)
    prompt.init_task_prompts()
    return prompt


def test_pool_has_one_slot_per_class_of_every_task():
    prompt = make_prompt()
    assert prompt.private_size == 224
    assert prompt.pool_sizes == SIZES
    assert private(prompt, "p").shape == (224, 10, 256)
    # constructor initialises task 1 only; later slots are zero until their task starts
    assert private(prompt, "k")[:100].norm(dim=1).min() > 0.99
    assert private(prompt, "k")[100:].abs().sum() == 0


def test_without_fix_new_task_slots_are_zero_and_get_no_gradient():
    """Documents the original bug: a new task's slots are zero and their gradient is zero."""
    prompt = make_prompt()
    prompt.set_task_id(1)  # no init_task_prompts()
    grads = train_step(prompt)
    s, f = task_slice(prompt, 1)
    for name in ("p", "k", "a"):
        assert private(prompt, name)[s:f].abs().sum() == 0
        assert grads[f"private_{name}_0"][s:f].abs().sum() == 0


def test_new_task_slots_are_nonzero_orthogonal_and_trainable(prompt_after_task1):
    prompt = prompt_after_task1
    s, f = task_slice(prompt, 1)
    assert (s, f) == (100, 125)
    for layer in prompt.e_layers:
        for name in ("p", "k", "a"):
            new = private(prompt, name, layer)[s:f].reshape(f - s, -1)
            old = private(prompt, name, layer)[:s].reshape(s, -1)
            assert new.norm(dim=1).min() > 0.99
            # new slots are orthogonal to the slots of the previous task
            cos = F.normalize(new, dim=1) @ F.normalize(old, dim=1).T
            assert cos.abs().max() < 1e-2
            # later tasks untouched
            assert private(prompt, name, layer)[f:].abs().sum() == 0

    grads = train_step(prompt)
    for name in ("p", "k", "a"):
        assert grads[f"private_{name}_0"][s:f].abs().sum() > 0


def test_old_task_slots_do_not_change(prompt_after_task1):
    prompt = prompt_after_task1
    s, _ = task_slice(prompt, 1)
    before = {n: private(prompt, n).detach().clone() for n in ("p", "k", "a")}
    grads = train_step(prompt)
    for name in ("p", "k", "a"):
        assert grads[f"private_{name}_0"][:s].abs().sum() == 0
        assert torch.equal(private(prompt, name)[:s], before[name][:s])


def test_every_task_of_the_rpc_config_runs():
    prompt = make_prompt()
    for task_count in range(len(SIZES)):
        prompt.set_task_id(task_count)
        if task_count:
            prompt.init_task_prompts()
        for train in (True, False):
            (ek, ev), _, _ = prompt(torch.randn(2, 256), 0, None, train=train)
            assert ek.shape == (2, 5, 256) and ev.shape == (2, 5, 256)
    assert private(prompt, "k").norm(dim=1).min() > 0.99  # all 224 slots initialised



# --- round 3 (docs/design/2026-11-07-vong-3, Task 6): --tune_old_private -------------------------------------------
@pytest.mark.parametrize("tune, moves", [(0, False), (1, True)])
def test_tune_old_private_lets_task1_prompts_learn_in_task5(tune, moves):
    torch.manual_seed(0)
    prompt = Prompt(emb_d=256, n_tasks=len(SIZES), prompt_param=[100, 10, 0], key_dim=256,
                    args=SimpleNamespace(local_query=0, tune_old_private=tune), task_num_classes=SIZES)
    train_step(prompt)  # stand-in for task 1
    prompt.set_task_id(4)  # task 5 again, as the improve mode does
    prompt.init_task_prompts()
    before = private(prompt, "p")[:100].detach().clone()  # task 1 slots
    grads = train_step(prompt)
    for name in ("p", "k", "a"):
        assert bool(grads[f"private_{name}_0"][:100].abs().sum() > 0) == moves
    assert (not torch.equal(before, private(prompt, "p")[:100].detach())) == moves
    s, f = task_slice(prompt, 4)
    assert grads["private_p_0"][s:f].abs().sum() > 0  # the current task trains either way


def test_tune_old_private_defaults_off():
    assert make_prompt().tune_old_private is False
