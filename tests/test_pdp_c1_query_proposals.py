"""C1 (docs/ai/planning/2026-11-25-plan-B-C1.md): K extra object queries start where the encoder sees an object.

The proposals of the first pass are reused by the second (the per-query prompts of A pair the queries of the two
passes by index), and an auxiliary binary loss trains the proposal head while the decoder stays frozen.
"""

import pytest
import torch

import runtime
from pdp_helpers import make_batch, make_trainer, pdp_args, use_tiny_detr
from two_pass import prompted_outputs

pytestmark = pytest.mark.slow

PER_QUERY = ("--per_query_prompt", "1", "--local_query", "0", "--lambda_query", "0")
K = 20
C1 = ("--query_proposals", str(K))


def sub(tmp_path, name):
    path = tmp_path / name
    path.mkdir(exist_ok=True)
    return path


def trainer(tmp_path, monkeypatch, name, *extra, task_id=1):
    use_tiny_detr(monkeypatch)
    return make_trainer(pdp_args(sub(tmp_path, name), *extra), task_id=task_id)


def outputs(model, batch, reuse=False):
    with torch.no_grad():
        return prompted_outputs(model, batch["pixel_values"], batch["pixel_mask"], use_prompts=True,
                                local_query=model.config.local_query, reuse_features=reuse)


def test_without_c1_there_is_no_proposal_module(tmp_path, monkeypatch):
    model = trainer(tmp_path, monkeypatch, "off", *PER_QUERY).model
    assert getattr(model.model, "proposals", None) is None
    assert outputs(model.eval(), make_batch([[0]])).logits.shape[1] == 300


@pytest.mark.parametrize("selection", [PER_QUERY, ("--local_query", "0")], ids=["per_query", "mean_query"])
def test_c1_adds_k_queries_and_predicts(tmp_path, monkeypatch, selection):
    model = trainer(tmp_path, monkeypatch, "on", *selection, *C1).model.eval()
    out = outputs(model, make_batch([[0], [1, 2]]))
    assert out.logits.shape[1] == 300 + K and torch.isfinite(out.logits).all()
    assert out.proposal_boxes.shape == (2, K, 4) and torch.isfinite(out.proposal_boxes).all()


def test_the_second_pass_reuses_the_proposals_of_the_first(tmp_path, monkeypatch):
    model = trainer(tmp_path, monkeypatch, "on", *PER_QUERY, *C1).model.eval()
    batch = make_batch([[0], [1, 2]])
    pixels, mask = batch["pixel_values"], batch["pixel_mask"]
    with torch.no_grad():
        first = model(pixel_values=pixels, pixel_mask=mask, train=False)
        given = first.proposal_boxes + 0.25
        second = model(pixel_values=pixels, pixel_mask=mask, query=first.last_hidden_state, train=False,
                       proposals=given)
        both = prompted_outputs(model, pixels, mask, use_prompts=True, local_query=False)
    assert torch.equal(second.proposal_boxes, given)
    assert torch.equal(both.proposal_boxes, first.proposal_boxes)


@pytest.mark.parametrize("use_reuse", [False, True])
def test_reuse_gives_the_same_outputs_with_c1(tmp_path, monkeypatch, use_reuse):
    model = trainer(tmp_path, monkeypatch, "on", *PER_QUERY, *C1).model.eval()
    batch = make_batch([[0], [1, 2]])
    expected, got = outputs(model, batch), outputs(model, batch, reuse=use_reuse)
    assert torch.equal(got.logits, expected.logits) and torch.equal(got.proposal_boxes, expected.proposal_boxes)


def test_training_step_trains_the_proposal_head_with_its_own_loss(tmp_path, monkeypatch):
    t = trainer(tmp_path, monkeypatch, "on", *PER_QUERY, *C1, "--ddl_lambda", "0",
                "--new_params", "class_embed,prompts.private,proposals", task_id=2)
    t.model.model.prompts.init_task_prompts()
    t.train()
    names = [n for n, _ in t.model.named_parameters() if ".proposals." in n]
    assert names and all(p.requires_grad for n, p in t.model.named_parameters() if n in names)
    assert not any(p.requires_grad for n, p in t.model.named_parameters() if ".decoder." in n)
    loss, loss_dict = t.common_step(make_batch([[0, 3], [4]]), 0)
    assert "loss_prop" in loss_dict and loss_dict["loss_prop"] > 0
    loss.backward()
    assert t.model.model.proposals.score.weight.grad.abs().sum() > 0
    assert t.model.model.proposals.pos_trans.weight.grad.abs().sum() > 0


def test_c1_rejects_local_query(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="query_proposals"):
        trainer(tmp_path, monkeypatch, "on", "--local_query", "1", *C1)


def test_c1_run_starts_from_a_checkpoint_without_c1(tmp_path, monkeypatch):
    old = trainer(tmp_path, monkeypatch, "old")
    path = tmp_path / "task_final.pth"
    old.save_task_final(str(path))

    args = pdp_args(sub(tmp_path, "new"), *PER_QUERY, *C1)
    args._runtime_snapshot = runtime.checkpoint_runtime(str(path))
    new = make_trainer(args, task_id=2)
    new.resume(str(path))
    assert torch.equal(new.model.model.prompts.shared_p_0, old.model.model.prompts.shared_p_0)
    args.log_file.flush()
    assert "proposals.score" in (tmp_path / "new" / "log.txt").read_text()


def test_training_contract_records_query_proposals_only_when_set(tmp_path):
    assert "query_proposals" not in runtime.training_contract(pdp_args(sub(tmp_path, "off")))
    contract = runtime.training_contract(pdp_args(sub(tmp_path, "on"), *C1, "--lambda_prop", "2"))
    assert contract["query_proposals"] == K and contract["lambda_prop"] == 2


def test_training_second_pass_gets_the_proposals_of_the_first(tmp_path, monkeypatch):
    """In train mode the encoder of the student runs twice (dropout): the second pass must not pick its own top-K."""
    t = trainer(tmp_path, monkeypatch, "on", *PER_QUERY, *C1, "--ddl_lambda", "0", task_id=2)
    t.model.model.prompts.init_task_prompts()
    t.train()
    calls, forward = [], t.model.forward

    def recording(*args, **kwargs):
        out = forward(*args, **kwargs)
        calls.append((kwargs.get("proposals"), out.proposal_boxes))
        return out

    monkeypatch.setattr(t.model, "forward", recording)
    t.common_step(make_batch([[0, 3], [4]]), 0)
    (given_1, made_1), (given_2, made_2) = calls
    assert given_1 is None and torch.equal(given_2, made_1) and torch.equal(made_2, made_1)
