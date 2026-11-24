"""A (docs/project/attention-prompt-plan-v2-2026-11-24.md): every object query picks its own prompt mixture.

With --per_query_prompt 1 the 300 queries of the prompt-free pass select prompts one by one, instead of one
mixture for the whole image built by query_tf.
"""

import pytest
import torch

import runtime
from pdp_helpers import make_batch, make_trainer, pdp_args, use_tiny_detr
from two_pass import prompted_outputs

pytestmark = pytest.mark.slow

PER_QUERY = ("--per_query_prompt", "1", "--local_query", "0", "--lambda_query", "0")
QUERY_TF_PARAMETERS = 300 * 256 * 300 + 300


def sub(tmp_path, name):
    path = tmp_path / name
    path.mkdir(exist_ok=True)
    return path


def per_query_trainer(tmp_path, monkeypatch, *extra, task_id=1):
    use_tiny_detr(monkeypatch)
    return make_trainer(pdp_args(tmp_path, *PER_QUERY, *extra), task_id=task_id)


def test_per_query_weights_match_image_level_when_all_queries_are_the_same(tmp_path, monkeypatch):
    prompts = per_query_trainer(tmp_path, monkeypatch).model.model.prompts
    image_query = torch.randn(2, 256)
    with torch.no_grad():
        (ek, ev), _, _ = prompts.forward(image_query, 0, None)
        (ek_q, ev_q), _, _ = prompts.forward(image_query[:, None].expand(2, 7, 256), 0, None)
    assert ek_q.shape == (2, 7, *ek.shape[1:]) and ev_q.shape == (2, 7, *ev.shape[1:])
    torch.testing.assert_close(ek_q, ek[:, None].expand_as(ek_q))
    torch.testing.assert_close(ev_q, ev[:, None].expand_as(ev_q))


def test_per_query_prefix_attention_matches_shared_prefix_when_prefixes_are_the_same(tmp_path, monkeypatch):
    model = per_query_trainer(tmp_path, monkeypatch).model.eval()
    attention = model.model.decoder.layers[0].self_attn
    hidden, position = torch.randn(2, 9, 256), torch.randn(2, 9, 256)
    pk, pv = torch.randn(2, 3, 256), torch.randn(2, 3, 256)
    per_query = [t[:, None].expand(2, 9, 3, 256) for t in (pk, pv)]
    with torch.no_grad():
        shared, _ = attention(hidden, position_embeddings=position, prompt_list=[pk, pv])
        got, _ = attention(hidden, position_embeddings=position, prompt_list=per_query)
    torch.testing.assert_close(got, shared)


def test_per_query_prompts_have_no_query_tf(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    image_level = make_trainer(pdp_args(sub(tmp_path, "image")), task_id=1).model
    per_query = make_trainer(pdp_args(sub(tmp_path, "query"), *PER_QUERY), task_id=1).model
    assert not hasattr(per_query.model.prompts, "query_tf")
    count = lambda model: sum(p.numel() for p in model.parameters())  # noqa: E731
    assert count(image_level) - count(per_query) == QUERY_TF_PARAMETERS


def test_per_query_prompt_rejects_local_query(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    with pytest.raises(ValueError, match="per_query_prompt"):
        make_trainer(pdp_args(tmp_path, "--per_query_prompt", "1", "--local_query", "1"), task_id=1)


def test_training_step_has_no_query_loss_and_trains_only_current_private_prompts(tmp_path, monkeypatch):
    trainer = per_query_trainer(tmp_path, monkeypatch, "--ddl_lambda", "0", task_id=2)
    trainer.model.model.prompts.init_task_prompts()  # as main() does when a task starts
    trainer.train()
    loss, loss_dict = trainer.common_step(make_batch([[0, 3], [4]]), 0)
    assert "query_loss" not in loss_dict
    loss.backward()
    prompts = trainer.model.model.prompts
    old = sum(prompts.pool_sizes[:1])
    grad = prompts.private_p_0.grad
    assert grad is not None
    assert grad[:old].abs().sum() == 0
    assert grad[old:].abs().sum() > 0


@pytest.mark.parametrize("use_reuse", [False, True])
def test_prompted_outputs_pass_every_query_and_reuse_gives_identical_outputs(tmp_path, monkeypatch, use_reuse):
    """The callers (teacher, validation, predict_batch of checkout-system) keep passing local_query=False."""
    model = per_query_trainer(tmp_path, monkeypatch).model.eval()
    batch = make_batch([[0], [1, 2]])
    pixels, mask = batch["pixel_values"], batch["pixel_mask"]
    with torch.no_grad():
        first = model(pixel_values=pixels, pixel_mask=mask, train=False)
        expected = model(pixel_values=pixels, pixel_mask=mask, query=first.last_hidden_state, train=False)
        got = prompted_outputs(model, pixels, mask, use_prompts=True, local_query=False, reuse_features=use_reuse)
    for name in ("logits", "pred_boxes", "last_hidden_state"):
        assert torch.equal(getattr(got, name), getattr(expected, name)), name


def test_teacher_uses_every_query(tmp_path, monkeypatch):
    trainer = per_query_trainer(tmp_path, monkeypatch, task_id=2)
    teacher = per_query_trainer(sub(tmp_path, "teacher"), monkeypatch, task_id=1).model.eval()
    trainer.__dict__["teacher"] = teacher
    batch = make_batch([[0], [1]])
    pixels, mask = batch["pixel_values"], batch["pixel_mask"]
    with torch.no_grad():
        got = trainer.teacher_outputs(pixels, mask)
        first = teacher(pixel_values=pixels, pixel_mask=mask, train=False, task_id=trainer.task_id - 1)
        expected = teacher(pixel_values=pixels, pixel_mask=mask, query=first.last_hidden_state, train=False,
                           task_id=trainer.task_id - 1)
    assert torch.equal(got.logits, expected.logits)


def test_per_query_run_starts_from_an_image_level_checkpoint(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    image_level = make_trainer(pdp_args(sub(tmp_path, "image")), task_id=1)
    path = tmp_path / "task_final.pth"
    image_level.save_task_final(str(path))

    args = pdp_args(sub(tmp_path, "query"), *PER_QUERY)
    args._runtime_snapshot = runtime.checkpoint_runtime(str(path))
    trainer = make_trainer(args, task_id=2)
    trainer.resume(str(path))

    prompts, old = trainer.model.model.prompts, image_level.model.model.prompts
    assert not hasattr(prompts, "query_tf")
    assert torch.equal(prompts.shared_p_0, old.shared_p_0)
    args.log_file.flush()
    assert "query_tf" in (tmp_path / "query" / "log.txt").read_text()


def test_image_level_run_cannot_resume_a_per_query_checkpoint(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    per_query = make_trainer(pdp_args(sub(tmp_path, "query"), *PER_QUERY), task_id=1)
    path = tmp_path / "task_final.pth"
    per_query.save_task_final(str(path))

    args = pdp_args(sub(tmp_path, "image"))
    args._runtime_snapshot = runtime.checkpoint_runtime(str(path))
    with pytest.raises(ValueError, match="conflicts with checkpoint runtime"):
        make_trainer(args, task_id=2)


def test_training_contract_records_per_query_prompt_only_when_set(tmp_path):
    assert "per_query_prompt" not in runtime.training_contract(pdp_args(sub(tmp_path, "off")))
    assert runtime.training_contract(pdp_args(sub(tmp_path, "on"), *PER_QUERY))["per_query_prompt"] == 1
