"""B (docs/ai/planning/2026-11-25-plan-B-C1.md): the prompt of every query also corrects the deformable cross-attention.

alpha * W_off(p_q) is added to the sampling offsets and alpha * W_att(p_q) to the attention logits of the decoder
cross-attention, with W = 0 at the start so a model with B starts exactly where its checkpoint was.
"""

import pytest
import torch

import runtime
from pdp_helpers import make_batch, make_trainer, pdp_args, use_tiny_detr
from two_pass import prompted_outputs

pytestmark = pytest.mark.slow

PER_QUERY = ("--per_query_prompt", "1", "--local_query", "0", "--lambda_query", "0")
B = ("--prompt_cross_attn", "1")
XATTN_PARAMETERS = 6 * (256 * 256 + 256 + 256 * 128 + 128) + 6


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


def with_b_weights_of(model_b, model):
    """model_b (B on) with the weights of model (B off); the B weights keep their initial values."""
    missing, unexpected = model_b.load_state_dict(model.state_dict(), strict=False)
    assert unexpected == [] and missing and all(".prompts.xattn_" in key for key in missing)
    return model_b.eval()


@pytest.mark.parametrize("selection", [PER_QUERY, ()], ids=["per_query", "image_level"])
def test_b_starts_with_the_same_outputs_as_without_b(tmp_path, monkeypatch, selection):
    model = trainer(tmp_path, monkeypatch, "off", *selection).model.eval()
    model_b = with_b_weights_of(trainer(tmp_path, monkeypatch, "on", *selection, *B).model, model)
    batch = make_batch([[0], [1, 2]])
    expected, got = outputs(model, batch), outputs(model_b, batch)
    for name in ("logits", "pred_boxes", "last_hidden_state", "encoder_last_hidden_state"):
        assert torch.equal(getattr(got, name), getattr(expected, name)), name


def test_b_moves_the_outputs_once_its_weights_are_not_zero(tmp_path, monkeypatch):
    model = trainer(tmp_path, monkeypatch, "on", *PER_QUERY, *B).model.eval()
    batch = make_batch([[0], [1, 2]])
    before = outputs(model, batch)
    with torch.no_grad():
        model.model.prompts.xattn_off[0].weight.normal_(0, 0.05)
    after = outputs(model, batch)
    assert not torch.equal(after.logits, before.logits)
    assert torch.equal(after.encoder_last_hidden_state, before.encoder_last_hidden_state)


def test_alpha_stays_within_alpha_max(tmp_path, monkeypatch):
    prompts = trainer(tmp_path, monkeypatch, "on", *PER_QUERY, *B, "--xattn_alpha_max", "0.3").model.model.prompts
    with torch.no_grad():
        prompts.xattn_beta.copy_(torch.tensor([-50.0, -1.0, 0.0, 0.1, 1.0, 50.0]))
    alpha = prompts.xattn_alpha()
    assert alpha.abs().max() <= 0.3 + 1e-6 and alpha[2] == 0


@pytest.mark.parametrize("use_reuse", [False, True])
def test_reuse_gives_the_same_outputs_with_b(tmp_path, monkeypatch, use_reuse):
    model = trainer(tmp_path, monkeypatch, "on", *PER_QUERY, *B).model.eval()
    with torch.no_grad():
        model.model.prompts.xattn_off[0].weight.normal_(0, 0.05)
    batch = make_batch([[0], [1, 2]])
    expected, got = outputs(model, batch), outputs(model, batch, reuse=use_reuse)
    assert torch.equal(got.logits, expected.logits)


def test_b_weights_train_at_lr_although_the_decoder_is_frozen(tmp_path, monkeypatch):
    t = trainer(tmp_path, monkeypatch, "on", *PER_QUERY, *B, "--ddl_lambda", "0",
                "--new_params", "class_embed,prompts.private,prompts.xattn", task_id=2)
    t.model.model.prompts.init_task_prompts()
    t.train()
    names = {n for n, p in t.model.named_parameters() if ".prompts.xattn_" in n}
    assert names and all(p.requires_grad for n, p in t.model.named_parameters() if n in names)
    assert not any(p.requires_grad for n, p in t.model.named_parameters() if ".decoder." in n)
    groups = t.configure_optimizers()["optimizer"].param_groups
    fast = next(g for g in groups if g["lr"] == t.args.lr)
    assert {id(p) for n, p in t.model.named_parameters() if n in names} <= {id(p) for p in fast["params"]}

    loss, _ = t.common_step(make_batch([[0, 3], [4]]), 0)
    loss.backward()
    prompts = t.model.model.prompts
    assert prompts.xattn_off[0].weight.grad.abs().sum() > 0
    assert prompts.xattn_att[0].weight.grad.abs().sum() > 0


def test_b_adds_a_fixed_number_of_parameters(tmp_path, monkeypatch):
    count = lambda model: sum(p.numel() for p in model.parameters())  # noqa: E731
    off = trainer(tmp_path, monkeypatch, "off", *PER_QUERY).model
    on = trainer(tmp_path, monkeypatch, "on", *PER_QUERY, *B).model
    assert count(on) - count(off) == XATTN_PARAMETERS


def test_b_run_starts_from_a_checkpoint_without_b(tmp_path, monkeypatch):
    old = trainer(tmp_path, monkeypatch, "old")
    path = tmp_path / "task_final.pth"
    old.save_task_final(str(path))

    args = pdp_args(sub(tmp_path, "new"), *PER_QUERY, *B)
    args._runtime_snapshot = runtime.checkpoint_runtime(str(path))
    new = make_trainer(args, task_id=2)
    new.resume(str(path))
    assert torch.equal(new.model.model.prompts.shared_p_0, old.model.model.prompts.shared_p_0)
    assert new.model.model.prompts.xattn_off[0].weight.abs().sum() == 0
    args.log_file.flush()
    log = (tmp_path / "new" / "log.txt").read_text()
    assert "xattn_off" in log and "query_tf" in log


def test_a_checkpoint_missing_other_weights_still_fails(tmp_path, monkeypatch):
    old = trainer(tmp_path, monkeypatch, "old")
    path = tmp_path / "task_final.pth"
    old.save_task_final(str(path))
    saved = torch.load(path, map_location="cpu", weights_only=False)
    del saved["model"]["class_embed.0.weight"]
    torch.save(saved, path)

    args = pdp_args(sub(tmp_path, "new"), *PER_QUERY, *B)
    args._runtime_snapshot = runtime.checkpoint_runtime(str(path))
    with pytest.raises(RuntimeError, match="class_embed"):
        make_trainer(args, task_id=2).resume(str(path))


def test_training_contract_records_prompt_cross_attn_only_when_set(tmp_path):
    assert "prompt_cross_attn" not in runtime.training_contract(pdp_args(sub(tmp_path, "off")))
    contract = runtime.training_contract(pdp_args(sub(tmp_path, "on"), *B, "--xattn_alpha_max", "0.3"))
    assert contract["prompt_cross_attn"] == 1 and contract["xattn_alpha_max"] == 0.3


def test_epoch_end_logs_the_size_of_the_correction(tmp_path, monkeypatch):
    """B5: alpha and mean |W| of every layer at the end of each epoch, to see a correction growing out of control."""
    t = trainer(tmp_path, monkeypatch, "on", *PER_QUERY, *B)
    with torch.no_grad():
        t.model.model.prompts.xattn_off[1].weight.fill_(0.5)
    stats = t.model.model.prompts.cross_attn_stats()
    assert len(stats["alpha"]) == 6 and stats["w_off"][1] == pytest.approx(0.5) and stats["w_att"][0] == 0
    t.on_train_epoch_end()
    t.args.log_file.flush()
    assert "B: epoch 0" in (tmp_path / "on" / "log.txt").read_text()
