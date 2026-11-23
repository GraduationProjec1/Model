"""P1 (docs/design/2026-11-22-toi-uu-2-luot): the backbone features are computed once and reused by both passes.

Both paths run the same operations, so the outputs must be bitwise equal (torch.equal), not just close.
"""

import pytest
import torch

from pdp_helpers import TINY_DETR, make_batch, make_trainer, pdp_args, use_tiny_detr

pytestmark = pytest.mark.slow

OUTPUTS = ("logits", "pred_boxes", "last_hidden_state")


@pytest.fixture(scope="module")
def task1_model(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    use_tiny_detr(monkeypatch)
    args = pdp_args(tmp_path_factory.mktemp("p1"))
    model = make_trainer(args, task_id=1).model.eval()
    yield model
    monkeypatch.undo()


def padded_batch():
    """Two images, the second padded on the right and bottom, as the collate function does."""
    batch = make_batch([[0], [1, 2]])
    batch["pixel_mask"][1, -20:, :] = 0
    batch["pixel_mask"][1, :, -12:] = 0
    return batch


def assert_same_outputs(got, expected):
    for name in OUTPUTS:
        assert torch.equal(getattr(got, name), getattr(expected, name)), name


# --- Task 3: backbone features computed outside the forward ----------------------------------------------------
def test_forward_with_precomputed_backbone_matches_forward(task1_model):
    batch = padded_batch()
    pixels, mask = batch["pixel_values"], batch["pixel_mask"]
    with torch.no_grad():
        expected = task1_model(pixel_values=pixels, pixel_mask=mask, train=False)
        features = task1_model.model.extract_features(pixels, mask)
        got = task1_model(pixel_values=pixels, pixel_mask=mask, backbone_features=features, train=False)
        # the prompted pass too
        query = expected.last_hidden_state
        expected_2 = task1_model(pixel_values=pixels, pixel_mask=mask, query=query, train=False)
        got_2 = task1_model(pixel_values=pixels, pixel_mask=mask, query=query, backbone_features=features,
                            train=False)
    assert_same_outputs(got, expected)
    assert_same_outputs(got_2, expected_2)


def test_backbone_features_reusable_twice(task1_model):
    batch = padded_batch()
    pixels, mask = batch["pixel_values"], batch["pixel_mask"]
    with torch.no_grad():
        features = task1_model.model.extract_features(pixels, mask)
        first = task1_model(pixel_values=pixels, pixel_mask=mask, backbone_features=features, train=False)
        second = task1_model(pixel_values=pixels, pixel_mask=mask, backbone_features=features, train=False)
    assert_same_outputs(second, first)


def test_backbone_features_not_mutated(task1_model):
    batch = padded_batch()
    pixels, mask = batch["pixel_values"], batch["pixel_mask"]
    with torch.no_grad():
        features = task1_model.model.extract_features(pixels, mask)
        levels = len(features.features)
        positions = [p.clone() for p in features.position_embeddings]
        task1_model(pixel_values=pixels, pixel_mask=mask, backbone_features=features, train=False)
    # the extra (4th) level is built from a new list, never appended to the shared one (modeling line 1722)
    assert len(features.features) == levels
    assert len(features.position_embeddings) == len(positions)
    assert all(torch.equal(a, b) for a, b in zip(features.position_embeddings, positions, strict=True))


def test_convnext_backbone_deterministic_in_train_mode(tmp_path, monkeypatch):
    """The pipelines use ConvNeXt-V2 (no BatchNorm); the toy model above uses resnet18, so check ConvNeXt here."""
    import engine
    from models.configuration_deformable_detr import DeformableDetrConfig

    monkeypatch.setattr(engine, "DeformableDetrConfig",
                        lambda: DeformableDetrConfig(**{**TINY_DETR, "backbone": "convnextv2_atto"}))
    model = make_trainer(pdp_args(tmp_path), task_id=1).model.train()
    assert all(m.p == 0 for m in model.model.backbone.modules() if isinstance(m, torch.nn.Dropout))
    batch = padded_batch()
    with torch.no_grad():
        first = model.model.extract_features(batch["pixel_values"], batch["pixel_mask"])
        second = model.model.extract_features(batch["pixel_values"], batch["pixel_mask"])
    for (map_1, mask_1), (map_2, mask_2) in zip(first.features, second.features, strict=True):
        assert torch.equal(map_1, map_2) and torch.equal(mask_1, mask_2)
    assert all(torch.equal(a, b) for a, b in zip(first.position_embeddings, second.position_embeddings, strict=True))


# --- Task 4: two passes with the backbone and encoder computed once (eval only) -----------------------------
def old_two_passes(model, pixels, mask, *, use_prompts, local_query, first_task_id=None, second_task_id=None):
    """The two passes as the code did them before P1 (both run the whole model)."""
    first = model(pixel_values=pixels, pixel_mask=mask, train=False, task_id=first_task_id)
    if not use_prompts:
        return first
    query = first.last_hidden_state if local_query else first.last_hidden_state.mean(dim=1)
    return model(pixel_values=pixels, pixel_mask=mask, query=query, train=False, task_id=second_task_id)


@pytest.mark.parametrize("use_prompts", [False, True])
@pytest.mark.parametrize("local_query", [0, 1])
@pytest.mark.parametrize("reuse_features", [False, True])
def test_prompted_outputs_matches_two_full_passes(task1_model, use_prompts, local_query, reuse_features):
    from two_pass import prompted_outputs

    batch = padded_batch()
    pixels, mask = batch["pixel_values"], batch["pixel_mask"]
    with torch.no_grad():
        expected = old_two_passes(task1_model, pixels, mask, use_prompts=use_prompts, local_query=local_query)
        got = prompted_outputs(task1_model, pixels, mask, use_prompts=use_prompts, local_query=local_query,
                               reuse_features=reuse_features)
    assert_same_outputs(got, expected)


def test_prompted_outputs_rejects_training_mode(task1_model):
    from two_pass import prompted_outputs

    batch = padded_batch()
    task1_model.train()
    try:
        with pytest.raises(RuntimeError, match="eval"):
            prompted_outputs(task1_model, batch["pixel_values"], batch["pixel_mask"], use_prompts=True,
                             local_query=1, reuse_features=True)
    finally:
        task1_model.eval()


def test_predict_batch_contract_unchanged_and_reuse_matches(task1_model):
    from inference import predict_batch

    batch = padded_batch()
    sizes = torch.tensor([[96, 96], [76, 84]])
    # the call of checkout-system/docs/ac-contract.md: positional model, pixels, mask, sizes + 3 keywords
    old = predict_batch(task1_model, batch["pixel_values"], batch["pixel_mask"], sizes, use_prompts=True,
                        local_query=True, seen_classes=3)
    new = predict_batch(task1_model, batch["pixel_values"], batch["pixel_mask"], sizes, use_prompts=True,
                        local_query=True, seen_classes=3, reuse_features=True)
    for old_image, new_image in zip(old, new, strict=True):
        assert old_image.keys() == new_image.keys()
        assert all(torch.equal(old_image[key], new_image[key]) for key in old_image)


def test_write_predictions_reuse_matches(tmp_path, monkeypatch):
    import numpy as np

    from inference import write_predictions
    from pdp_helpers import make_toy_dataset, small_processor

    use_tiny_detr(monkeypatch)
    data = make_toy_dataset(tmp_path / "data")
    files = {}
    for reuse in ("0", "1"):
        (tmp_path / reuse).mkdir()
        args = pdp_args(tmp_path / reuse, "--reuse_features", reuse, "--test_img_dir", str(data["images"]),
                        "--batch_size", "2", "--num_workers", "0")
        torch.manual_seed(0)  # same random weights for both models
        model = make_trainer(args, task_id=1).model
        files[reuse] = tmp_path / f"pred_{reuse}.npz"
        write_predictions(model, args, ann_file=data["tasks"] / "val_full.json", out_file=files[reuse], task_id=1,
                          seen_classes=3, split="val", processor=small_processor(), device=torch.device("cpu"))
    old, new = np.load(files["0"]), np.load(files["1"])
    for name in ("image_id", "query", "label", "score", "boxes"):
        assert np.array_equal(old[name], new[name]), name


def test_teacher_outputs_reuse_matches(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    args = pdp_args(tmp_path, "--reuse_features", "1")
    trainer = make_trainer(args, task_id=2)
    trainer.set_teacher()
    batch = padded_batch()
    trainer.args.reuse_features = 0
    expected = trainer.teacher_outputs(batch["pixel_values"], batch["pixel_mask"])
    trainer.args.reuse_features = 1
    got = trainer.teacher_outputs(batch["pixel_values"], batch["pixel_mask"])
    assert_same_outputs(got, expected)


@pytest.mark.parametrize("reuse_features, expected", [(False, (2, 2, 2)), (True, (1, 1, 2))])
def test_reuse_runs_backbone_and_encoder_once(task1_model, reuse_features, expected):
    """The point of P1: with reuse the backbone and encoder run once, the decoder twice."""
    from two_pass import prompted_outputs

    detr = task1_model.model
    calls = {"backbone": 0, "encoder": 0, "decoder": 0}

    def counter(name):
        def hook(module, inputs, output):
            calls[name] += 1
        return hook

    handles = [getattr(detr, name).register_forward_hook(counter(name)) for name in calls]
    batch = padded_batch()
    try:
        with torch.no_grad():
            prompted_outputs(task1_model, batch["pixel_values"], batch["pixel_mask"], use_prompts=True,
                             local_query=1, reuse_features=reuse_features)
    finally:
        for handle in handles:
            handle.remove()
    assert (calls["backbone"], calls["encoder"], calls["decoder"]) == expected


# --- Task 5: training step shares the backbone; startup guard; no wasted loss in the first student pass ------
def sub(tmp_path, name):
    path = tmp_path / name
    path.mkdir(exist_ok=True)
    return path


def train_steps(args, task_id, *, steps=2, batches=None):
    """Losses and the gradients of every trained parameter over a few real optimizer steps (seeded)."""
    torch.manual_seed(0)  # same random weights for every call
    trainer = make_trainer(args, task_id=task_id)
    if task_id > 1 and args.pseudo != "none":
        trainer.set_teacher()
    trainer.train()
    optimizer = trainer.configure_optimizers()["optimizer"]
    batches = batches or [padded_batch(), make_batch([[2], [0, 1]], seed=1)]
    torch.manual_seed(1)  # same dropout masks
    losses, grads = [], []
    for step in range(steps):
        loss, _ = trainer.common_step(batches[step % len(batches)], step)
        loss.backward()
        losses.append(loss.detach().clone())
        grads.append({n: p.grad.clone() for n, p in trainer.named_parameters() if p.grad is not None})
        optimizer.step()
        optimizer.zero_grad()
    return losses, grads, trainer


def assert_same_training(old, new):
    old_losses, old_grads, _ = old
    new_losses, new_grads, _ = new
    assert all(torch.equal(a, b) for a, b in zip(old_losses, new_losses, strict=True))
    for before, after in zip(old_grads, new_grads, strict=True):
        assert before.keys() == after.keys()
        assert all(torch.equal(before[name], after[name]) for name in before), [
            name for name in before if not torch.equal(before[name], after[name])]


@pytest.mark.parametrize("local_query", ["0", "1"])
def test_training_unchanged_with_reuse(tmp_path, monkeypatch, local_query):
    """Task 2 with teacher + PPG, two optimizer steps: the student's input_proj moves away from the teacher's,
    which sharing only the backbone (never the encoder) must survive."""
    use_tiny_detr(monkeypatch)
    old = train_steps(pdp_args(sub(tmp_path, "old"), "--local_query", local_query, "--reuse_features", "0"), 2)
    new = train_steps(pdp_args(sub(tmp_path, "new"), "--local_query", local_query, "--reuse_features", "1"), 2)
    assert_same_training(old, new)


def test_input_proj_and_level_embed_receive_gradient(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    _, grads, _ = train_steps(pdp_args(tmp_path, "--reuse_features", "1"), 2, steps=1)
    proj = [n for n in grads[0] if "input_proj" in n]
    assert proj and all(grads[0][n].abs().sum() > 0 for n in proj)
    # level_embed only reaches the loss through the encoder's sampling_offsets / attention_weights, which the
    # untrained toy model initialises to zero (Deformable DETR init), so its gradient is 0 here even before P1.
    # It must still be connected to the graph; equal values with and without P1: test_training_unchanged_with_reuse.
    assert "model.model.level_embed" in grads[0]


def test_backbone_runs_once_per_training_step(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    calls = []
    for reuse, expected in (("0", 4), ("1", 1)):  # teacher 2 passes + student 2 passes, or once for all
        args = pdp_args(sub(tmp_path, reuse), "--reuse_features", reuse)
        trainer = make_trainer(args, task_id=2)
        trainer.set_teacher()
        counted = []
        hooks = [module.backbone.register_forward_hook(lambda *a, counted=counted: counted.append(1))
                 for module in (trainer.model.model, trainer.teacher.model)]
        trainer.train()
        trainer.common_step(padded_batch(), 0)
        for hook in hooks:
            hook.remove()
        calls.append((len(counted), expected))
    assert all(got == expected for got, expected in calls), calls


def test_reuse_noop_for_fsa(tmp_path, monkeypatch):
    """FSA: no prompts, no teacher, backbone trained (--freeze ''). Nothing to share: the flag must change nothing."""
    use_tiny_detr(monkeypatch)
    fsa = ("--use_prompts", "0", "--local_query", "0", "--pseudo", "none", "--freeze", "")
    old = train_steps(pdp_args(sub(tmp_path, "old"), *fsa, "--reuse_features", "0"), 1)
    new = train_steps(pdp_args(sub(tmp_path, "new"), *fsa, "--reuse_features", "1"), 1)
    assert_same_training(old, new)
    assert any("backbone" in name and grad.abs().sum() > 0 for name, grad in new[1][0].items())


def test_reuse_rejects_trainable_backbone_at_startup(tmp_path, monkeypatch):
    """M5 opens a backbone stage from task 2 on: refuse before task 1 trains, not hours later."""
    use_tiny_detr(monkeypatch)
    args = pdp_args(tmp_path, "--reuse_features", "1", "--tune_after_task1", "layer4,encoder")
    with pytest.raises(ValueError, match="reuse_features"):
        make_trainer(args, task_id=1)
    # the same run without the flag is accepted
    make_trainer(pdp_args(sub(tmp_path, "off"), "--tune_after_task1", "layer4,encoder"), task_id=1)


def test_student_first_pass_without_labels(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    trainer = make_trainer(pdp_args(tmp_path, "--reuse_features", "1"), task_id=1)
    trainer.train()
    seen = []
    forward = trainer.model.forward
    monkeypatch.setattr(trainer.model, "forward", lambda *a, **k: seen.append(k.get("labels")) or forward(*a, **k))
    trainer.common_step(padded_batch(), 0)
    assert seen[0] is None and seen[-1] is not None  # query pass: no loss; training pass: loss


def test_mode_improve_args_unchanged_with_reuse(tmp_path, monkeypatch):
    """EXP-FB-FT style (rounds of active learning): no teacher, only some classifier rows and private prompts."""
    import json

    use_tiny_detr(monkeypatch)
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([3, 4]))
    improve = ("--pseudo", "none", "--new_params", "class_embed,prompts.private", "--train_rows", str(rows))
    old = train_steps(pdp_args(sub(tmp_path, "old"), *improve, "--reuse_features", "0"), 2)
    new = train_steps(pdp_args(sub(tmp_path, "new"), *improve, "--reuse_features", "1"), 2)
    assert_same_training(old, new)


@pytest.mark.parametrize("no_training", [("--predict_only", "1"), ("--eval",)])
def test_reuse_guard_skipped_when_nothing_trains(tmp_path, monkeypatch, no_training):
    """Predict-only and eval runs train nothing, so an unfrozen backbone is no reason to refuse the flag."""
    use_tiny_detr(monkeypatch)
    make_trainer(pdp_args(tmp_path, "--reuse_features", "1", "--freeze", "", *no_training), task_id=1)
