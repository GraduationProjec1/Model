"""Expansion of a trained checkpoint to more class slots (docs/project/task6-relearn-plan-2026-10-30.md, Task 4)."""

import math
from types import SimpleNamespace

import pytest
import torch

import runtime
from autocheckout.taskcfg import TaskConfig
from expand import expand_checkpoint
from pdp_helpers import make_batch, make_trainer, pdp_args, task_config_dict, use_tiny_detr
from tools.make_relearn_config import relearn_config

pytestmark = pytest.mark.slow

LEARNED = 5  # classes of tasks 1-2 of config A


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """A's task-2 checkpoint (prompts of tasks 1-2 initialised, some prototype memory) and its expansion to B."""
    monkeypatch = pytest.MonkeyPatch()
    use_tiny_detr(monkeypatch)
    tmp = tmp_path_factory.mktemp("expand")
    (tmp / "a").mkdir()
    (tmp / "b").mkdir()
    cfg_a = TaskConfig.from_dict(task_config_dict(sizes=(3, 2), reserved=1))         # n_classes 7
    cfg_b = relearn_config(cfg_a, [[0, 3], [1]], reserved=1, name="toy+relearn")  # tasks 3, 2, 2, 1, 1 -> 10
    args_a, args_b = pdp_args(tmp / "a", config=cfg_a), pdp_args(tmp / "b", config=cfg_b)
    trainer_a = make_trainer(args_a, task_id=2)
    trainer_a.model.model.prompts.init_task_prompts()
    trainer_a.class_query_cache[0] = [torch.ones(256)]
    trainer_a.class_prototypes[0] = torch.ones(256)
    trainer_a.class_cache_count[0] = 1
    path_a = str(tmp / "a" / "task_final.pth")
    trainer_a.save_task_final(path_a)
    old = torch.load(path_a, map_location="cpu")
    new = expand_checkpoint(old, cfg_a, cfg_b, learned=LEARNED)
    path_b = str(tmp / "b" / "expanded.pth")
    torch.save(new, path_b)
    yield SimpleNamespace(cfg_a=cfg_a, cfg_b=cfg_b, args_a=args_a, args_b=args_b, trainer_a=trainer_a,
                          old=old, new=new, path_a=path_a, path_b=path_b)
    monkeypatch.undo()


def two_pass(model, batch):
    first = model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"], train=False)
    return model(pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"],
                 query=first.last_hidden_state, train=False)


def test_learned_rows_are_copied_and_new_rows_start_untrained(toy):
    old, new = toy.old["model"], toy.new["model"]
    w_old, w_new = old["class_embed.0.weight"], new["class_embed.0.weight"]
    assert w_new.shape[0] == 10 and torch.equal(w_new[:5], w_old[:5]) and torch.equal(w_new[-1], w_old[-1])
    assert 0.01 < w_new[5:9].std() < 0.03  # N(0, init_std = 0.02)
    b_new = new["class_embed.0.bias"]
    assert torch.equal(b_new[:5], old["class_embed.0.bias"][:5]) and torch.equal(b_new[-1], old["class_embed.0.bias"][-1])
    assert torch.allclose(b_new[5:9], torch.full((4,), -math.log(99)))
    heads = sorted({k.split(".")[1] for k in new if k.startswith("class_embed.")})
    assert len(heads) == 2  # one per decoder layer of the tiny model
    assert all(torch.equal(new[f"class_embed.{i}.weight"], w_new) for i in heads)
    assert all(torch.equal(new[f"class_embed.{i}.bias"], b_new) for i in heads)

    p_old, p = old["model.prompts.private_k_0"], new["model.prompts.private_k_0"]
    assert p_old[:5].norm(dim=1).min() > 0.99  # tasks 1-2 prompts are set in the old checkpoint
    assert p.shape[0] == toy.cfg_b.num_slots and torch.equal(p[:5], p_old[:5]) and p[5:].abs().sum() == 0

    assert len(toy.new["class_prototypes"]) == toy.cfg_b.num_slots and toy.new["class_query_cache"][8] == []
    assert toy.new["class_cache_count"][0] == 1 and torch.equal(toy.new["class_prototypes"][0], torch.ones(256))
    assert toy.new["class_cache_count"][8] == 0 and toy.new["class_prototypes"][8].abs().sum() == 0
    assert toy.new["expanded_from"] == {"old_classes": 7, "new_classes": 10, "learned": 5, "seed": 0}

    runtime.validate_runtime(toy.args_b, toy.new["runtime"])
    with pytest.raises(ValueError):
        runtime.validate_runtime(toy.args_a, toy.new["runtime"])
    assert len(toy.old["runtime"]["model_config"]["id2label"]) == 7  # the input is not changed


def test_expanded_model_predicts_the_learned_classes_exactly_as_before(toy):
    trainer_b = make_trainer(toy.args_b, task_id=3)
    trainer_b.resume(toy.path_b)
    batch = make_batch([[0], [1, 2]])
    outputs = []
    for model in (toy.trainer_a.model, trainer_b.model):
        model.eval()
        model.model.prompts.set_task_id(1)
        with torch.no_grad():
            outputs.append(two_pass(model, batch))
    a, b = outputs
    assert torch.equal(b.last_hidden_state, a.last_hidden_state) and torch.equal(b.pred_boxes, a.pred_boxes)
    # same rows, but a linear layer of 10 outputs is not computed bit for bit like one of 7 (BLAS kernels)
    torch.testing.assert_close(b.logits[..., :5], a.logits[..., :5], rtol=0, atol=1e-5)
    torch.testing.assert_close(b.logits[..., -1], a.logits[..., -1], rtol=0, atol=1e-5)


def test_task_after_expansion_trains_on_alias_labels(toy):
    trainer = make_trainer(toy.args_b, task_id=3)
    trainer.resume(toy.path_b)
    trainer.set_teacher()
    trainer.model.model.prompts.init_task_prompts()
    trainer.train()
    loss, _ = trainer.common_step(make_batch([[5, 6], [6]]), 0)
    assert torch.isfinite(loss)
    loss.backward()
    grad = trainer.model.model.prompts.private_p_0.grad
    assert grad[:5].abs().sum() == 0 and grad[5:7].abs().sum() > 0


def test_unexpanded_checkpoint_does_not_load_into_the_larger_model(toy):
    trainer = make_trainer(toy.args_b, task_id=3)
    with pytest.raises(ValueError, match="n_classes"):  # runtime check, before load_state_dict
        trainer.resume(toy.path_a)


@pytest.mark.parametrize("break_it, message", [("rows", "rows"), ("trained_later_slot", "not zero"),
                                               ("changed_learned_class", "differs")])
def test_expansion_refuses_a_checkpoint_or_config_that_does_not_match(toy, break_it, message):
    old, cfg_a, cfg_b = dict(toy.old), toy.cfg_a, toy.cfg_b
    if break_it == "rows":  # the checkpoint was not trained with this config
        cfg_a = TaskConfig.from_dict(task_config_dict(sizes=(3, 2), reserved=2))
    elif break_it == "trained_later_slot":  # the slot after the learned ones has a prompt
        old["model"] = dict(old["model"])
        old["model"]["model.prompts.private_k_0"] = old["model"]["model.prompts.private_k_0"].clone()
        old["model"]["model.prompts.private_k_0"][5] = 1.0
    else:
        data = cfg_b.to_dict()
        data["tasks"][0]["classes"][2]["name"] = "sku_x"
        cfg_b = TaskConfig.from_dict(data)
    with pytest.raises(ValueError, match=message):
        expand_checkpoint(old, cfg_a, cfg_b, learned=LEARNED)


# --- repartition: same slots, other task boundaries after the learned classes (chia 12+13 SKU) ---------------------
@pytest.fixture(scope="module")
def split(tmp_path_factory):
    """Task 1 of config A (2 + 2 classes, 1 reserved) and its repartition to B (2 + 1 + 1, 1 reserved)."""
    from expand import repartition_checkpoint

    monkeypatch = pytest.MonkeyPatch()
    use_tiny_detr(monkeypatch)
    tmp = tmp_path_factory.mktemp("repartition")
    (tmp / "a").mkdir()
    (tmp / "b").mkdir()
    cfg_a = TaskConfig.from_dict(task_config_dict(sizes=(2, 2), reserved=1, name="toy-2x2"))
    cfg_b = TaskConfig.from_dict(task_config_dict(sizes=(2, 1, 1), reserved=1, name="toy-2+1+1"))
    args_a, args_b = pdp_args(tmp / "a", config=cfg_a), pdp_args(tmp / "b", config=cfg_b)
    trainer_a = make_trainer(args_a, task_id=1)
    trainer_a.model.model.prompts.init_task_prompts()
    trainer_a.class_query_cache[0] = [torch.ones(256)]
    trainer_a.class_prototypes[0] = torch.ones(256)
    trainer_a.class_cache_count[0] = 1
    path_a = str(tmp / "a" / "task_final.pth")
    trainer_a.save_task_final(path_a)
    old = torch.load(path_a, map_location="cpu")
    new = repartition_checkpoint(old, cfg_a, cfg_b, learned=2)
    path_b = str(tmp / "b" / "task_1.pth")
    torch.save(new, path_b)
    yield SimpleNamespace(cfg_a=cfg_a, cfg_b=cfg_b, args_a=args_a, args_b=args_b, trainer_a=trainer_a, old=old,
                          new=new, path_b=path_b)
    monkeypatch.undo()


def test_repartition_changes_only_the_task_boundaries(split):
    old, new = split.old, split.new
    assert new["runtime"]["model_config"]["task_num_classes"] == [2, 1, 1, 1]
    assert old["runtime"]["model_config"]["task_num_classes"] == [2, 2, 1]  # the input is not changed
    assert set(new["model"]) == set(old["model"])
    assert all(torch.equal(new["model"][k], old["model"][k]) for k in old["model"])
    for name in ("class_query_cache", "class_prototypes", "class_cache_count"):
        assert new[name] is old[name]
    assert new["task_id"] == old["task_id"]
    assert new["repartitioned_from"]["old_task_num_classes"] == [2, 2, 1]
    runtime.validate_runtime(split.args_b, new["runtime"])
    with pytest.raises(ValueError, match="task_num_classes"):
        runtime.validate_runtime(split.args_a, new["runtime"])
    with pytest.raises(ValueError, match="task_num_classes"):
        runtime.validate_runtime(split.args_b, old["runtime"])  # the unrepartitioned checkpoint is refused


def test_repartitioned_model_predicts_task_1_exactly_as_before(split):
    trainer_b = make_trainer(split.args_b, task_id=1)
    trainer_b.resume(split.path_b)
    batch = make_batch([[0], [1, 0]])
    outputs = []
    for model in (split.trainer_a.model, trainer_b.model):
        model.eval()
        model.model.prompts.set_task_id(0)
        with torch.no_grad():
            outputs.append(two_pass(model, batch))
    a, b = outputs
    assert torch.equal(b.logits, a.logits) and torch.equal(b.pred_boxes, a.pred_boxes)


def test_task_2_of_the_new_config_trains_from_the_repartitioned_task_1(split):
    trainer = make_trainer(split.args_b, task_id=2)
    trainer.resume(split.path_b)
    trainer.set_teacher()
    trainer.model.model.prompts.init_task_prompts()
    trainer.train()
    loss, _ = trainer.common_step(make_batch([[2], [2, 0]]), 0)
    assert torch.isfinite(loss)
    loss.backward()
    grad = trainer.model.model.prompts.private_p_0.grad
    assert grad[:2].abs().sum() == 0 and grad[2].abs().sum() > 0 and grad[3:].abs().sum() == 0


@pytest.mark.parametrize("break_it, message", [
    ("changed_learned_class", "differs"), ("other_slots", "slots"), ("not_a_boundary", "does not start"),
    ("trained_later_slot", "not zero"), ("later_prototype", "prototype memory"), ("wrong_old_config", "is not"),
])
def test_repartition_refuses_what_it_cannot_keep_exactly(split, break_it, message):
    from expand import repartition_checkpoint

    old, cfg_a, cfg_b, learned = dict(split.old), split.cfg_a, split.cfg_b, 2
    if break_it == "changed_learned_class":
        data = cfg_b.to_dict()
        data["tasks"][0]["classes"][1]["name"] = "sku_x"
        cfg_b = TaskConfig.from_dict(data)
    elif break_it == "other_slots":
        cfg_b = TaskConfig.from_dict(task_config_dict(sizes=(2, 1, 1), reserved=2))
    elif break_it == "not_a_boundary":
        learned = 1
        cfg_b = cfg_a
    elif break_it == "trained_later_slot":
        old["model"] = dict(old["model"])
        old["model"]["model.prompts.private_k_0"] = old["model"]["model.prompts.private_k_0"].clone()
        old["model"]["model.prompts.private_k_0"][2] = 1.0
    elif break_it == "later_prototype":
        old["class_cache_count"] = {**old["class_cache_count"], 3: 1}
    else:
        cfg_a = TaskConfig.from_dict(task_config_dict(sizes=(2, 1, 1), reserved=1))
    with pytest.raises(ValueError, match=message):
        repartition_checkpoint(old, cfg_a, cfg_b, learned=learned)


def test_cli_repartition_writes_the_checkpoint(split, tmp_path):
    from expand import main

    paths = {}
    for name, cfg in (("a", split.cfg_a), ("b", split.cfg_b)):
        paths[name] = tmp_path / f"{name}.json"
        cfg.save(paths[name])
    torch.save(split.old, tmp_path / "in.pth")
    main(["--repartition", "--checkpoint", str(tmp_path / "in.pth"), "--old-config", str(paths["a"]),
          "--new-config", str(paths["b"]), "--learned", "2", "--out", str(tmp_path / "out.pth")])
    out = torch.load(tmp_path / "out.pth", map_location="cpu")
    assert out["runtime"]["model_config"]["task_num_classes"] == [2, 1, 1, 1]
    assert all(torch.equal(out["model"][k], split.old["model"][k]) for k in split.old["model"])
