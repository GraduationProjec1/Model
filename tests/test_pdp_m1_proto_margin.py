"""M1 (docs/ai/planning/2026-11-26-plan-M1-M6-kiem-prototype.md §3): prototype margin loss.

The decoder output of every matched query must be closer (cosine) to the anchor of its class than to the anchors of
the most similar other classes, by a margin. Anchors are fixed for the run (snapshot at fit start, detached), so
the loss moves the features (through the prompts) and never the anchors.
"""

import subprocess

import pytest
import torch

import runtime
from autocheckout import update_model as um
from conftest import REPO_ROOT
from pdp_helpers import make_batch, make_trainer, pdp_args, use_tiny_detr
from proto_margin import anchor_matrix, margin_loss


def unit(*rows):
    return torch.nn.functional.normalize(torch.tensor(rows, dtype=torch.float), dim=1)


# ---------------------------------------------------------------- pure functions

def test_anchor_matrix_keeps_classes_with_a_prototype_and_maps_their_rows():
    vectors = {0: torch.tensor([2.0, 0.0]), 1: torch.zeros(2), 2: torch.tensor([0.0, 3.0])}
    anchors, slot = anchor_matrix(vectors, valid=[0, 2], seen=3)
    assert torch.allclose(anchors, unit([1, 0], [0, 1]))
    assert slot.tolist() == [0, -1, 1]
    assert not anchors.requires_grad


def test_loss_is_smaller_when_the_feature_sits_on_its_own_anchor():
    anchors, slot = anchor_matrix({0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])}, [0, 1], seen=2)
    labels = torch.tensor([0])
    near = margin_loss(torch.tensor([[1.0, 0.1]]), labels, anchors, slot, margin=0.1, temp=0.1, topk=5)
    far = margin_loss(torch.tensor([[0.1, 1.0]]), labels, anchors, slot, margin=0.1, temp=0.1, topk=5)
    assert near < far


def test_a_larger_margin_gives_a_larger_loss():
    anchors, slot = anchor_matrix({0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.6, 0.8])}, [0, 1], seen=2)
    features, labels = torch.tensor([[1.0, 0.2]]), torch.tensor([0])
    small = margin_loss(features, labels, anchors, slot, margin=0.0, temp=0.1, topk=5)
    large = margin_loss(features, labels, anchors, slot, margin=0.3, temp=0.1, topk=5)
    assert large > small


def test_queries_of_classes_without_anchor_are_skipped_and_never_give_nan():
    anchors, slot = anchor_matrix({0: torch.tensor([1.0, 0.0]), 1: torch.zeros(2)}, [0], seen=2)
    features = torch.randn(2, 2, requires_grad=True)
    loss = margin_loss(features, torch.tensor([1, 7]), anchors, slot, margin=0.1, temp=0.1, topk=5)
    assert loss == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.equal(features.grad, torch.zeros(2, 2))


def test_only_the_topk_hardest_negatives_count():
    anchors, slot = anchor_matrix({0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.8, 0.6]),
                                   2: torch.tensor([0.6, -0.8])}, [0, 1, 2], seen=3)
    features, labels = torch.tensor([[1.0, 0.0]]), torch.tensor([0])
    hardest = margin_loss(features, labels, anchors, slot, margin=0.1, temp=0.1, topk=1)
    both = margin_loss(features, labels, anchors, slot, margin=0.1, temp=0.1, topk=2)
    assert hardest < both


# ---------------------------------------------------------------- trainer

pytest_slow = pytest.mark.slow
M1 = ("--proto_margin_lambda", "0.1")


def trainer(tmp_path, monkeypatch, *extra):
    use_tiny_detr(monkeypatch)
    t = make_trainer(pdp_args(tmp_path, "--ddl_lambda", "0", *extra), task_id=2)
    t.model.model.prompts.init_task_prompts()
    generator = torch.Generator().manual_seed(0)
    for c in range(3):
        t.class_prototypes[c] = torch.randn(t.model.config.d_model, generator=generator)
        t.class_cache_count[c] = 4
    t.train()
    return t


@pytest_slow
def test_off_by_default_adds_no_loss(tmp_path, monkeypatch):
    t = trainer(tmp_path, monkeypatch)
    assert t.args.proto_margin_lambda == 0
    t.on_fit_start()
    _, loss_dict = t.common_step(make_batch([[0, 3], [4]]), 0)
    assert "loss_proto" not in loss_dict


@pytest_slow
def test_the_loss_reaches_the_prompts_but_not_the_classifier(tmp_path, monkeypatch):
    t = trainer(tmp_path, monkeypatch, *M1)
    t.on_fit_start()
    _, loss_dict = t.common_step(make_batch([[0, 1, 2], [1]]), 0)
    proto = loss_dict["loss_proto"]
    assert torch.isfinite(proto) and proto > 0
    proto.backward()
    grads = [p.grad for n, p in t.model.model.prompts.named_parameters() if n.startswith("private_")]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in t.model.class_embed.parameters())


@pytest_slow
def test_anchors_stay_fixed_when_the_prototypes_change(tmp_path, monkeypatch):
    t = trainer(tmp_path, monkeypatch, *M1)
    t.on_fit_start()
    before = t._margin_anchors[0].clone()
    t.class_prototypes[0] = torch.ones(t.model.config.d_model)
    t.common_step(make_batch([[0]]), 0)
    assert torch.equal(t._margin_anchors[0], before)


@pytest_slow
def test_class_embed_anchors_use_the_classifier_rows(tmp_path, monkeypatch):
    t = trainer(tmp_path, monkeypatch, *M1, "--proto_anchor", "class_embed")
    t.on_fit_start()
    anchors, slot = t._margin_anchors
    rows = t.model.class_embed[-1].weight[:t.seen_classes]
    assert torch.allclose(anchors, torch.nn.functional.normalize(rows.detach(), dim=1))
    assert slot.tolist() == list(range(t.seen_classes))


@pytest_slow
def test_a_run_without_any_prototype_fails_fast(tmp_path, monkeypatch):
    t = trainer(tmp_path, monkeypatch, *M1)
    for c in range(t.seen_classes):
        t.class_cache_count[c] = 0
    with pytest.raises(ValueError, match="proto_margin"):
        t.on_fit_start()


# ---------------------------------------------------------------- wiring

def test_training_contract_records_m1_only_when_on(tmp_path):
    off = runtime.training_contract(pdp_args(tmp_path))
    on = runtime.training_contract(pdp_args(tmp_path, *M1))
    assert "proto_margin_lambda" not in off
    assert on["proto_margin_lambda"] == 0.1


def test_round4_m1_tag_is_a_plus_m1():
    name = "al_margin_k_B500_r4_pqm"
    assert um._kind(name) == "r4_pqm"
    recipe = um.VARIANTS["r4_pqm"]
    assert recipe == {**um.VARIANTS["r4"], "per_query_prompt": 1, "proto_margin_lambda": 0.1}
    assert um._proto_margin_lambda(recipe) == 0.1 and um._proto_margin_lambda(um.VARIANTS["r4"]) == 0
    assert um.data_variant(name) == "al_margin_k_B500_r4"
    assert um.GATES["r4_pqm"] == um.GATES["r4"]


def test_common_sh_passes_the_m1_weight():
    script = (f'BACKBONE=r50; REPO={REPO_ROOT}; source {REPO_ROOT}/configs/exp/native/common.sh; '
              'printf "%s\\0" "${ARGS[@]}"')

    def weight(env):
        out = subprocess.run(["bash", "-ec", script], env=env, capture_output=True, text=True,
                             check=True).stdout.split("\0")[:-1]
        return out[out.index("--proto_margin_lambda") + 1]

    assert weight({"PATH": "/usr/bin:/bin"}) == "0"
    assert weight({"PATH": "/usr/bin:/bin", "PROTO_MARGIN_LAMBDA": "0.1"}) == "0.1"
