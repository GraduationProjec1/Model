"""N3 (docs/ai/planning/2026-11-26-plan-N1-N5-hieu-chinh-task-va-loss.md §4): --group_neg_weight weighs the focal loss
of the negative logits that share the supercategory of the target class; off (1.0) it is the original loss bit for
bit."""

import json
from types import SimpleNamespace

import pytest
import runtime
import torch
from torch import nn

from engine import same_group_matrix
from models.modeling_deformable_detr import DeformableDetrLoss, sigmoid_focal_loss
from pdp_helpers import pdp_args, task_config_dict

# classes 0 and 1 share a supercategory, class 2 is alone
SAME = torch.tensor([[False, True, False], [True, False, False], [False, False, False]])


def original_focal(inputs, targets, num_boxes, alpha=0.25, gamma=2):
    """sigmoid_focal_loss before N3, the reference."""
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * loss).mean(1).sum() / num_boxes


def criterion(same=None, weight=1.0):
    return DeformableDetrLoss(matcher=None, num_classes=3, focal_alpha=0.25, losses=['labels'],
                              invalid_cls_logits=None, same_group=same, group_neg_weight=weight)


def batch():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 3)
    targets = [{'class_labels': torch.tensor([0])}]
    indices = [(torch.tensor([1]), torch.tensor([0]))]  # query 1 matched to an object of class 0
    return {'logits': logits}, targets, indices


def test_unweighted_focal_loss_is_unchanged():
    inputs, targets = torch.randn(2, 5, 3), (torch.rand(2, 5, 3) > 0.7).float()
    assert torch.equal(sigmoid_focal_loss(inputs, targets, 3), original_focal(inputs, targets, 3))
    assert torch.equal(sigmoid_focal_loss(inputs, targets, 3, weight=None), original_focal(inputs, targets, 3))


def test_weight_one_gives_the_same_loss_as_no_group():
    outputs, targets, indices = batch()
    off = criterion().loss_labels(outputs, targets, indices, 1)['loss_ce']
    assert torch.equal(criterion(SAME, 1.0).loss_labels(outputs, targets, indices, 1)['loss_ce'], off)


def test_only_the_negative_of_the_same_group_class_is_weighted():
    outputs, targets, indices = batch()
    weight = criterion(SAME, 3.0).group_weight(outputs['logits'], (torch.tensor([0]), torch.tensor([1])),
                                               torch.tensor([0]))
    expected = torch.ones(1, 4, 3)
    expected[0, 1, 1] = 3.0
    assert torch.equal(weight, expected)


def test_weighted_loss_adds_the_extra_share_of_that_element():
    outputs, targets, indices = batch()
    off = criterion().loss_labels(outputs, targets, indices, 1)['loss_ce']
    on = criterion(SAME, 3.0).loss_labels(outputs, targets, indices, 1)['loss_ce']
    onehot = torch.zeros(1, 4, 3)
    onehot[0, 1, 0] = 1
    mask = torch.zeros(1, 4, 3)
    mask[0, 1, 1] = 1
    element = sigmoid_focal_loss(outputs['logits'], onehot, 1, weight=mask) * 4  # loss_labels scales by 4 queries
    assert torch.allclose(on - off, 2.0 * element)
    assert on > off


def test_a_matrix_with_the_diagonal_set_still_leaves_the_target_at_one():
    outputs, targets, indices = batch()
    weight = criterion(torch.ones(3, 3, dtype=torch.bool), 2.0).group_weight(
        outputs['logits'], (torch.tensor([0]), torch.tensor([1])), torch.tensor([0]))
    assert weight[0, 1].tolist() == [1.0, 2.0, 2.0]


def write_config(tmp_path, supercategories):
    raw = task_config_dict(sizes=(3,), reserved=0)
    for c, name in zip(raw['tasks'][0]['classes'], supercategories, strict=True):
        c['supercategory'] = name
    path = tmp_path / 'cfg.json'
    path.write_text(json.dumps(raw))
    return path


def test_same_group_matrix_from_the_task_config(tmp_path):
    args = SimpleNamespace(group_neg_weight=2.0, task_config=str(write_config(tmp_path, ['milk', 'milk', None])),
                           n_classes=4)
    same = same_group_matrix(args)
    assert same.shape == (4, 4)
    assert same[0, 1] and same[1, 0] and not same[0, 0]
    assert not same[2].any() and not same[:, 2].any() and not same[3].any()


def test_classes_without_supercategory_are_in_no_group(tmp_path):
    args = SimpleNamespace(group_neg_weight=2.0, task_config=str(write_config(tmp_path, [None, None, None])),
                           n_classes=4)
    assert not same_group_matrix(args).any()


def test_off_by_default_and_refused_below_one(tmp_path):
    args = pdp_args(tmp_path)
    assert args.group_neg_weight == 1.0 and same_group_matrix(args) is None
    assert 'group_neg_weight' not in runtime.training_contract(args)
    args.group_neg_weight = 0.5
    with pytest.raises(ValueError, match='group_neg_weight'):
        same_group_matrix(args)


def test_recorded_in_the_training_contract_when_on(tmp_path):
    args = pdp_args(tmp_path, '--group_neg_weight', '2')
    assert runtime.training_contract(args)['group_neg_weight'] == 2.0


def common_sh_value(flag, env):
    import subprocess

    from conftest import REPO_ROOT

    script = (f'BACKBONE=r50; REPO={REPO_ROOT}; source {REPO_ROOT}/configs/exp/native/common.sh; '
              'printf "%s\\0" "${ARGS[@]}"')
    out = subprocess.run(["bash", "-ec", script], env={"PATH": "/usr/bin:/bin", **env}, capture_output=True,
                         text=True, check=True).stdout.split("\0")[:-1]
    return out[out.index(flag) + 1]


@pytest.mark.parametrize("tag, weight", [("g2", 2.0), ("g3", 3.0)])
def test_round4_n3_tags_are_r4_plus_the_group_weight(tag, weight):
    from autocheckout import update_model as um

    name = f"al_margin_k_B500_r4_{tag}"
    assert um._kind(name) == f"r4_{tag}"
    recipe = um.VARIANTS[f"r4_{tag}"]
    assert recipe == {**um.VARIANTS["r4"], "group_neg_weight": weight}
    assert um._group_neg_weight(recipe) == weight and um._group_neg_weight(um.VARIANTS["r4"]) == 1.0
    assert um.data_variant(name) == "al_margin_k_B500_r4" and um.GATES[f"r4_{tag}"] == um.GATES["r4"]


def test_common_sh_passes_the_group_weight():
    assert common_sh_value("--group_neg_weight", {}) == "1"
    assert common_sh_value("--group_neg_weight", {"GROUP_NEG_WEIGHT": "2.0"}) == "2.0"
