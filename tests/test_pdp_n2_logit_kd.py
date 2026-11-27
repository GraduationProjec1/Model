"""N2 (docs/ai/planning/2026-11-26-plan-N1-N5-hieu-chinh-task-va-loss.md §5): logit/box distillation from the teacher on
the queries it is confident about (ERD), from task 2 on; it needs the teacher every step, so no pseudo-label cache."""

import pytest
import torch

from engine import box_kd_loss, logit_kd_loss, select_kd_queries, validate_logit_kd
from pdp_helpers import make_batch, make_trainer, pdp_args, use_tiny_detr


def test_student_equal_to_teacher_is_the_minimum_with_zero_gradient():
    torch.manual_seed(0)
    teacher = torch.randn(2, 6, 5)
    student = teacher.clone().requires_grad_(True)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, :2] = mask[1, 3] = True
    loss = logit_kd_loss(student, teacher, mask, seen_old=3)
    p = teacher[mask][:, :3].sigmoid()
    entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
    assert torch.allclose(loss, entropy, atol=1e-6)
    loss.backward()
    assert torch.allclose(student.grad, torch.zeros_like(student.grad), atol=1e-7)
    assert loss < logit_kd_loss(teacher + 0.5, teacher, mask, seen_old=3)


def test_only_the_earlier_classes_are_distilled():
    teacher = torch.zeros(1, 2, 4)
    student = teacher.clone()
    student[..., 3] = 9.0
    mask = torch.tensor([[True, True]])
    assert logit_kd_loss(student, teacher, mask, seen_old=3) == logit_kd_loss(teacher, teacher, mask, seen_old=3)


def test_selected_queries_are_above_mean_plus_std_and_at_most_topk():
    logits = torch.full((1, 20, 3), -6.0)
    logits[0, [2, 5, 7, 11], 0] = torch.tensor([4.0, 3.0, 2.0, 5.0])
    logits[0, 9, 2] = 9.0  # a later class: not an earlier-class query
    assert select_kd_queries(logits, seen_old=2, topk=10)[0].nonzero().flatten().tolist() == [2, 5, 7, 11]
    assert select_kd_queries(logits, seen_old=2, topk=2)[0].nonzero().flatten().tolist() == [2, 11]


def test_nothing_selected_gives_zero_losses_that_still_backpropagate():
    student = torch.randn(1, 3, 4, requires_grad=True)
    boxes = torch.rand(1, 3, 4, requires_grad=True)
    mask = torch.zeros(1, 3, dtype=torch.bool)
    loss = logit_kd_loss(student, student.detach(), mask, 2) + box_kd_loss(boxes, boxes.detach(), mask)
    assert loss.item() == 0.0
    loss.backward()


def test_box_kd_is_the_l1_of_the_selected_boxes():
    student, teacher = torch.zeros(1, 2, 4), torch.ones(1, 2, 4)
    teacher[0, 1] = 5.0
    assert box_kd_loss(student, teacher, torch.tensor([[True, False]])) == 1.0


@pytest.mark.parametrize('flags, message', [(('--logit_kd', '-1'), 'logit_kd'),
                                            (('--logit_kd_box', '-0.5'), 'logit_kd'),
                                            (('--logit_kd_topk', '0'), 'logit_kd_topk')])
def test_invalid_values_are_refused(tmp_path, flags, message):
    with pytest.raises(ValueError, match=message):
        validate_logit_kd(pdp_args(tmp_path, *flags))


@pytest.fixture
def tiny(monkeypatch):
    use_tiny_detr(monkeypatch)


def task_trainer(tmp_path, task_id, *extra):
    torch.manual_seed(0)
    trainer = make_trainer(pdp_args(tmp_path, '--pseudo_thresh_high', '0.01', *extra), task_id=task_id)
    if task_id > 1:
        trainer.set_teacher()
    trainer.train()
    return trainer


@pytest.mark.slow
def test_cache_pseudo_1_is_refused_and_auto_turns_the_cache_off(tmp_path, tiny):
    with pytest.raises(ValueError, match='--logit_kd'):
        make_trainer(pdp_args(tmp_path, '--cache_pseudo', '1', '--logit_kd', '1'), task_id=2)
    trainer = task_trainer(tmp_path, 2, '--cache_pseudo', 'auto', '--logit_kd', '1')
    assert not trainer.use_pseudo_cache
    trainer.args.log_file.flush()
    assert 'P2: pseudo-label cache off (--logit_kd)' in (tmp_path / 'log.txt').read_text()


@pytest.mark.slow
def test_task2_adds_the_distillation_losses(tmp_path, tiny):
    trainer = task_trainer(tmp_path, 2, '--logit_kd', '1', '--logit_kd_box', '1')
    loss, loss_dict = trainer.common_step(make_batch([[3], [3, 4]]), 0)
    assert {'loss_logit_kd', 'loss_box_kd'} <= set(loss_dict)
    assert torch.isfinite(loss)


@pytest.mark.slow
def test_no_distillation_without_a_teacher_or_when_off(tmp_path, tiny):
    first = task_trainer(tmp_path, 1, '--logit_kd', '1')
    _, loss_dict = first.common_step(make_batch([[0], [1, 2]]), 0)
    assert 'loss_logit_kd' not in loss_dict
    off = task_trainer(tmp_path, 2)
    _, loss_dict = off.common_step(make_batch([[3], [3, 4]]), 0)
    assert 'loss_logit_kd' not in loss_dict and 'loss_box_kd' not in loss_dict


@pytest.mark.slow
def test_group_neg_weight_reaches_the_loss_of_a_training_step(tmp_path, tiny):
    """N3 end to end: the trainer gives the model the same-supercategory matrix and the step still runs."""
    trainer = task_trainer(tmp_path, 2, '--group_neg_weight', '2')
    assert trainer.model.same_group is not None and trainer.model.group_neg_weight == 2.0
    loss, _ = trainer.common_step(make_batch([[3], [3, 4]]), 0)
    assert torch.isfinite(loss)


def test_round4_n2_tag_is_r4_plus_the_logit_kd():
    from test_pdp_n3_group_focal import common_sh_value

    from autocheckout import update_model as um

    recipe = um.VARIANTS[um._kind("al_margin_k_B500_r4_lkd")]
    assert recipe == {**um.VARIANTS["r4"], "logit_kd": 1.0}
    assert um._logit_kd(recipe) == 1.0 and um._logit_kd(um.VARIANTS["r4"]) == 0
    assert common_sh_value("--logit_kd", {}) == "0"
    assert common_sh_value("--logit_kd", {"LOGIT_KD": "1.0"}) == "1.0"
