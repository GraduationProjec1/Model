"""Unit and integration tests for:
- r4_all_g3 (200-row train_rows='all' + group_neg_weight=3.0)
- --allow_resolution_change 1 (r4_res800 fine-tune at 800px from 640px checkpoint)
- Decoder LoRA rank-8 (pdp/models/lora.py) + N2 Logit KD (r4_lora)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.lora import LoRALinear, inject_decoder_lora
from pdp_helpers import make_batch, make_trainer, pdp_args, use_tiny_detr
import runtime


def test_lora_linear_zero_init_is_bitwise_identical_and_trains():
    torch.manual_seed(0)
    base = nn.Linear(16, 16)
    x = torch.randn(2, 5, 16)
    y_base = base(x)

    lora = LoRALinear(base, rank=4, alpha=8.0)
    assert torch.equal(lora(x), y_base)

    # Freeze base weights; only LoRA parameters train and receive non-zero gradients
    lora.weight.requires_grad = False
    lora.bias.requires_grad = False
    loss = lora(x).pow(2).mean()
    loss.backward()
    assert lora.lora_B.grad is not None and lora.lora_B.grad.abs().sum().item() > 0
    assert lora.weight.grad is None


def test_allow_resolution_change_permits_training_at_800px(tmp_path):
    saved = {
        "model_config": {
            "backbone": "convnextv2_tiny",
            "id2label": {str(i): f"c{i}" for i in range(5)},
            "use_prompts": 1,
            "local_query": 1,
            "per_query_prompt": 0,
            "prompt_cross_attn": 0,
            "query_proposals": 0,
            "decoder_lora_rank": 0,
            "num_prompts": 4,
            "prompt_len": 2,
            "use_shared_pool": 1,
            "use_private_pool": 1,
            "task_num_classes": [3, 2],
        },
        "processor_config": {"size": {"shortest_edge": 640, "longest_edge": 640}},
    }
    args_strict = pdp_args(tmp_path, "--image_size", "800", "--max_image_size", "800")
    args_strict.backbone = "convnextv2_tiny"
    args_strict.n_classes = 5
    args_strict.num_prompts = 4
    args_strict.prompt_len = 2
    args_strict.task_num_classes = [3, 2]
    with pytest.raises(ValueError, match="resolution conflicts"):
        runtime.validate_runtime(args_strict, saved)

    args_ok = pdp_args(
        tmp_path, "--image_size", "800", "--max_image_size", "800", "--allow_resolution_change", "1"
    )
    args_ok.backbone = "convnextv2_tiny"
    args_ok.n_classes = 5
    args_ok.num_prompts = 4
    args_ok.prompt_len = 2
    args_ok.task_num_classes = [3, 2]
    args_ok._runtime_snapshot = saved
    runtime.validate_runtime(args_ok, saved)
    proc = runtime.processor(args_ok, factory=None)
    assert proc.size == {"shortest_edge": 800, "longest_edge": 800}


@pytest.fixture
def tiny(monkeypatch):
    use_tiny_detr(monkeypatch)


@pytest.mark.slow
def test_decoder_lora_migrates_non_lora_checkpoint_stays_unfrozen_and_runs_with_lkd(tmp_path, tiny):
    # 1. Save a non-LoRA task-1 checkpoint
    torch.manual_seed(0)
    base_trainer = make_trainer(pdp_args(tmp_path, "--pseudo", "none"), task_id=1)
    ckpt_path = tmp_path / "base_task1.pth"
    base_trainer.save_task_final(str(ckpt_path))

    # 2. Load into a task-2 trainer with decoder LoRA rank=4 + N2 logit_kd=1.0 + --freeze backbone,encoder,decoder
    args = pdp_args(
        tmp_path,
        "--pseudo", "none",
        "--freeze", "backbone,encoder,decoder",
        "--new_params", "class_embed,prompts.private,lora_A,lora_B",
        "--decoder_lora_rank", "4",
        "--decoder_lora_alpha", "8.0",
        "--logit_kd", "1.0",
    )
    args._runtime_snapshot = runtime.checkpoint_runtime(str(ckpt_path))
    trainer = make_trainer(args, task_id=2)
    trainer.resume(str(ckpt_path))
    trainer.set_teacher()
    trainer.train()

    # Verify decoder has LoRA layers, LoRA params are trainable, and base decoder weights are frozen
    lora_params = [(n, p) for n, p in trainer.model.named_parameters() if "lora_A" in n or "lora_B" in n]
    assert len(lora_params) > 0
    assert all(p.requires_grad for _, p in lora_params)
    frozen_dec = [
        (n, p) for n, p in trainer.model.named_parameters()
        if ".decoder." in n and "lora_A" not in n and "lora_B" not in n
    ]
    assert len(frozen_dec) > 0 and all(not p.requires_grad for _, p in frozen_dec)

    # 3. Forward + backward step runs cleanly with LKD and produces grad on lora_B
    loss, loss_dict = trainer.common_step(make_batch([[3], [3, 4]]), 0)
    assert "loss_logit_kd" in loss_dict and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for n, p in lora_params if "lora_B" in n)


def test_r4_new_variant_recipes():
    from autocheckout import update_model as um

    r_all_g3 = um.VARIANTS[um._kind("al_margin_k_B500_r4_all_g3")]
    assert r_all_g3["train_rows"] == "all" and r_all_g3["group_neg_weight"] == 3.0

    r_res800 = um.VARIANTS[um._kind("al_margin_k_B500_r4_res800")]
    assert r_res800["resolution"] == 800 and r_res800["allow_resolution_change"] == 1 and r_res800["batch_size"] == 4

    r_lora = um.VARIANTS[um._kind("al_margin_k_B500_r4_lora")]
    assert r_lora["decoder_lora_rank"] == 8 and r_lora["logit_kd"] == 1.0
    assert "lora_A" in r_lora["new_params_extra"]
