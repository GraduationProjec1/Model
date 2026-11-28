"""Lightweight zero-initialized LoRA adapters for Deformable DETR decoder projections.

``LoRALinear`` subclasses ``nn.Linear`` directly so ``weight`` and ``bias`` keep the exact same
``state_dict`` keys as a standard ``nn.Linear``. Only ``lora_A`` and ``lora_B`` are added, and
``lora_B`` is initialized to zero so the adapted layer produces bitwise-identical outputs to the
original layer at initialization.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DECODER_LORA_TARGETS = (
    ("self_attn", ("q_proj", "k_proj", "v_proj", "out_proj")),
    ("encoder_attn", ("value_proj", "output_proj")),
)


class LoRALinear(nn.Linear):
    """``nn.Linear`` with a parallel low-rank adapter ``(alpha / rank) * B(A(x))``, ``B = 0`` at init."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 16.0) -> None:
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None)
        with torch.no_grad():
            self.weight.copy_(base.weight)
            if base.bias is not None and self.bias is not None:
                self.bias.copy_(base.bias)
        self.lora_rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_scale = self.lora_alpha / self.lora_rank
        self.lora_A = nn.Parameter(torch.empty(self.lora_rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.lora_rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(input, self.weight, self.bias)
        delta = F.linear(F.linear(input, self.lora_A), self.lora_B)
        return base_out + delta * self.lora_scale


def inject_decoder_lora(decoder: nn.Module, rank: int, alpha: float = 16.0) -> int:
    """Replace target linear projections in every decoder layer with ``LoRALinear``. Returns count replaced."""
    if rank <= 0:
        return 0
    replaced = 0
    for layer in getattr(decoder, "layers", ()):
        for sub_name, proj_names in DECODER_LORA_TARGETS:
            sub = getattr(layer, sub_name, None)
            if sub is None:
                continue
            for proj_name in proj_names:
                mod = getattr(sub, proj_name, None)
                if isinstance(mod, nn.Linear) and not isinstance(mod, LoRALinear):
                    setattr(sub, proj_name, LoRALinear(mod, rank=rank, alpha=alpha))
                    replaced += 1
    return replaced
