"""N1 (docs/ai/planning/2026-11-26-plan-N1-N5-hieu-chinh-task-va-loss.md §3): per-task logit calibration after
training (BiC / Weight Aligning), no retraining.

``logit'_c = a_t(c) * logit_c + b_t(c)`` for the classes ``c`` of each task ``t``. The last task is the reference
(a = 1, b = 0), so 5 tasks leave 8 free parameters (4 with ``bias_only``). The parameters are fitted with L-BFGS on
the queries matched to a GT object: softmax cross-entropy over the learned classes (the argmax between classes is
what a cross-task swap gets wrong), plus an L2 pull towards (a = 1, b = 0). Applying it never changes its input.

Fit on Val-A-calib only: ``assert_disjoint`` refuses fit images that are scored (Val-B).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

MODES = ("scale_bias", "bias_only")
DEFAULT_L2 = 1e-2
LBFGS_ITERATIONS = 200


@dataclass(frozen=True)
class TaskCalib:
    """One (scale, shift) per group of labels (the classes of one task)."""

    scale: tuple[float, ...]
    shift: tuple[float, ...]
    groups: tuple[tuple[int, ...], ...]
    mode: str = "scale_bias"

    def __post_init__(self) -> None:
        if not len(self.scale) == len(self.shift) == len(self.groups):
            raise ValueError("task calibration: one scale and one shift per group")
        if any(not np.isfinite(a) or a <= 0 for a in self.scale) or not np.all(np.isfinite(self.shift)):
            raise ValueError(f"task calibration: scales must be finite and > 0, got {self.scale}")
        labels = [label for group in self.groups for label in group]
        if len(set(labels)) != len(labels):
            raise ValueError("task calibration: a label is in two groups")
        if self.mode not in MODES:
            raise ValueError(f"task calibration: mode must be one of {MODES}, got {self.mode!r}")

    @classmethod
    def identity(cls, groups: Sequence[Sequence[int]], mode: str = "scale_bias") -> TaskCalib:
        n = len(groups)
        return cls((1.0,) * n, (0.0,) * n, tuple(tuple(int(x) for x in g) for g in groups), mode)

    def class_vectors(self, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-class (scale, shift) over ``num_classes`` logits; classes in no group keep (1, 0)."""
        scale, shift = np.ones(num_classes, dtype=np.float32), np.zeros(num_classes, dtype=np.float32)
        for group, a, b in zip(self.groups, self.scale, self.shift, strict=True):
            if group and max(group) >= num_classes:
                raise ValueError(f"task calibration: label {max(group)} but only {num_classes} logits")
            scale[list(group)], shift[list(group)] = a, b
        return scale, shift

    def apply(self, logits):
        """New array/tensor of calibrated logits (last dimension = classes); the input is not modified."""
        scale, shift = self.class_vectors(logits.shape[-1])
        if isinstance(logits, torch.Tensor):
            return (logits * torch.as_tensor(scale, dtype=logits.dtype, device=logits.device)
                    + torch.as_tensor(shift, dtype=logits.dtype, device=logits.device))
        return logits * scale.astype(logits.dtype) + shift.astype(logits.dtype)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "groups": [list(g) for g in self.groups]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskCalib:
        return cls(tuple(float(a) for a in data["scale"]), tuple(float(b) for b in data["shift"]),
                   tuple(tuple(int(x) for x in g) for g in data["groups"]), data.get("mode", "scale_bias"))

    def save(self, path: str | os.PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | os.PathLike) -> TaskCalib:
        return cls.from_dict(json.loads(Path(path).read_text()))


def assert_disjoint(fit_ids: Iterable[int], scored_ids: Iterable[int]) -> None:
    """The images a calibration is fitted on must not be scored (no Val-B image in the fit)."""
    shared = set(int(i) for i in fit_ids) & set(int(i) for i in scored_ids)
    if shared:
        raise ValueError(f"task calibration: {len(shared)} fit images are scored images (e.g. {min(shared)})")


def _group_index(groups: Sequence[Sequence[int]], num_classes: int) -> np.ndarray:
    index = np.full(num_classes, -1, dtype=np.int64)
    for g, group in enumerate(groups):
        index[list(group)] = g
    if (index < 0).any():
        raise ValueError(f"task calibration: labels {np.flatnonzero(index < 0).tolist()[:5]} are in no group")
    return index


def fit_task_calib(logits: np.ndarray, labels: np.ndarray, groups: Sequence[Sequence[int]], *,
                   mode: str = "scale_bias", l2: float = DEFAULT_L2) -> TaskCalib:
    """Fit on matched queries: ``logits`` [M, C] over the learned classes, ``labels`` [M] their GT class.
    The last group is the reference (scale 1, shift 0)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if l2 < 0:
        raise ValueError("l2 must be >= 0")
    if len(logits) == 0 or len(logits) != len(labels):
        raise ValueError(f"task calibration: {len(logits)} logit rows for {len(labels)} labels")
    groups = [tuple(int(x) for x in g) for g in groups]
    group = torch.from_numpy(_group_index(groups, logits.shape[1]))
    x = torch.from_numpy(np.asarray(logits, dtype=np.float64))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    free = len(groups) - 1
    log_scale = torch.zeros(free, dtype=torch.float64, requires_grad=mode == "scale_bias")
    shift = torch.zeros(free, dtype=torch.float64, requires_grad=True)
    zero = torch.zeros(1, dtype=torch.float64)
    params = [shift, log_scale] if mode == "scale_bias" else [shift]
    optimizer = torch.optim.LBFGS(params, lr=1.0, max_iter=LBFGS_ITERATIONS, line_search_fn="strong_wolfe")

    def objective():
        optimizer.zero_grad()
        a = torch.cat([log_scale, zero]).exp()[group]
        b = torch.cat([shift, zero])[group]
        loss = torch.nn.functional.cross_entropy(x * a + b, y) + l2 * (log_scale.square().sum()
                                                                       + shift.square().sum())
        loss.backward()
        return loss

    optimizer.step(objective)
    scale = tuple(float(v) for v in torch.cat([log_scale, zero]).exp().detach())
    return TaskCalib(scale, tuple(float(v) for v in torch.cat([shift, zero]).detach()), tuple(groups), mode)


def cross_entropy(calib: TaskCalib, logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean softmax cross-entropy of the calibrated logits (reported before/after the fit)."""
    calibrated = torch.from_numpy(calib.apply(np.asarray(logits, dtype=np.float64)))
    return float(torch.nn.functional.cross_entropy(calibrated, torch.from_numpy(np.asarray(labels, dtype=np.int64))))
