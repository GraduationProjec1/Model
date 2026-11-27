"""N0 (docs/ai/planning/2026-11-26-plan-N1-N5-hieu-chinh-task-va-loss.md §2): raw class logits of a model, saved
next to its ``pred_<split>.npz`` as ``logits_<split>.npz``.

``pred_<split>.npz`` keeps the top-100 (query, class) pairs of each image after the sigmoid, so a per-task logit
calibration (N1) cannot be applied to it. The dump keeps, per image, the ``n`` queries with the highest score over
the learned classes, with their logits over every learned class (float16) and their boxes (``[x1, y1, x2, y2]`` in
pixels, like the predictions). Every query of a top-``n`` (query, class) pair is among the ``n`` queries with the
highest best score, so ``LogitDump.predictions()`` without calibration ranks the same pairs as the model did (up to
the float16 rounding of the logits).

``rank_topk`` is the ranking of ``pdp/inference.predict_batch``, shared so both rank exactly alike.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autocheckout.io import default_file_mode
from autocheckout.predictions import Predictions

TOPK_PER_IMAGE = 100
_ARRAYS = ("image_id", "query", "logits", "boxes")


def rank_topk(prob: torch.Tensor, boxes: torch.Tensor, topk: int = TOPK_PER_IMAGE) -> list[dict[str, torch.Tensor]]:
    """Top-``topk`` (query, class) pairs of each image. ``prob``: [B, Q, C] scores; ``boxes``: [B, Q, 4].
    Returns one dict per image: scores, labels, queries (indices into Q), boxes."""
    batch_size, num_queries, num_classes = prob.shape
    scores, index = prob.reshape(batch_size, -1).topk(min(topk, num_queries * num_classes), dim=1)
    queries = torch.div(index, num_classes, rounding_mode="floor")
    labels = index % num_classes
    picked = boxes.gather(1, queries.unsqueeze(-1).expand(-1, -1, 4))
    return [
        {"scores": s, "labels": label, "queries": q, "boxes": b}
        for s, label, q, b in zip(scores, labels, queries, picked, strict=True)
    ]


def top_queries(logits: torch.Tensor, n: int = TOPK_PER_IMAGE) -> torch.Tensor:
    """[B, n] indices of the queries with the highest best logit (= highest best sigmoid score), best first."""
    return logits.max(dim=-1).values.topk(min(n, logits.shape[1]), dim=1).indices


@dataclass(frozen=True)
class LogitDump:
    """``image_id`` [I]; ``query`` [I, N] (query index in the model); ``logits`` [I, N, C] over the learned
    classes; ``boxes`` [I, N, 4] in pixels."""

    image_id: np.ndarray
    query: np.ndarray
    logits: np.ndarray
    boxes: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_images = len(self.image_id)
        if self.query.shape[0] != n_images or self.logits.shape[:2] != self.query.shape:
            raise ValueError(f"logit dump: query {self.query.shape} and logits {self.logits.shape} do not match "
                             f"{n_images} images")
        if self.boxes.shape != (*self.query.shape, 4):
            raise ValueError(f"logit dump: boxes {self.boxes.shape}, expected {(*self.query.shape, 4)}")

    @property
    def num_classes(self) -> int:
        return self.logits.shape[-1]

    def predictions(self, calib=None, topk: int = TOPK_PER_IMAGE, producer: str = "logit_dump") -> Predictions:
        """Predictions ranked like the model, from the (optionally calibrated: ``calib.apply``) logits."""
        logits = self.logits.astype(np.float32)
        if calib is not None:
            logits = calib.apply(logits)
        ranked = rank_topk(torch.from_numpy(logits).sigmoid(), torch.from_numpy(self.boxes.astype(np.float32)), topk)
        columns: dict[str, list[np.ndarray]] = {"image_id": [], "query": [], "label": [], "score": [], "boxes": []}
        for image_id, queries, result in zip(self.image_id, self.query, ranked, strict=True):
            rows = result["queries"].numpy()
            columns["image_id"].append(np.full(len(rows), int(image_id)))
            columns["query"].append(queries[rows])
            columns["label"].append(result["labels"].numpy())
            columns["score"].append(result["scores"].numpy())
            columns["boxes"].append(result["boxes"].numpy())
        arrays = {k: np.concatenate(v) if v else np.zeros((0, 4) if k == "boxes" else 0) for k, v in columns.items()}
        return Predictions(**arrays, meta={**self.meta, "producer": producer})


def save_logits(path: str | os.PathLike, dump: LogitDump) -> None:
    """Write atomically (temporary file + rename), as ``save_predictions``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, meta=np.array(json.dumps(dump.meta)),
                            image_id=dump.image_id.astype(np.int64), query=dump.query.astype(np.int32),
                            logits=dump.logits.astype(np.float16), boxes=dump.boxes.astype(np.float32))
        os.chmod(tmp, default_file_mode())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_logits(path: str | os.PathLike) -> LogitDump:
    with np.load(path, allow_pickle=False) as data:
        return LogitDump(*(data[name] for name in _ARRAYS), meta=json.loads(str(data["meta"])))
