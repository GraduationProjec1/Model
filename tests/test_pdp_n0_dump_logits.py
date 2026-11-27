"""N0 (docs/ai/planning/2026-11-26-plan-N1-N5-hieu-chinh-task-va-loss.md §2): the ranking moved to rank_topk gives the
same predictions, and --dump_logits 1 (predict only) saves the raw logits of the most confident queries."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.image_transforms import center_to_corners_format

from autocheckout.logit_dump import LogitDump, load_logits, save_logits
from autocheckout.task_calib import TaskCalib

from inference import predict_batch
from pdp_helpers import pdp_args, run_main


class FixedOutputs(torch.nn.Module):
    def __init__(self, logits, boxes):
        super().__init__()
        self.logits, self.boxes = logits, boxes

    def forward(self, pixel_values, pixel_mask, query=None, train=False):
        return SimpleNamespace(logits=self.logits.clone(), pred_boxes=self.boxes,
                               last_hidden_state=torch.zeros(len(self.logits), 300, 256))


def original_predict_batch(logits, pred_boxes, target_sizes, seen_classes, topk=100):
    """predict_batch before N0, kept verbatim as the reference."""
    prob = logits[..., :seen_classes].sigmoid()
    batch_size, num_queries, num_classes = prob.shape
    scores, index = prob.reshape(batch_size, -1).topk(min(topk, num_queries * num_classes), dim=1)
    queries = torch.div(index, num_classes, rounding_mode="floor")
    labels = index % num_classes
    boxes = center_to_corners_format(pred_boxes).gather(1, queries.unsqueeze(-1).expand(-1, -1, 4))
    height, width = target_sizes.unbind(1)
    boxes = boxes * torch.stack([width, height, width, height], dim=1)[:, None, :].to(boxes.dtype)
    return [{"scores": s, "labels": label, "queries": q, "boxes": b}
            for s, label, q, b in zip(scores, labels, queries, boxes, strict=True)]


def fixed_batch(seed=0, n_classes=12):
    torch.manual_seed(seed)
    return torch.randn(2, 300, n_classes), torch.rand(2, 300, 4) * 0.5 + 0.1, torch.tensor([[800, 800], [600, 800]])


def run_batch(logits, boxes, sizes, seen, **kwargs):
    return predict_batch(FixedOutputs(logits, boxes), None, None, sizes, use_prompts=False, local_query=True,
                         seen_classes=seen, **kwargs)


@pytest.mark.parametrize("return_logits", [False, True])
def test_rank_topk_gives_bit_identical_predictions(return_logits):
    logits, boxes, sizes = fixed_batch()
    ours = run_batch(logits, boxes, sizes, 9, return_logits=return_logits)
    for mine, reference in zip(ours, original_predict_batch(logits, boxes, sizes, 9), strict=True):
        for key in ("scores", "labels", "queries", "boxes"):
            assert torch.equal(mine[key], reference[key]), key


def test_return_logits_keeps_the_raw_logits_of_the_most_confident_queries():
    logits, boxes, sizes = fixed_batch()
    [first, _] = run_batch(logits, boxes, sizes, 9, return_logits=True)
    best = logits[0, :, :9].max(dim=-1).values
    assert first["dump_queries"].shape == (100,)
    assert torch.equal(first["dump_queries"], best.topk(100).indices)
    assert torch.equal(first["dump_logits"], logits[0, first["dump_queries"], :9])
    assert set(first["queries"].tolist()) <= set(first["dump_queries"].tolist())


def test_dump_reproduces_the_predictions_of_the_model():
    torch.manual_seed(0)
    # distinct multiples of 1/256 in (-6, 6): exact in float16, so no score ties after the round trip
    logits = ((torch.randperm(2 * 120 * 12).float() - 1434) / 256).reshape(2, 120, 12)
    boxes, sizes = torch.rand(2, 120, 4) * 0.5 + 0.1, torch.tensor([[800, 800], [600, 800]])
    results = run_batch(logits, boxes, sizes, 9, return_logits=True)
    dump = LogitDump(np.array([7, 8]), np.stack([r["dump_queries"].numpy() for r in results]),
                     np.stack([r["dump_logits"].numpy().astype(np.float16) for r in results]),
                     np.stack([r["dump_boxes"].numpy() for r in results]))
    preds = dump.predictions()
    for image_id, result in zip((7, 8), results, strict=True):
        rows = preds.image_id == image_id
        assert np.array_equal(preds.query[rows], result["queries"].numpy())
        assert np.array_equal(preds.label[rows], result["labels"].numpy())
        np.testing.assert_allclose(preds.score[rows], result["scores"].numpy(), rtol=0, atol=1e-6)
        np.testing.assert_allclose(preds.boxes[rows], result["boxes"].numpy(), rtol=1e-6)


def test_task_calib_none_and_identity_leave_predictions_unchanged():
    logits, boxes, sizes = fixed_batch()
    identity = TaskCalib.identity([range(0, 5), range(5, 9)])
    for calib in (None, identity):
        for mine, reference in zip(run_batch(logits, boxes, sizes, 9, task_calib=calib),
                                   original_predict_batch(logits, boxes, sizes, 9), strict=True):
            assert torch.equal(mine["labels"], reference["labels"])
            assert torch.equal(mine["scores"], reference["scores"])


def test_task_calib_shifts_the_scores_of_its_task():
    logits, boxes, sizes = fixed_batch()
    calib = TaskCalib((1.0, 1.0), (-50.0, 0.0), ((0, 1, 2, 3, 4), (5, 6, 7, 8)))
    for result in run_batch(logits, boxes, sizes, 9, task_calib=calib, topk=20):
        assert (result["labels"] >= 5).all()


def test_logit_file_round_trip_shapes_and_dtypes(tmp_path):
    dump = LogitDump(np.array([1, 2]), np.zeros((2, 3), dtype=np.int32), np.ones((2, 3, 4), dtype=np.float32),
                     np.zeros((2, 3, 4), dtype=np.float32), meta={"split": "val"})
    save_logits(tmp_path / "logits_val.npz", dump)
    back = load_logits(tmp_path / "logits_val.npz")
    assert back.logits.dtype == np.float16 and back.logits.shape == (2, 3, 4)
    assert back.boxes.shape == (2, 3, 4) and back.meta == {"split": "val"}


def test_logit_dump_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="boxes"):
        LogitDump(np.array([1]), np.zeros((1, 3)), np.zeros((1, 3, 4)), np.zeros((1, 2, 4)))


@pytest.mark.parametrize("flag, value", [("--dump_logits", "1"), ("--task_calib", "calib.json")])
def test_prediction_flags_are_refused_when_training(tmp_path, flag, value):
    import main as pdp_main

    args = pdp_args(tmp_path, flag, value)
    with pytest.raises(SystemExit, match=flag):
        pdp_main.main(args)


@pytest.mark.slow
def test_predict_only_writes_the_logits_next_to_the_predictions(tmp_path, monkeypatch):
    from autocheckout.predictions import load_predictions

    import main as pdp_main

    args = run_main(tmp_path, monkeypatch, "--n_tasks", "1", "--pred_splits", "val")
    task = tmp_path / "run" / "task_1"
    args.predict_only, args.dump_logits, args.output_dir = 1, 1, str(tmp_path / "run")
    pdp_main.main(args)
    dump, preds = load_logits(task / "logits_val.npz"), load_predictions(task / "pred_val.npz")
    assert dump.logits.dtype == np.float16 and dump.logits.shape[-1] == preds.meta["seen_classes"]
    assert sorted(dump.image_id.tolist()) == sorted(set(preds.image_id.tolist()))
    replay = dump.predictions()
    assert len(replay) == len(preds)
    np.testing.assert_allclose(np.sort(replay.score), np.sort(preds.score), atol=2e-3)
