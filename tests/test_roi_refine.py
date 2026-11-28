"""Unit and end-to-end tests for M6.2 RoI-Refine (autocheckout/roi_refine.py & tools/roi_refine_*.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from autocheckout.logit_dump import LogitDump
from autocheckout.roi_refine import (
    CropRecord,
    RefineConfig,
    ambiguous_query_mask,
    assert_no_overlap,
    build_crop_records,
    build_roi_classifier,
    crop_box,
    fuse_logits,
)
from autocheckout.taskcfg import TaskConfig
from tests.coco_helpers import make_ann, make_gt, make_image
from tools.roi_refine_eval import evaluate_roi_refine, infer_roi_logits
from tools.roi_refine_train import train_classifier


def test_assert_no_overlap_raises_on_leakage():
    assert_no_overlap([1, 2, 3], [4, 5, 6])
    with pytest.raises(ValueError, match="data leakage"):
        assert_no_overlap([1, 2, 3], [3, 4, 5], label="Val-B")


def test_crop_box_clamps_and_pads_safely():
    img = Image.new("RGB", (100, 80), color=(10, 20, 30))
    c1 = crop_box(img, [0, 0, 10, 10], pad_ratio=0.2)
    assert c1.size == (12, 12)
    c2 = crop_box(img, [95, 75, 105, 90], pad_ratio=0.1)
    assert c2.size[0] >= 1 and c2.size[1] >= 1
    assert c2.size[0] <= 100 and c2.size[1] <= 80


def test_refine_config_json_roundtrip():
    cfg = RefineConfig(margin_thresh=0.25, alpha=0.6, temp=1.2, topk=4, arch="resnet18")
    assert RefineConfig.from_json(cfg.to_json()) == cfg


def test_fuse_logits_identity_when_disabled_and_never_mutates():
    rng = np.random.default_rng(0)
    det = rng.normal(size=(2, 5, 10)).astype(np.float32)
    roi = rng.normal(size=(2, 5, 10)).astype(np.float32)
    det_copy = det.copy()

    out0 = fuse_logits(det, roi, margin_thresh=0.5, alpha=0.0)
    out_m0 = fuse_logits(det, roi, margin_thresh=0.0, alpha=0.7)
    np.testing.assert_array_equal(out0, det_copy)
    np.testing.assert_array_equal(out_m0, det_copy)
    np.testing.assert_array_equal(det, det_copy)


def test_fuse_logits_fixes_ambiguous_swap_and_respects_topk():
    # 1 image, 2 queries, 5 classes:
    # Query 0: ambiguous between class 0 (det=1.0) and class 1 (det=0.9); ROI strongly prefers class 1 (roi=5.0)
    #          and has huge score on non-top-2 class 4 (roi=20.0) which MUST be ignored when topk=2.
    # Query 1: confident on class 2 (det=4.0 vs -4.0) -> margin > 0.35 -> MUST stay untouched.
    det = np.array([[
        [1.0, 0.9, -3.0, -3.0, -3.0],
        [-4.0, -4.0, 4.0, -4.0, -4.0],
    ]], dtype=np.float32)
    roi = np.array([[
        [-2.0, 5.0, 0.0, 0.0, 20.0],
        [10.0, 0.0, -10.0, 0.0, 0.0],
    ]], dtype=np.float32)

    mask = ambiguous_query_mask(det, margin_thresh=0.35, min_prob=0.15)
    assert mask.tolist() == [[True, False]]

    fused = fuse_logits(det, roi, margin_thresh=0.35, alpha=0.5, topk=2, min_prob=0.15)
    # Query 0 flipped from class 0 -> class 1, while class 4 stayed untouched at -3.0
    assert int(np.argmax(fused[0, 0])) == 1
    assert fused[0, 0, 4] == pytest.approx(-3.0)
    # Query 1 completely untouched
    np.testing.assert_array_equal(fused[0, 1], det[0, 1])


def test_end_to_end_tiny_train_infer_and_eval(tmp_path):
    # Create 6 tiny images with distinct solid colors for class 0 (red) and class 1 (blue).
    img_dir = tmp_path / "val2019"
    img_dir.mkdir()
    images, anns = [], []
    ann_id = 1
    for img_id in range(1, 7):
        cls = 0 if img_id % 2 == 1 else 1
        color = (220, 20, 20) if cls == 0 else (20, 20, 220)
        fname = f"im_{img_id}.jpg"
        Image.new("RGB", (64, 64), color=color).save(img_dir / fname)
        images.append({**make_image(img_id), "file_name": fname, "width": 64, "height": 64})
        anns.append(make_ann(ann_id, img_id, cls, [8, 8, 48, 48]))
        ann_id += 1
    coco = make_gt(images=images, annotations=anns, num_labels=2)

    # Train on images 1, 2; calib on 3, 4; Val-B on 5, 6
    train_ids, calib_ids, valb_ids = [1, 2], [3, 4], [5, 6]
    assert_no_overlap(train_ids, calib_ids + valb_ids)
    records = build_crop_records(coco, tmp_path, train_ids, num_classes=2)
    assert len(records) == 2

    model, summary = train_classifier(
        records, num_classes=2, arch="tiny_test", pretrained=False,
        img_size=32, epochs=2, batch_size=2, num_workers=0, device="cpu", seed=0,
    )
    assert summary["num_crops"] == 2
    assert summary["classes_covered"] == 2
    assert summary["missing_classes_count"] == 0
    assert isinstance(build_roi_classifier("tiny_test", 2, pretrained=False), torch.nn.Module)

    # Build a synthetic LogitDump for calib+valb (ids 3,4,5,6) with 1 ambiguous query per image
    eval_ids = np.array([3, 4, 5, 6], dtype=np.int64)
    queries = np.zeros((4, 1), dtype=np.int32)
    logits = np.array([
        [[0.5, 0.45]],
        [[0.45, 0.5]],
        [[0.5, 0.45]],
        [[0.45, 0.5]],
    ], dtype=np.float32)
    boxes = np.tile(np.array([[[8.0, 8.0, 56.0, 56.0]]], dtype=np.float32), (4, 1, 1))
    dump = LogitDump(eval_ids, queries, logits, boxes, meta={"split": "val"})

    roi_logits, n_crops = infer_roi_logits(
        dump, coco, tmp_path, model, max_margin_thresh=0.5, min_prob=0.15,
        img_size=32, batch_size=2, device="cpu",
    )
    assert n_crops == 4
    assert roi_logits.shape == logits.shape

    # Minimal 1-task TaskConfig with 2 classes (0 and 1)
    cfg_dict = {
        "name": "tiny",
        "seed": 0,
        "num_slots": 2,
        "num_classes": 2,
        "tasks": [
            {"task_id": 1, "offset": 0, "classes": [
                {"label": 0, "name": "c0", "rpc_category_id": 1, "supercategory": "g"},
                {"label": 1, "name": "c1", "rpc_category_id": 2, "supercategory": "g"},
            ]}
        ],
    }
    cfg_path = tmp_path / "task_config.json"
    import json
    cfg_path.write_text(json.dumps(cfg_dict))
    cfg = TaskConfig.load(cfg_path)

    report, refined_preds = evaluate_roi_refine(
        dump, roi_logits, coco, cfg, calib_ids, valb_ids,
        stage=1, margins=(0.25,), alphas=(0.5,), topk=2, robust_calib=True,
    )
    assert "base" in report and "refined" in report and report["decision"] in ("accept", "reject")
    assert "calib_ACD" in report["best_config"]
    assert report["best_config"]["stage"] == 1
    assert len(refined_preds) > 0
