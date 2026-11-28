"""CPU unit & end-to-end tests for Decoupled Continual Object Detection
(Class-agnostic Detector dataset + Continual Crop Recognizer + End-to-end evaluator)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from autocheckout.continual_roi import (
    ContinualCropModel,
    CosineClassifier,
    collect_background_crops,
    compute_cl_metrics,
    kd_logit_loss,
    select_exemplars_herding,
)
from autocheckout.roi_refine import CropRecord, build_crop_records
from autocheckout.taskcfg import TaskConfig
from tests.coco_helpers import make_ann, make_gt, make_image
from tools.al_score import group_labels, score as al_score_fn
from tools.continual_roi_train import run_continual_training
from tools.eval_decoupled import compute_agnostic_recall, predict_decoupled
from tools.make_agnostic_dataset import build_agnostic_dataset


def _make_color_dataset(tmp_path):
    """4-class synthetic color dataset:
    Task 1: class 0 (red), class 1 (green)
    Task 2: class 2 (blue), class 3 (yellow)
    Images 1..8 = train (2/class), 9..12 = calib (1/class), 13..16 = valb (1/class).
    """
    colors = {
        0: (220, 20, 20),
        1: (20, 220, 20),
        2: (20, 20, 220),
        3: (220, 220, 20),
    }
    img_dir = tmp_path / "val2019"
    img_dir.mkdir(exist_ok=True)
    images, anns = [], []
    ann_id = 1
    img_id = 1
    train_ids, calib_ids, valb_ids = [], [], []
    for split_bucket, reps in ((train_ids, 2), (calib_ids, 1), (valb_ids, 1)):
        for _ in range(reps):
            for cls in range(4):
                fname = f"im_{img_id}.jpg"
                Image.new("RGB", (64, 64), color=colors[cls]).save(img_dir / fname)
                images.append({**make_image(img_id), "file_name": fname, "width": 64, "height": 64})
                anns.append(make_ann(ann_id, img_id, cls, [8, 8, 48, 48]))
                split_bucket.append(img_id)
                ann_id += 1
                img_id += 1
    coco = make_gt(images=images, annotations=anns, num_labels=4)
    cfg_dict = {
        "name": "tiny_2task",
        "seed": 0,
        "num_slots": 4,
        "num_classes": 4,
        "tasks": [
            {"task_id": 1, "offset": 0, "classes": [
                {"label": 0, "name": "c0", "rpc_category_id": 1, "supercategory": "g0"},
                {"label": 1, "name": "c1", "rpc_category_id": 2, "supercategory": "g0"},
            ]},
            {"task_id": 2, "offset": 2, "classes": [
                {"label": 2, "name": "c2", "rpc_category_id": 3, "supercategory": "g1"},
                {"label": 3, "name": "c3", "rpc_category_id": 4, "supercategory": "g1"},
            ]},
        ],
    }
    cfg_path = tmp_path / "task_config.json"
    cfg_path.write_text(json.dumps(cfg_dict))
    return coco, TaskConfig.load(cfg_path), train_ids, calib_ids, valb_ids


def test_make_agnostic_dataset_maps_all_labels_to_zero_and_blocks_leakage(tmp_path):
    coco, _, train_ids, calib_ids, valb_ids = _make_color_dataset(tmp_path)
    out = tmp_path / "agnostic_out"
    summary = build_agnostic_dataset(
        val_full=coco, val_img_root=tmp_path,
        train_real_ids=train_ids, calib_ids=calib_ids, valb_ids=valb_ids,
        out_dir=out,
    )
    assert summary["train"]["images"] == len(train_ids)
    assert summary["val_calib"]["images"] == len(calib_ids)
    assert (out / "dataset.yaml").is_file()
    # Every YOLO label line must start with class id '0 '
    lbl_files = list((out / "labels/train").glob("*.txt"))
    assert len(lbl_files) == len(train_ids)
    for lf in lbl_files:
        for line in lf.read_text().strip().splitlines():
            assert line.startswith("0 ")

    # Leakage into Val-B or calib must raise ValueError
    with pytest.raises(ValueError, match="data leakage"):
        build_agnostic_dataset(
            val_full=coco, val_img_root=tmp_path,
            train_real_ids=train_ids + [valb_ids[0]], calib_ids=calib_ids, valb_ids=valb_ids,
            out_dir=tmp_path / "leak",
        )


def test_cosine_classifier_add_classes_preserves_old_logits_bit_for_bit():
    torch.manual_seed(0)
    head = CosineClassifier(in_features=16, num_classes=3, scale=20.0, margin=0.2)
    head.eval()
    x = torch.randn(4, 16)
    old_logits = head(x).clone()

    new_w = torch.randn(2, 16)
    head.add_classes(new_w)
    assert head.num_classes == 5
    new_logits = head(x)
    assert torch.allclose(old_logits, new_logits[:, :3], atol=1e-6)

    # ArcFace margin reduces target logit during training
    head.train()
    targets = torch.tensor([0, 1, 2, 3])
    with_margin = head(x, targets=targets)
    no_margin = head(x, targets=None)
    assert (with_margin[torch.arange(4), targets] < no_margin[torch.arange(4), targets]).all()


def test_herding_exemplar_selection_and_kd_loss():
    rng = np.random.default_rng(0)
    recs = [CropRecord("a.jpg", (0, 0, 10, 10), cls, i) for i, cls in enumerate([0] * 10 + [1] * 10)]
    embs = rng.normal(size=(20, 8)).astype(np.float32)
    chosen = select_exemplars_herding(recs, embs, per_class=3)
    assert len(chosen) == 6
    assert sum(1 for r in chosen if r.label == 0) == 3
    assert sum(1 for r in chosen if r.label == 1) == 3

    t_logits = torch.randn(4, 5)
    assert kd_logit_loss(t_logits, t_logits, num_old_classes=3).item() == pytest.approx(0.0, abs=1e-6)
    assert kd_logit_loss(t_logits + 1.0, t_logits, num_old_classes=0).item() == 0.0


def test_collect_background_crops_filters_by_iou(tmp_path):
    coco, _, train_ids, _, _ = _make_color_dataset(tmp_path)
    # For image 1, GT is [8, 8, 56, 56]. Box 0 overlaps GT (IoU=1.0), Box 1 is far away (IoU=0.0).
    props = {1: np.array([[8, 8, 56, 56], [0, 0, 6, 6]], dtype=np.float32)}
    bgs = collect_background_crops(props, coco, tmp_path, bg_label=4, iou_thresh=0.3, max_per_image=3)
    assert len(bgs) == 1
    assert bgs[0].label == 4 and bgs[0].image_id == 1


def test_cl_metrics_computation():
    # Task 1 acc 0.90 -> 0.85 (drop 0.05); Task 2 acc 0.80 -> 0.80 (drop 0.0); Task 3 acc 0.75
    mat = [
        [0.90, None, None],
        [0.88, 0.80, None],
        [0.85, 0.80, 0.75],
    ]
    m = compute_cl_metrics(mat)
    assert m["ACC"] == pytest.approx((0.85 + 0.80 + 0.75) / 3.0)
    assert m["Forgetting"] == pytest.approx((0.05 + 0.0) / 2.0)
    assert m["BWT"] == pytest.approx((-0.05 + 0.0) / 2.0)


def test_end_to_end_2task_prototype_and_replay_kd_with_decoupled_eval(tmp_path):
    coco, cfg, train_ids, calib_ids, valb_ids = _make_color_dataset(tmp_path)
    all_train = build_crop_records(coco, tmp_path, train_ids, num_classes=4)
    all_calib = build_crop_records(coco, tmp_path, calib_ids, num_classes=4)

    task_cls = [[0, 1], [2, 3]]
    t_train = [[r for r in all_train if r.label in set(c)] for c in task_cls]
    t_eval = [[r for r in all_calib if r.label in set(c)] for c in task_cls]

    # Mode A: Prototype-only (backbone frozen after Task 1)
    model_proto, sum_proto = run_continual_training(
        t_train, t_eval, task_cls,
        mode="prototype", arch="tiny_test", pretrained=False,
        img_size=32, epochs_task1=3, batch_size=4, device="cpu", seed=0,
    )
    assert sum_proto["final_fg_classes"] == 4
    assert len(sum_proto["acc_matrix"]) == 2
    # Mode B: Replay + Logit KD + Background class
    bg_dummy = [CropRecord(all_train[0].image_path, (0, 0, 10, 10), 2, all_train[0].image_id)]
    model_rep, sum_rep = run_continual_training(
        t_train, t_eval, task_cls,
        mode="replay_kd", arch="tiny_test", pretrained=False,
        include_background=True, bg_records=bg_dummy,
        img_size=32, epochs_task1=3, epochs_inc=2, batch_size=4,
        exemplars_per_class=2, device="cpu", seed=0,
    )
    assert model_rep.num_fg_classes == 4 and model_rep.bg_class_index == 4
    assert sum_rep["tasks"][1]["exemplars_stored"] == 8

    # End-to-end Stage-1 proposals -> Stage-2 ContinualCropModel -> al_score
    eval_ids = calib_ids + valb_ids
    p_img = np.array(eval_ids, dtype=np.int64)
    p_box = np.tile(np.array([[8.0, 8.0, 56.0, 56.0]], dtype=np.float32), (len(eval_ids), 1))
    p_scr = np.full(len(eval_ids), 0.95, dtype=np.float32)

    recall = compute_agnostic_recall(
        {int(i): p_box[idx : idx + 1] for idx, i in enumerate(eval_ids)},
        coco, valb_ids, seen_classes=4,
    )
    assert recall["recall_at_iou"] == pytest.approx(1.0)

    preds = predict_decoupled(p_img, p_box, p_scr, coco, tmp_path, model_proto, img_size=32, device="cpu")
    assert len(preds) == len(eval_ids)
    res = al_score_fn(coco, cfg, preds, calib_ids, valb_ids, groups=group_labels(cfg, 2), robust=True)
    assert 0.0 <= res["cAcc_m2"] <= 1.0
