"""End-to-end evaluator for Decoupled Continual Object Detection:
Stage 1 class-agnostic box proposals (``.npz`` with ``image_id``, ``boxes``, ``score``) ->
high-res crop -> Stage 2 ContinualCropModel -> standard repo ``Predictions (.npz)`` ->
strict Val-A-calib threshold selection + single Val-B evaluation via ``tools.al_score``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from autocheckout.continual_roi import ContinualCropModel, _box_iou_matrix
from autocheckout.datapaths import Layout
from autocheckout.io import load_json, save_json
from autocheckout.predictions import Predictions, save_predictions
from autocheckout.roi_refine import build_transforms, crop_box, resolve_image_path
from autocheckout.taskcfg import TaskConfig
from tools.al_score import STAGE, UNLEARNED_IOU, group_labels, score as al_score_fn


def compute_agnostic_recall(
    boxes_by_img: dict[int, np.ndarray],
    val_full: dict[str, Any],
    eval_ids: list[int],
    *,
    seen_classes: int = 200,
    iou_thresh: float = 0.50,
) -> dict[str, float | int]:
    """Compute class-agnostic Recall@IoU for Stage-1 box proposals on ``eval_ids``."""
    eval_set = set(int(i) for i in eval_ids)
    gt_by_img: dict[int, list[list[float]]] = defaultdict(list)
    for ann in val_full["annotations"]:
        img_id = int(ann["image_id"])
        if img_id in eval_set and 0 <= int(ann["category_id"]) < seen_classes:
            x, y, w, h = (float(v) for v in ann["bbox"])
            gt_by_img[img_id].append([x, y, x + w, y + h])

    total_gt = 0
    matched_gt = 0
    total_props = 0
    for img_id in eval_ids:
        gt_arr = np.asarray(gt_by_img.get(img_id, []), dtype=np.float32).reshape(-1, 4)
        prop_arr = np.asarray(boxes_by_img.get(img_id, []), dtype=np.float32).reshape(-1, 4)
        total_gt += len(gt_arr)
        total_props += len(prop_arr)
        if len(gt_arr) == 0 or len(prop_arr) == 0:
            continue
        iou = _box_iou_matrix(gt_arr, prop_arr)
        matched_gt += int((iou.max(axis=1) >= iou_thresh).sum())

    return {
        "eval_images": len(eval_ids),
        "total_gt": total_gt,
        "matched_gt": matched_gt,
        "missed_gt": total_gt - matched_gt,
        "total_proposals": total_props,
        "recall_at_iou": matched_gt / max(1, total_gt),
        "iou_thresh": iou_thresh,
    }


@torch.no_grad()
def predict_decoupled(
    proposals_image_id: np.ndarray,
    proposals_boxes: np.ndarray,
    proposals_score: np.ndarray,
    val_full: dict[str, Any],
    img_root: Path,
    model: ContinualCropModel,
    *,
    min_obj_score: float = 0.15,
    img_size: int = 224,
    pad_ratio: float = 0.10,
    batch_size: int = 128,
    device: str = "cuda",
) -> Predictions:
    """Classify Stage-1 proposals with Stage-2 ``ContinualCropModel`` and return repo ``Predictions``."""
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = model.to(dev).eval()
    tf = build_transforms(img_size=img_size, train=False)

    keep = np.asarray(proposals_score, dtype=np.float32) >= min_obj_score
    img_ids = np.asarray(proposals_image_id, dtype=np.int64)[keep]
    boxes = np.asarray(proposals_boxes, dtype=np.float32).reshape(-1, 4)[keep]
    obj_scores = np.asarray(proposals_score, dtype=np.float32)[keep]

    id_to_file = {int(im["id"]): im["file_name"] for im in val_full["images"]}
    by_img: dict[int, list[int]] = defaultdict(list)
    for idx, iid in enumerate(img_ids.tolist()):
        if iid in id_to_file:
            by_img[iid].append(idx)

    out_img: list[int] = []
    out_q: list[int] = []
    out_lbl: list[int] = []
    out_scr: list[float] = []
    out_box: list[np.ndarray] = []

    fg = model.num_fg_classes
    bg_idx = model.bg_class_index

    batch_tensors: list[torch.Tensor] = []
    batch_meta: list[tuple[int, int, float, np.ndarray]] = []

    def flush() -> None:
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors, dim=0).to(dev)
        probs = F.softmax(model(x).float(), dim=-1).cpu().numpy()
        for (iid, qi, osc, bx), p_vec in zip(batch_meta, probs, strict=True):
            best_all = int(np.argmax(p_vec))
            if bg_idx is not None and best_all == bg_idx:
                continue
            fg_vec = p_vec[:fg]
            best_fg = int(np.argmax(fg_vec))
            combined_score = float(osc * fg_vec[best_fg])
            out_img.append(iid)
            out_q.append(qi)
            out_lbl.append(best_fg)
            out_scr.append(combined_score)
            out_box.append(bx)
        batch_tensors.clear()
        batch_meta.clear()

    for iid, idxs in sorted(by_img.items()):
        path = resolve_image_path(img_root, id_to_file[iid])
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            for q_local, idx in enumerate(idxs):
                bx = boxes[idx]
                crop = crop_box(rgb, bx, pad_ratio=pad_ratio)
                batch_tensors.append(tf(crop))
                batch_meta.append((iid, q_local, float(obj_scores[idx]), bx))
                if len(batch_tensors) >= batch_size:
                    flush()
    flush()

    if not out_img:
        return Predictions(
            np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            meta={"producer": "decoupled_cl"},
        )
    return Predictions(
        np.asarray(out_img, dtype=np.int64),
        np.asarray(out_q, dtype=np.int32),
        np.asarray(out_lbl, dtype=np.int32),
        np.asarray(out_scr, dtype=np.float32),
        np.stack(out_box, axis=0).astype(np.float32),
        meta={"producer": "decoupled_cl", "min_obj_score": min_obj_score},
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, help="autocheckout-data folder (default: $AC_DATA)")
    ap.add_argument("--img-root", type=Path, help="image root containing val2019")
    ap.add_argument("--proposals", type=Path, required=True, help="Stage-1 proposals .npz (image_id, boxes, score)")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Stage-2 continual_roi.pth")
    ap.add_argument("--name", default="decoupled_cl_val")
    ap.add_argument("--stage", type=int, default=STAGE)
    ap.add_argument("--unlearned-iou", type=float, default=UNLEARNED_IOU)
    ap.add_argument("--min-obj-score", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    data = Layout.at(args.data)
    img_root = args.img_root or data.rpc
    if not (img_root / "val2019").is_dir() and (data.root / "archive" / "val2019").is_dir():
        img_root = data.root / "archive"

    val_full = load_json(data.eval / "val_full.json")
    cfg = TaskConfig.load(data.eval / "task_config.json")
    calib_ids = sorted(int(i) for i in load_json(data.relearn / "active/splits/val_a_split.json")["calib"])
    valb_ids = sorted(int(im["id"]) for im in load_json(data.relearn / "meta/val_b.json")["images"])

    with np.load(args.proposals, allow_pickle=False) as prop:
        p_img = prop["image_id"]
        p_box = prop["boxes"]
        p_scr = prop["score"]

    boxes_by_img: dict[int, list[np.ndarray]] = defaultdict(list)
    for iid, bx, sc in zip(p_img.tolist(), p_box, p_scr.tolist(), strict=True):
        if sc >= args.min_obj_score:
            boxes_by_img[int(iid)].append(bx)
    boxes_np = {k: np.stack(v, axis=0) for k, v in boxes_by_img.items()}
    recall = compute_agnostic_recall(boxes_np, val_full, valb_ids, seen_classes=cfg.seen_classes(args.stage))

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = ContinualCropModel(
        arch=ckpt["arch"],
        num_classes=int(ckpt["num_fg_classes"]),
        include_background=bool(ckpt.get("include_background", False)),
        scale=float(ckpt.get("scale", 20.0)),
        margin=0.0,
        pretrained=False,
    )
    model.load_state_dict(ckpt["state_dict"])

    preds = predict_decoupled(
        p_img, p_box, p_scr, val_full, img_root, model,
        min_obj_score=args.min_obj_score,
        img_size=int(ckpt.get("img_size", 224)),
        pad_ratio=float(ckpt.get("pad_ratio", 0.10)),
        device=args.device,
    )
    groups = group_labels(cfg, args.stage)
    eff_unlearned = args.unlearned_iou if args.stage < len(cfg.data_tasks) else None
    std_res = al_score_fn(val_full, cfg, preds, calib_ids, valb_ids, groups=groups, unlearned_iou=eff_unlearned, robust=False)
    rob_res = al_score_fn(val_full, cfg, preds, calib_ids, valb_ids, groups=groups, unlearned_iou=eff_unlearned, robust=True)

    args.out.mkdir(parents=True, exist_ok=True)
    save_predictions(args.out / f"{args.name}_pred_val.npz", preds)
    report = {
        "name": args.name,
        "stage1_recall_valb": recall,
        "standard_calib": {k: v for k, v in std_res.items() if k != "right_images"},
        "robust_calib": {k: v for k, v in rob_res.items() if k != "right_images"},
    }
    save_json(args.out / f"{args.name}.json", report, indent=2)
    print(
        f"[DecoupledCL] {args.name}: Stage1 Recall@0.5={recall['recall_at_iou']*100:.2f}% | "
        f"Val-B cAcc={std_res['cAcc_m2']*100:.2f}% (robust={rob_res['cAcc_m2']*100:.2f}%)",
        flush=True,
    )
    return report


if __name__ == "__main__":
    main()
