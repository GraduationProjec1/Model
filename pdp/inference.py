"""Two-pass inference of the PDP model and export of prediction files (fix F9, task V1).

Inference is the same everywhere (validation during training, predictions after each task,
predict-only runs): a first pass without prompts gives the query, the second pass uses the prompts
of every learned task. Per image, the top-100 (query, class) pairs over the learned classes are kept,
like the post-processing of the original evaluation (which masked unlearned logits and dropped the
unused last slot). Files follow docs/data_preprocessing/formats.md section 5.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.image_transforms import center_to_corners_format

from autocheckout.io import md5_file
from autocheckout.logit_dump import TOPK_PER_IMAGE, LogitDump, rank_topk, save_logits, top_queries
from autocheckout.predictions import Predictions, save_predictions
from autocheckout.task_calib import TaskCalib
from datasets.coco_hug import CocoDetection
from two_pass import prompted_outputs


@torch.no_grad()
def predict_batch(
    model,
    pixel_values,
    pixel_mask,
    target_sizes,
    *,
    use_prompts,
    local_query,
    seen_classes,
    topk=TOPK_PER_IMAGE,
    reuse_features=False,
    task_calib=None,
    return_logits=False,
):
    """Top-k (query, class) pairs of each image over classes < seen_classes.

    target_sizes: [B, 2] (height, width) the normalised boxes are scaled to.
    Returns one dict per image: scores, labels, queries, boxes (x1, y1, x2, y2 in pixels).
    reuse_features (P1): the second pass reuses the backbone and encoder of the first; same results.
    task_calib (N1): a TaskCalib applied to the logits before the sigmoid (None = as before).
    return_logits (N0): each dict also gets the raw logits and boxes of the topk most confident queries
    (dump_queries, dump_logits, dump_boxes).
    """
    outputs = prompted_outputs(
        model,
        pixel_values,
        pixel_mask,
        use_prompts=use_prompts,
        local_query=local_query,
        reuse_features=reuse_features,
    )

    raw = outputs.logits[..., :seen_classes]
    logits = raw if task_calib is None else task_calib.apply(raw)
    boxes = center_to_corners_format(outputs.pred_boxes)
    height, width = target_sizes.unbind(1)
    boxes = boxes * torch.stack([width, height, width, height], dim=1)[:, None, :].to(
        boxes.dtype
    )
    results = rank_topk(logits.sigmoid(), boxes, topk)
    if return_logits:
        for result, kept, image_logits, image_boxes in zip(results, top_queries(raw, topk), raw, boxes):
            result.update(dump_queries=kept, dump_logits=image_logits[kept], dump_boxes=image_boxes[kept])
    return results


def predict_loader(
    model,
    dataloader,
    device,
    *,
    use_prompts,
    local_query,
    seen_classes,
    meta=None,
    reuse_features=False,
    task_calib=None,
    dump_logits=False,
):
    """Predictions of the model on every image of a dataloader of CocoDetection batches.

    dump_logits (N0): return (Predictions, LogitDump) instead of the Predictions alone."""
    model.eval()
    columns = {"image_id": [], "query": [], "label": [], "score": [], "boxes": []}
    dump = {"image_id": [], "query": [], "logits": [], "boxes": []}
    for batch in dataloader:
        targets = batch["labels"]
        sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
        results = predict_batch(
            model,
            batch["pixel_values"].to(device),
            batch["pixel_mask"].to(device),
            sizes,
            use_prompts=use_prompts,
            local_query=local_query,
            seen_classes=seen_classes,
            reuse_features=reuse_features,
            task_calib=task_calib,
            return_logits=dump_logits,
        )
        for target, result in zip(targets, results):
            columns["image_id"].append(
                np.full(len(result["scores"]), int(target["image_id"]))
            )
            columns["query"].append(result["queries"].cpu().numpy())
            columns["label"].append(result["labels"].cpu().numpy())
            columns["score"].append(result["scores"].cpu().numpy())
            columns["boxes"].append(result["boxes"].cpu().numpy())
            if dump_logits:
                dump["image_id"].append(int(target["image_id"]))
                dump["query"].append(result["dump_queries"].cpu().numpy())
                dump["logits"].append(result["dump_logits"].cpu().numpy().astype(np.float16))
                dump["boxes"].append(result["dump_boxes"].cpu().numpy())
    arrays = {
        name: (
            np.concatenate(parts)
            if parts
            else np.zeros((0, 4) if name == "boxes" else 0)
        )
        for name, parts in columns.items()
    }
    predictions = Predictions(**arrays, meta=dict(meta or {}))
    if not dump_logits:
        return predictions
    logits = LogitDump(np.array(dump["image_id"], dtype=np.int64), np.stack(dump["query"]),
                       np.stack(dump["logits"]), np.stack(dump["boxes"]),
                       meta={**dict(meta or {}), "queries_per_image": TOPK_PER_IMAGE})
    return predictions, logits


def write_predictions(
    model, args, *, ann_file, out_file, task_id, seen_classes, split, processor, device
):
    """Predict every image of a COCO file (e.g. val_full.json) and save pred_<split>.npz (V1).

    --task_calib (N1): the per-task logit calibration applied before the sigmoid, recorded in the meta;
    --dump_logits 1 (N0): also save the raw logits as logits_<split>.npz next to out_file."""
    dataset = CocoDetection(
        img_folder=args.test_img_dir, ann_file=ann_file, processor=processor
    )
    loader = DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    meta = {
        "task_id": task_id,
        "seen_classes": seen_classes,
        "split": split,
        "ann_file": str(ann_file),
        "ann_md5": md5_file(ann_file),
        "producer": "pdp",
        "topk_per_image": TOPK_PER_IMAGE,
        "processor_size": dict(processor.size),
        "coordinates": "native_pixels",
        "backbone": model.config.backbone,
    }
    task_calib = None
    if getattr(args, "task_calib", ""):
        task_calib = TaskCalib.load(args.task_calib)
        meta["task_calib"] = {"file": str(args.task_calib), **task_calib.to_dict()}
    dump_logits = bool(getattr(args, "dump_logits", 0))
    predictions = predict_loader(
        model,
        loader,
        device,
        use_prompts=bool(args.use_prompts),
        local_query=bool(args.local_query),
        seen_classes=seen_classes,
        meta=meta,
        reuse_features=bool(getattr(args, "reuse_features", 0)),
        task_calib=task_calib,
        dump_logits=dump_logits,
    )
    if dump_logits:
        predictions, logits = predictions
        save_logits(os.path.join(os.path.dirname(out_file), f"logits_{split}.npz"), logits)
    save_predictions(out_file, predictions)
    return predictions
