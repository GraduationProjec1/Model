"""G0 (docs/ai/planning/2026-11-25-ke-hoach-chay.md): P1 (--reuse_features) on a real checkpoint, speed and same outputs.

Port of docs/design/2026-11-22-toi-uu-2-luot/assets/speed_evidence.py (the CPU evidence on r4) with the checkpoint,
images and device as arguments. Every image runs the old way (reuse off) and the new way (reuse on) back to back,
in alternating order; hooks count and time backbone/encoder/decoder. Passes when every image keeps the same top-k
(query, class) list and no score differs by more than --tol. Exit code 0 when it passes, 2 when it does not.

    python -m tools.p1_check --checkpoint <task_final.pth> --val <val_full.json> --img-root <raw root> \\
        --split <val_a_split.json> --device cuda --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from autocheckout.datapaths import REPO

TOPK = 100
STAGES = ("backbone", "encoder", "decoder")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="task_final.pth with its runtime")
    parser.add_argument("--val", type=Path, required=True, help="COCO file of the images (e.g. eval/val_full.json)")
    parser.add_argument("--img-root", type=Path, required=True, help="folder the file_name entries are relative to")
    parser.add_argument("--split", type=Path, required=True, help="JSON with the image ids to use (Val-A split)")
    parser.add_argument("--split-key", default="pool")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tol", type=float, default=1e-4, help="largest score difference allowed (GPU plan: 1e-4)")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def load_model(path: Path, device):
    import torch
    from models.configuration_deformable_detr import DeformableDetrConfig
    from models.image_processing_deformable_detr import DeformableDetrImageProcessor
    from models.modeling_deformable_detr import DeformableDetrForObjectDetection

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = DeformableDetrConfig.from_dict(checkpoint["runtime"]["model_config"])
    config.use_pretrained_backbone = False
    config.backbone_pretrained_file = None
    model = DeformableDetrForObjectDetection(config, default=True, log_file=open(os.devnull, "w"))  # noqa: SIM115
    model.load_state_dict(checkpoint["model"], strict=True)
    model.model.prompts.set_task_id(checkpoint["task_id"] - 1)
    processor = DeformableDetrImageProcessor.from_dict(checkpoint["runtime"]["processor_config"])
    return model.to(device).eval(), processor


def load_images(args, processor):
    from datasets.coco_hug import CocoDetection

    ids = set(json.loads(args.split.read_text())[args.split_key])
    val = json.loads(args.val.read_text())
    images = [image for image in val["images"] if image["id"] in ids][:args.n + args.warmup]
    if len(images) < args.n + args.warmup:
        raise ValueError(f"only {len(images)} images of {args.split_key} in {args.val}, need {args.n + args.warmup}")
    keep = {image["id"] for image in images}
    subset = {"images": images, "annotations": [a for a in val["annotations"] if a["image_id"] in keep],
              "categories": val["categories"]}
    ann = args.out / "p1_check_images.json"
    ann.write_text(json.dumps(subset))
    dataset = CocoDetection(img_folder=str(args.img_root), ann_file=str(ann), processor=processor)
    return [(images[i]["id"], dataset.collate_fn([dataset[i]])) for i in range(len(dataset))]


def compare(old, new, topk):
    """Bitwise equality, largest score difference and same top-k (query, class) list of one image."""
    import torch

    names = ("logits", "pred_boxes", "last_hidden_state")
    old_scores, new_scores = old.logits.sigmoid().flatten(), new.logits.sigmoid().flatten()
    return {"identical": all(torch.equal(getattr(old, k), getattr(new, k)) for k in names),
            "max_abs_diff": float((old_scores - new_scores).abs().max()),
            "topk_identical": torch.equal(old_scores.topk(topk).indices, new_scores.topk(topk).indices)}


def measure(model, samples, device, warmup):
    import torch
    from two_pass import prompted_outputs

    stats = {}

    def before(name):
        def hook(module, inputs):
            stats.setdefault(name, {"calls": 0, "seconds": 0.0})["start"] = time.perf_counter()
        return hook

    def after(name):
        def hook(module, inputs, output):
            sync()
            stats[name]["calls"] += 1
            stats[name]["seconds"] += time.perf_counter() - stats[name]["start"]
        return hook

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for name in STAGES:
        getattr(model.model, name).register_forward_pre_hook(before(name))
        getattr(model.model, name).register_forward_hook(after(name))

    def run(pixels, mask, reuse):
        stats.clear()
        sync()
        start = time.perf_counter()
        with torch.no_grad():
            out = prompted_outputs(model, pixels, mask, use_prompts=True, local_query=model.config.local_query,
                                   reuse_features=reuse)
        sync()
        return time.perf_counter() - start, {n: dict(stats[n]) for n in STAGES}, out

    rows = []
    for index, (image_id, batch) in enumerate(samples):
        pixels, mask = batch["pixel_values"].to(device), batch["pixel_mask"].to(device)
        order = (False, True) if index % 2 == 0 else (True, False)
        result = {reuse: run(pixels, mask, reuse) for reuse in order}
        if index < warmup:
            continue
        (old_s, old_stages, old), (new_s, new_stages, new) = result[False], result[True]
        row = {"image_id": image_id, "seconds_old": old_s, "seconds_new": new_s, **compare(old, new, TOPK)}
        for tag, stages in (("old", old_stages), ("new", new_stages)):
            for name in STAGES:
                row[f"{name}_calls_{tag}"] = stages[name]["calls"]
                row[f"{name}_s_{tag}"] = stages[name]["seconds"]
        rows.append(row)
        print(f"{len(rows):3d} id={image_id} cũ={old_s:.3f}s mới={new_s:.3f}s giống top-k={row['topk_identical']}",
              flush=True)
    return rows


def summarize(rows, args, device):
    import torch

    mean = lambda key: sum(r[key] for r in rows) / len(rows)  # noqa: E731
    old, new = sum(r["seconds_old"] for r in rows), sum(r["seconds_new"] for r in rows)
    max_diff = max(r["max_abs_diff"] for r in rows)
    topk = sum(r["topk_identical"] for r in rows)
    return {
        "checkpoint": str(args.checkpoint), "images": len(rows), "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__,
        "seconds_per_image_old": round(old / len(rows), 4), "seconds_per_image_new": round(new / len(rows), 4),
        "saving_total": round(1 - new / old, 4),
        "calls_old": {n: mean(f"{n}_calls_old") for n in STAGES},
        "calls_new": {n: mean(f"{n}_calls_new") for n in STAGES},
        "stage_seconds_old": {n: round(mean(f"{n}_s_old"), 4) for n in STAGES},
        "stage_seconds_new": {n: round(mean(f"{n}_s_new"), 4) for n in STAGES},
        "bitwise_identical": sum(r["identical"] for r in rows), "topk_identical": topk,
        "max_abs_diff": max_diff, "tol": args.tol, "passed": topk == len(rows) and max_diff <= args.tol,
    }


def main(argv=None) -> dict:
    sys.path.insert(0, str(REPO / "pdp"))
    import torch

    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    device = torch.device(args.device)
    model, processor = load_model(args.checkpoint, device)
    rows = measure(model, load_images(args, processor), device, args.warmup)
    with (args.out / "p1_check.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, args, device)
    (args.out / "p1_check_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


def cli() -> None:
    sys.exit(0 if main()["passed"] else 2)


if __name__ == "__main__":
    cli()
