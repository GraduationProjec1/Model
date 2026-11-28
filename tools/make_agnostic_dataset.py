"""Stage 1 dataset builder for Decoupled Continual Object Detection:
converts COCO annotations (synthetic + real Val-A selection) into a 1-class (``0 = product``)
YOLO / RT-DETR dataset, strictly verifying zero overlap with Val-A-calib and Val-B.

Example:
    python -m tools.make_agnostic_dataset --data /workspace/autocheckout-data \\
        --selection margin_k_B500_r3 --out /workspace/autocheckout-data/agnostic_det
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

from autocheckout.datapaths import Layout
from autocheckout.io import load_json, save_json
from autocheckout.roi_refine import assert_no_overlap, resolve_image_path


def _bbox_coco_to_yolo(bbox: list[float], img_w: float, img_h: float) -> tuple[float, float, float, float] | None:
    x, y, w, h = (float(v) for v in bbox)
    if w <= 1.0 or h <= 1.0 or img_w <= 0 or img_h <= 0:
        return None
    x1 = max(0.0, min(img_w, x))
    y1 = max(0.0, min(img_h, y))
    x2 = max(0.0, min(img_w, x + w))
    y2 = max(0.0, min(img_h, y + h))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None
    xc = (x1 + x2) / (2.0 * img_w)
    yc = (y1 + y2) / (2.0 * img_h)
    return xc, yc, bw / img_w, bh / img_h


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def export_split_yolo(
    coco_sources: list[tuple[dict[str, Any], Path, Iterable[int] | None, str]],
    out_dir: Path,
    split_name: str,
    *,
    allowed_labels: set[int] | None = None,
) -> dict[str, int]:
    """Export one split (``train`` or ``val``) as 1-class (``0 = product``) YOLO images + labels + COCO json."""
    img_dir = out_dir / "images" / split_name
    lbl_dir = out_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_boxes = 0
    coco_images: list[dict[str, Any]] = []
    coco_anns: list[dict[str, Any]] = []
    next_img_id = 1
    next_ann_id = 1

    for coco, img_root, allowed_ids, prefix in coco_sources:
        allowed = None if allowed_ids is None else {int(i) for i in allowed_ids}
        anns_by_img: dict[int, list[dict[str, Any]]] = {}
        for ann in coco["annotations"]:
            img_id = int(ann["image_id"])
            if allowed is not None and img_id not in allowed:
                continue
            if allowed_labels is not None and int(ann["category_id"]) not in allowed_labels:
                continue
            anns_by_img.setdefault(img_id, []).append(ann)

        for im in coco["images"]:
            img_id = int(im["id"])
            if allowed is not None and img_id not in allowed:
                continue
            anns = anns_by_img.get(img_id, [])
            if not anns:
                continue
            src = resolve_image_path(img_root, im["file_name"])
            if not src.is_file():
                continue
            w = float(im["width"])
            h = float(im["height"])
            yolo_lines: list[str] = []
            valid_anns: list[dict[str, Any]] = []
            for ann in anns:
                yb = _bbox_coco_to_yolo(ann["bbox"], w, h)
                if yb is None:
                    continue
                xc, yc, bw, bh = yb
                yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                valid_anns.append(ann)
            if not yolo_lines:
                continue

            stem = f"{prefix}_{img_id}"
            ext = src.suffix or ".jpg"
            dst_img = img_dir / f"{stem}{ext}"
            dst_lbl = lbl_dir / f"{stem}.txt"
            _link_or_copy(src, dst_img)
            dst_lbl.write_text("\n".join(yolo_lines) + "\n")

            coco_images.append({
                "id": next_img_id,
                "orig_id": img_id,
                "source": prefix,
                "file_name": f"{split_name}/{stem}{ext}",
                "width": int(w),
                "height": int(h),
            })
            for ann in valid_anns:
                coco_anns.append({
                    "id": next_ann_id,
                    "image_id": next_img_id,
                    "category_id": 0,
                    "bbox": [float(v) for v in ann["bbox"]],
                    "area": float(ann["bbox"][2] * ann["bbox"][3]),
                    "iscrowd": 0,
                })
                next_ann_id += 1
            next_img_id += 1
            n_images += 1
            n_boxes += len(yolo_lines)

    coco_out = {
        "images": coco_images,
        "annotations": coco_anns,
        "categories": [{"id": 0, "name": "product", "supercategory": "product"}],
    }
    save_json(out_dir / f"{split_name}_agnostic_coco.json", coco_out)
    return {"images": n_images, "boxes": n_boxes}


def build_agnostic_dataset(
    *,
    val_full: dict[str, Any],
    val_img_root: Path,
    train_real_ids: list[int],
    calib_ids: list[int],
    valb_ids: list[int],
    out_dir: Path,
    synth_sources: list[tuple[dict[str, Any], Path, str]] | None = None,
    allowed_labels: set[int] | None = None,
) -> dict[str, Any]:
    """Build a 1-class (``0 = product``) YOLO/RT-DETR dataset with strict anti-leakage checks.

    ``train`` uses ``synth_sources`` + ``train_real_ids``; ``val`` uses ``calib_ids`` (never Val-B).
    """
    assert_no_overlap(train_real_ids, calib_ids, label="Val-A-calib")
    assert_no_overlap(train_real_ids, valb_ids, label="Val-B")

    train_sources: list[tuple[dict[str, Any], Path, Iterable[int] | None, str]] = [
        (val_full, val_img_root, train_real_ids, "real_vala"),
    ]
    for s_coco, s_root, s_prefix in (synth_sources or []):
        synth_ids = [int(im.get("orig_id", im["id"])) for im in s_coco["images"]]
        assert_no_overlap(synth_ids, calib_ids, label=f"Val-A-calib ({s_prefix})")
        assert_no_overlap(synth_ids, valb_ids, label=f"Val-B ({s_prefix})")
        train_sources.append((s_coco, s_root, None, s_prefix))

    out_dir.mkdir(parents=True, exist_ok=True)
    train_stats = export_split_yolo(train_sources, out_dir, "train", allowed_labels=allowed_labels)
    val_stats = export_split_yolo(
        [(val_full, val_img_root, calib_ids, "calib")], out_dir, "val", allowed_labels=None
    )

    yaml_text = (
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n"
        "  0: product\n"
    )
    (out_dir / "dataset.yaml").write_text(yaml_text)
    summary = {
        "out_dir": str(out_dir.resolve()),
        "dataset_yaml": str((out_dir / "dataset.yaml").resolve()),
        "train": train_stats,
        "val_calib": val_stats,
        "train_real_images_requested": len(train_real_ids),
        "val_calib_images_requested": len(calib_ids),
        "val_b_overlap": 0,
    }
    save_json(out_dir / "summary.json", summary, indent=2)
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, help="autocheckout-data folder (default: $AC_DATA)")
    ap.add_argument("--img-root", type=Path, help="image root containing val2019")
    ap.add_argument("--selection", type=str, default="margin_k_B500_r3",
                    help="AL selection under relearn/active/selections (empty = full Val-A pool)")
    ap.add_argument("--include-synth", action="store_true",
                    help="also include synth_v2 task JSONs (all 5 tasks, or only task 1 with --task1-only)")
    ap.add_argument("--task1-only", action="store_true",
                    help="restrict training annotations to Task 1 classes (0..99) for strict CL Task-1-only Stage 1")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    data = Layout.at(args.data)
    img_root = args.img_root or data.rpc
    if not (img_root / "val2019").is_dir() and (data.root / "archive" / "val2019").is_dir():
        img_root = data.root / "archive"

    val_full = load_json(data.eval / "val_full.json")
    split = load_json(data.relearn / "active/splits/val_a_split.json")
    calib_ids = sorted(int(i) for i in split["calib"])
    valb_ids = sorted(int(im["id"]) for im in load_json(data.relearn / "meta/val_b.json")["images"])

    if args.selection:
        sel_path = Path(args.selection)
        if not sel_path.is_file():
            sel_path = data.relearn / f"active/selections/{args.selection}.json"
        sel = load_json(sel_path)
        train_ids = sorted(set(int(i) for i in sel.get("selected", sel.get("image_ids", []))))
    else:
        train_ids = sorted(set(int(i) for i in split["pool"]))

    synth_sources: list[tuple[dict[str, Any], Path, str]] = []
    if args.include_synth and data.synth.is_dir():
        tasks = [1] if args.task1_only else [1, 2, 3, 4, 5]
        for t in tasks:
            cand = data.synth / f"train_task_{t}.json"
            if cand.is_file():
                synth_sources.append((load_json(cand), data.synth, f"synth_t{t}"))

    allowed_labels = set(range(100)) if args.task1_only else None
    summary = build_agnostic_dataset(
        val_full=val_full,
        val_img_root=img_root,
        train_real_ids=train_ids,
        calib_ids=calib_ids,
        valb_ids=valb_ids,
        out_dir=args.out,
        synth_sources=synth_sources,
        allowed_labels=allowed_labels,
    )
    print(f"[AgnosticDet] Built 1-class dataset: train={summary['train']} | val_calib={summary['val_calib']}", flush=True)
    return summary


if __name__ == "__main__":
    main()
