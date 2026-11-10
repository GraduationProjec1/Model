"""Cut again the v2 cutouts that are missing on disk (sets other than sam_v1), from their exact source annotations.

Reuses crop/clean/finalize/SAM of data/archive/scripts/02_extract_objects.py with the parameters of
cutouts/sam/index.json. Cutouts that fail QC are kept (flag ``qc_fail``) so the catalog keeps the v2 ids.
Resumable: PNGs already in the set's index.json are skipped.

    python -m tools.recut_v2_cutouts --sets cam0_v1 cam2s_v1 cam2s_v2 cam0s_v2 --sam-model sam2.1_b.pt
    python -m tools.recut_v2_cutouts --sets repl_v1 --sam-model sam2.1_l.pt
"""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

SCRIPT = "data/archive/scripts/02_extract_objects.py"
QC = {"pad": 0.10, "clip_margin": 0.01, "max_side": 1000, "sam_size": 1024, "min_fill": 0.30, "max_fill": 0.98, "min_iou": 0.90}


def load_extractor():
    spec = importlib.util.spec_from_file_location("extract_objects", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--ids", default="data/synth_v2_12x_build/checks/catalog_ids_only.json")
    ap.add_argument("--out-root", default="data/synth_v2_12x_build")
    ap.add_argument("--image-root", default="data/archive")
    ap.add_argument("--sam-model", default="sam2.1_b.pt")
    ap.add_argument("--device")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    ex = load_extractor()
    data = json.load(open(os.path.join(a.image_root, "instances_train2019.json")))
    imgs = {i["id"]: i for i in data["images"]}
    anns = {x["id"]: x for x in data["annotations"]}
    cats = {c["id"]: c["name"] for c in data["categories"]}
    wanted = [c for c in json.load(open(a.ids))["cutouts"] if c["cutout_set"] in a.sets]
    seg = ex.SamSegmenter(argparse.Namespace(device=a.device, sam_model=a.sam_model, sam_size=QC["sam_size"]))

    for cutout_set in a.sets:
        out = Path(a.out_root) / f"cutouts_{cutout_set}"
        idx_path = out / "index.json"
        done = {}
        if idx_path.exists():
            done = {o["ann_id"]: o for o in json.load(open(idx_path))["objects"] if (Path(a.out_root) / o["file"]).is_file()}
        jobs = [c for c in wanted if c["cutout_set"] == cutout_set and c["ann_id"] not in done][: a.limit]
        print(f"{cutout_set}: {len(done)} done, {len(jobs)} to cut -> {out}", flush=True)

        def save():
            objs = sorted(done.values(), key=lambda r: (r["category_id"], r["file"]))
            meta = {"method": "sam", "params": {**QC, "sam_model": seg.name, "device": seg.device,
                    "source": "recut of v2 ids (tools/recut_v2_cutouts.py)", "qc_policy": "keep, flag qc_fail"}, "objects": objs}
            tmp = idx_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
            os.replace(tmp, idx_path)

        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for i, c in enumerate(jobs, 1):
            ann = anns[c["ann_id"]]
            im = imgs[ann["image_id"]]
            cid = ann["category_id"]
            bbox = [round(v, 2) for v in ann["bbox"]]
            img = cv2.imread(os.path.join(a.image_root, "train2019", im["file_name"]))
            if img is None:
                raise SystemExit(f"cannot read {im['file_name']}")
            crop, off = ex.crop_region(img, bbox, QC["pad"])
            box = [bbox[0] - off[0], bbox[1] - off[1], bbox[2], bbox[3]]
            rgba, q = ex.finalize(crop, off, seg(crop, box), bbox, QC["clip_margin"])
            reason = ("empty_mask" if rgba is None else "fill" if not (QC["min_fill"] <= q["fill"] <= QC["max_fill"])
                      else "iou" if q["iou"] < QC["min_iou"] else None)
            if rgba is None:  # keep the id: fall back to the whole bbox crop so the catalog stays complete
                x0, y0 = int(box[0]), int(box[1])
                rgba = cv2.cvtColor(crop[y0:y0 + int(box[3]), x0:x0 + int(box[2])], cv2.COLOR_BGR2BGRA)
            if max(rgba.shape[:2]) > QC["max_side"]:
                f = QC["max_side"] / max(rgba.shape[:2])
                rgb = cv2.resize(rgba[:, :, :3], None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                al = cv2.resize(rgba[:, :, 3], (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                rgba = np.dstack([rgb, np.where(al > 127, 255, 0).astype(np.uint8)])
            rel = f"cutouts_{cutout_set}/objects/{ex.folder_name(cid, cats[cid])}/{os.path.splitext(im['file_name'])[0]}.png"
            dst = Path(a.out_root) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), rgba)
            done[c["ann_id"]] = {"category_id": cid, "category": cats[cid], "ann_id": c["ann_id"], "source": im["file_name"],
                                 "camera": ex.camera_of(im["file_name"]), "bbox": bbox, "file": rel, **q,
                                 "size": [rgba.shape[1], rgba.shape[0]], "model": seg.name, "qc_fail": reason}
            if i % 25 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {cutout_set} {i}/{len(jobs)}  {el:.0f}s  ~{el / i * (len(jobs) - i) / 60:.0f} min left", flush=True)
                save()
        save()
        fails = sum(1 for o in done.values() if o["qc_fail"])
        print(f"{cutout_set}: {len(done)} cutouts, qc_fail {fails}", flush=True)


if __name__ == "__main__":
    main()
