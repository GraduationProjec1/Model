"""A2 (AN/cyclegan/03_QUYET_DINH_VA_PLAN.md): camera weights per SKU, measured on Val-A.

match: every Val-A object (val_full minus Val-B) of each SKU is compared with ``--per-camera`` cutouts of each
       camera, pasted on a tray background; ConvNeXt-V2 cosine, best of 4 quarter turns. The camera of the
       best-matching cutout wins the object. Writes per-SKU counts and every object's best similarity per camera.
write: counts -> skus.json (autocheckout.camera_weights.build_skus): shrunk to the supercategory mean with a
       fitted strength, every camera with cutouts floored at 3 %, upright SKUs get standing_cameras [2]
       (shape groups from the team's table, AN/cyclegan/skus_cu/skus.json).

    python -m tools.camera_weights match --catalog <cutouts.json> --out <dir>/match.json [--per-camera 10]
    python -m tools.camera_weights write --match <dir>/match.json --shapes AN/cyclegan/skus_cu/skus.json \\
        --out <dir>/skus.json
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from autocheckout.camera_weights import build_skus
from autocheckout.io import load_json, save_json
from autocheckout.taskcfg import TaskConfig

DATA = Path(os.environ.get("AC_DATA", "../autocheckout-data"))


def _cutouts(path: Path) -> list[dict]:
    """Catalog records (``cutouts`` of tools.synth_v2 inventory) or a raw SAM index (``objects``)."""
    data = load_json(path)
    if "cutouts" in data:
        return [{"file": c["file"], "camera": c["camera"], "rpc_category_id": c["rpc_category_id"]} for c in data["cutouts"]]
    root = path.parent.parent.parent       # index at <root>/cutouts/<set>/index.json, files relative to <root>
    return [{"file": str(root / o["file"]), "camera": o["camera"], "rpc_category_id": o["category_id"]}
            for o in data["objects"]]


def cmd_match(args) -> None:
    import timm
    import torch
    from PIL import Image

    rng = random.Random(args.seed)
    val = load_json(DATA / "eval/val_full.json")
    val_b = {im["id"] for im in load_json(DATA / "relearn/meta/val_b.json")["images"]}
    label_to_rpc = TaskConfig.load(DATA / "eval/task_config.json").label_to_rpc()
    images = {im["id"]: im for im in val["images"] if im["id"] not in val_b}
    real = defaultdict(list)
    for a in val["annotations"]:
        if a["image_id"] in images:
            real[label_to_rpc[a["category_id"]]].append(a)
    cuts = defaultdict(lambda: defaultdict(list))
    for c in _cutouts(Path(args.catalog)):
        cuts[c["rpc_category_id"]][c["camera"]].append(c)
    backgrounds = [p for p in sorted(Path(args.backgrounds).glob("*.jpg"))
                   if len(set(Image.open(p).size)) == 1]          # square trays only (one background is 1440x720)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k", pretrained=True, num_classes=0).eval().to(device)
    transform = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))

    def embed(pils):
        out = []
        with torch.no_grad():
            for i in range(0, len(pils), 64):
                f = model(torch.stack([transform(p) for p in pils[i:i + 64]]).to(device)).cpu().numpy()
                out.append(f / np.linalg.norm(f, axis=1, keepdims=True))
        return np.concatenate(out)

    def square(img):
        w, h = img.size
        s = max(w, h)
        sq = Image.new("RGB", (s, s), (128, 128, 128))
        sq.paste(img, ((s - w) // 2, (s - h) // 2))
        return sq

    bg_cache = {}

    def on_tray(c):
        cut = Image.open(c["file"]).convert("RGBA")
        w, h = cut.size
        b = backgrounds[rng.randrange(len(backgrounds))]
        if b not in bg_cache:
            bg_cache[b] = Image.open(b).convert("RGB")
        tray, s = bg_cache[b], int(max(w, h) * 1.1)
        x0, y0 = rng.randrange(0, max(1, tray.width - s)), rng.randrange(0, max(1, tray.height - s))
        tile = tray.crop((x0, y0, x0 + s, y0 + s)).convert("RGBA")
        tile.alpha_composite(cut, ((s - w) // 2, (s - h) // 2))
        return tile.convert("RGB")

    out = {"per_camera": args.per_camera, "catalog": args.catalog, "seed": args.seed, "skus": {}}
    if args.resume:                       # chunked runs (MPS memory growth): one seed per SKU, skip SKUs already done
        out["seed_per_sku"] = True
        if Path(args.out).exists():
            prev = load_json(Path(args.out))
            assert prev.get("seed_per_sku") and prev["catalog"] == args.catalog and prev["per_camera"] == args.per_camera
            out["skus"] = prev["skus"]
    open_images = {}
    todo = [s for s in sorted(real)[:args.limit or None] if str(s) not in out["skus"]]
    for k, sku in enumerate(todo):
        if args.max_skus and k >= args.max_skus:
            print(f"chunk done, {len(todo) - k} SKUs left", flush=True)
            return
        if args.resume:
            rng = random.Random(f"{args.seed}:{sku}")
        crops = []
        for a in real[sku]:
            im_id = a["image_id"]
            if im_id not in open_images:
                open_images.clear()                       # annotations come grouped by image: keep one open
                open_images[im_id] = Image.open(Path(args.image_root) / images[im_id]["file_name"]).convert("RGB")
            x, y, w, h = a["bbox"]
            pad = 0.05 * max(w, h)
            crops.append(square(open_images[im_id].crop((int(x - pad), int(y - pad), int(x + w + pad), int(y + h + pad)))))
        real_emb = embed(crops)
        best = {}
        for cam in sorted(cuts[sku]):
            chosen = rng.sample(cuts[sku][cam], min(args.per_camera, len(cuts[sku][cam])))
            tiles = [on_tray(c) for c in chosen]
            cut_emb = embed([t.rotate(r, expand=True) for t in tiles for r in (0, 90, 180, 270)])
            best[cam] = (real_emb @ cut_emb.T).max(1)
        cams = sorted(best)
        winner = np.array(cams)[np.stack([best[c] for c in cams]).argmax(0)] if cams else np.array([])
        out["skus"][str(sku)] = {"counts": {str(c): int((winner == c).sum()) for c in cams},
                                 # val box area and best camera of every object: input of fit_scales(matches=...)
                                 "matches": [[round(a["bbox"][2] * a["bbox"][3], 1), int(c)]
                                             for a, c in zip(real[sku], winner)],
                                 "n_cutouts": {str(c): len(cuts[sku][c]) for c in cams},
                                 "best": {str(c): np.round(best[c], 4).tolist() for c in cams}}
        del crops, real_emb, best
        if device == "mps":                   # MPS keeps freed buffers cached: footprint grew to 14 GB by SKU 22 (24/11)
            torch.mps.empty_cache()
        print(f"SKU {sku}: {len(out['skus'][str(sku)]['matches'])} objects, counts {out['skus'][str(sku)]['counts']}", flush=True)
        if args.resume:
            save_json(Path(args.out + ".tmp"), out)
            os.replace(args.out + ".tmp", args.out)
    save_json(Path(args.out), out)
    print(f"ALL SKUS DONE ({len(out['skus'])})", flush=True)


def cmd_write(args) -> None:
    match = load_json(args.match)["skus"]
    config = TaskConfig.load(DATA / "eval/task_config.json")
    groups = {c.rpc_category_id: c.supercategory for t in config.data_tasks for c in t.classes}
    shapes = {s["rpc_category_id"]: s["shape_group"] for s in load_json(args.shapes)["skus"]}
    counts = {int(s): {int(c): n for c, n in v["counts"].items()} for s, v in match.items()}
    available = {int(s): {int(c) for c, n in v["n_cutouts"].items() if n > 0} for s, v in match.items()}
    table = build_skus(counts, {s: groups[s] for s in counts}, {s: shapes[s] for s in counts}, available,
                       floor=args.floor)
    table.update(match=str(args.match), shapes=str(args.shapes))
    save_json(Path(args.out), table, indent=1)
    print(f"{len(table['skus'])} SKUs, fitted strength {table['strength']:.2f}, floor {args.floor}; wrote {args.out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("match")
    m.add_argument("--catalog", required=True, help="cutouts.json (catalog) or a SAM index.json")
    m.add_argument("--image-root", default="data/archive")
    m.add_argument("--backgrounds", default="data/archive/backgrounds")
    m.add_argument("--per-camera", type=int, default=10)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--limit", type=int, default=0, help="first N SKUs only (smoke test)")
    m.add_argument("--out", required=True)
    m.add_argument("--resume", action="store_true", help="seed per SKU, save after every SKU, skip SKUs in --out")
    m.add_argument("--max-skus", type=int, default=0, help="with --resume: stop after this many new SKUs")
    w = sub.add_parser("write")
    w.add_argument("--match", type=Path, required=True)
    w.add_argument("--shapes", type=Path, required=True, help="team skus.json holding shape_group")
    w.add_argument("--floor", type=float, default=0.03)
    w.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"match": cmd_match, "write": cmd_write}[args.cmd](args)


if __name__ == "__main__":
    main()
