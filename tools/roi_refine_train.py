"""M6.2 RoI-Refine trainer: trains a high-res crop classifier on real Val-A pool (or AL selection) GT boxes,
strictly checking zero overlap with Val-A-calib and Val-B.

Example:
    python -m tools.roi_refine_train --out data/training/update-model/m6_roi \\
        --selection margin_k_B500_r3 --arch convnext_tiny --epochs 12
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from autocheckout.datapaths import Layout
from autocheckout.io import load_json, save_json
from autocheckout.roi_refine import (
    RoICropDataset,
    assert_no_overlap,
    build_crop_records,
    build_roi_classifier,
)
from autocheckout.taskcfg import TaskConfig


def resolve_train_ids(rel_dir: Path, selection: str | None) -> list[int]:
    split = load_json(rel_dir / "active/splits/val_a_split.json")
    calib = set(int(i) for i in split["calib"])
    val_b = {int(im["id"]) for im in load_json(rel_dir / "meta/val_b.json")["images"]}
    if selection:
        sel_path = Path(selection)
        if not sel_path.is_file():
            sel_path = rel_dir / f"active/selections/{selection}.json"
        sel = load_json(sel_path)
        train_ids = [int(i) for i in sel.get("selected", sel.get("image_ids", []))]
    else:
        train_ids = [int(i) for i in split["pool"]]
    assert_no_overlap(train_ids, calib, label="Val-A-calib")
    assert_no_overlap(train_ids, val_b, label="Val-B")
    return sorted(set(train_ids))


def train_classifier(
    records,
    *,
    num_classes: int,
    arch: str = "convnext_tiny",
    pretrained: bool = True,
    img_size: int = 224,
    pad_ratio: float = 0.10,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
    num_workers: int = 4,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    ds = RoICropDataset(records, img_size=img_size, pad_ratio=pad_ratio, train=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(dev.type == "cuda"))

    model = build_roi_classifier(arch=arch, num_classes=num_classes, pretrained=pretrained).to(dev)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    use_amp = dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    counts_per_class = [0] * num_classes
    for rec in records:
        if 0 <= rec.label < num_classes:
            counts_per_class[rec.label] += 1
    covered_classes = sum(1 for c in counts_per_class if c > 0)
    missing_classes = [i for i, c in enumerate(counts_per_class) if c == 0]

    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for imgs, targets in loader:
            imgs = imgs.to(dev, non_blocking=True)
            targets = targets.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = targets.size(0)
            total_loss += float(loss.item()) * bs
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += bs
        scheduler.step()
        ep_info = {
            "epoch": epoch + 1,
            "loss": round(total_loss / max(1, total), 5),
            "train_acc": round(correct / max(1, total), 5),
            "lr": round(optimizer.param_groups[0]["lr"], 7),
        }
        history.append(ep_info)
        print(f"[RoI-Refine] Epoch {epoch + 1}/{epochs} | loss={ep_info['loss']:.4f} | acc={ep_info['train_acc']*100:.2f}% | lr={ep_info['lr']:.2e}", flush=True)

    summary = {
        "arch": arch,
        "num_classes": num_classes,
        "img_size": img_size,
        "pad_ratio": pad_ratio,
        "num_crops": len(records),
        "classes_covered": covered_classes,
        "missing_classes_count": len(missing_classes),
        "missing_classes_sample": missing_classes[:20],
        "min_crops_per_class": min(counts_per_class) if counts_per_class else 0,
        "epochs": epochs,
        "train_seconds": round(time.time() - t0, 2),
        "final_loss": history[-1]["loss"] if history else None,
        "final_train_acc": history[-1]["train_acc"] if history else None,
        "history": history,
    }
    return model, summary


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, help="autocheckout-data folder (default: $AC_DATA)")
    ap.add_argument("--img-root", type=Path, help="image root containing val2019 (default: <data>/rpc)")
    ap.add_argument("--selection", type=str, default="margin_k_B500_r3",
                    help="AL selection name or path under relearn/active/selections (empty = full Val-A pool)")
    ap.add_argument("--synth-json", type=Path, help="optional synthetic COCO json for class-coverage replay crops")
    ap.add_argument("--synth-img-root", type=Path, help="image root for --synth-json")
    ap.add_argument("--max-synth-per-class", type=int, default=60)
    ap.add_argument("--stage", type=int, default=5)
    ap.add_argument("--arch", default="convnext_tiny", choices=("convnext_tiny", "resnet18", "resnet50", "tiny_test"))
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--pad-ratio", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    data = Layout.at(args.data)
    img_root = args.img_root or (data.root / "rpc")
    if not (img_root / "val2019").is_dir() and (data.root / "archive" / "val2019").is_dir():
        img_root = data.root / "archive"

    val_full = load_json(data.eval / "val_full.json")
    cfg = TaskConfig.load(data.eval / "task_config.json")
    seen_classes = cfg.seen_classes(args.stage)
    train_ids = resolve_train_ids(data.relearn, args.selection or None)
    records = build_crop_records(val_full, img_root, train_ids, num_classes=seen_classes, seed=args.seed)

    synth_count = 0
    if args.synth_json and args.synth_json.is_file():
        synth_coco = load_json(args.synth_json)
        split = load_json(data.relearn / "active/splits/val_a_split.json")
        calib_ids = {int(i) for i in split["calib"]}
        val_b_ids = {int(im["id"]) for im in load_json(data.relearn / "meta/val_b.json")["images"]}
        synth_img_ids = [int(im.get("orig_id", im["id"])) for im in synth_coco["images"]]
        assert_no_overlap(synth_img_ids, calib_ids, label="Val-A-calib (synth-json)")
        assert_no_overlap(synth_img_ids, val_b_ids, label="Val-B (synth-json)")
        s_root = args.synth_img_root or args.synth_json.parent
        synth_recs = build_crop_records(
            synth_coco, s_root, None, num_classes=seen_classes, max_per_class=args.max_synth_per_class, seed=args.seed
        )
        synth_count = len(synth_recs)
        records.extend(synth_recs)

    print(f"[RoI-Refine] Training {args.arch} on {len(records)} crops (real={len(records)-synth_count} from {len(train_ids)} images, synth={synth_count})", flush=True)
    model, summary = train_classifier(
        records,
        num_classes=seen_classes,
        arch=args.arch,
        pretrained=not args.no_pretrained,
        img_size=args.img_size,
        pad_ratio=args.pad_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / "roi_classifier.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "arch": args.arch,
        "num_classes": seen_classes,
        "img_size": args.img_size,
        "pad_ratio": args.pad_ratio,
        "selection": args.selection,
        "train_images": len(train_ids),
    }, ckpt_path)
    summary["checkpoint"] = str(ckpt_path)
    summary["real_crops"] = len(records) - synth_count
    summary["synth_crops"] = synth_count
    summary["train_images"] = len(train_ids)
    save_json(args.out / "train_summary.json", summary, indent=2)
    return summary


if __name__ == "__main__":
    main()
