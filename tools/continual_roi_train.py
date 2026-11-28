"""Stage 2 Continual Crop Recognizer trainer: sequential multi-task learning (Task 1 -> Task N).

Supports two CL modes:
- ``prototype`` (Mode A): train backbone + Cosine/ArcFace head on Task 1, freeze backbone, and for each
  subsequent task compute L2-normalized class prototypes and append to the Cosine head (~0% forgetting).
- ``replay_kd`` (Mode B): maintain an iCaRL herding exemplar bank (``exemplars_per_class`` crops/SKU)
  and fine-tune each new task with Cosine/ArcFace CE + temperature-scaled Logit KD on old classes.

Example:
    python -m tools.continual_roi_train --data /workspace/autocheckout-data \\
        --mode replay_kd --arch convnext_tiny --exemplars-per-class 20 \\
        --out data/training/update-model/decoupled_cl/stage2_replay
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from autocheckout.continual_roi import (
    ContinualCropModel,
    compute_cl_metrics,
    compute_prototypes_from_embeddings,
    extract_embeddings,
    kd_logit_loss,
    select_exemplars_herding,
)
from autocheckout.datapaths import Layout
from autocheckout.io import load_json, save_json
from autocheckout.roi_refine import CropRecord, RoICropDataset, assert_no_overlap, build_crop_records
from autocheckout.taskcfg import TaskConfig


@torch.no_grad()
def eval_crop_accuracy(
    model: ContinualCropModel,
    records: Sequence[CropRecord],
    *,
    img_size: int = 224,
    pad_ratio: float = 0.10,
    batch_size: int = 64,
    num_workers: int = 0,
    device: str = "cpu",
) -> float:
    """Top-1 classification accuracy over foreground classes on ``records``."""
    if not records:
        return float("nan")
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = model.to(dev).eval()
    ds = RoICropDataset(records, img_size=img_size, pad_ratio=pad_ratio, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    correct = 0
    total = 0
    fg = model.num_fg_classes
    for imgs, targets in loader:
        logits = model(imgs.to(dev))[:, :fg]
        preds = logits.argmax(dim=1).cpu()
        correct += int((preds == targets).sum().item())
        total += len(targets)
    return correct / max(1, total)


def _train_epochs(
    model: ContinualCropModel,
    records: Sequence[CropRecord],
    *,
    epochs: int,
    batch_size: int,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
    label_smoothing: float,
    img_size: int,
    pad_ratio: float,
    num_workers: int,
    device: str,
    teacher: ContinualCropModel | None = None,
    num_old_classes: int = 0,
    kd_lambda: float = 1.0,
    kd_temp: float = 2.0,
) -> list[dict]:
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = model.to(dev)
    if teacher is not None:
        teacher = teacher.to(dev).eval()

    ds = RoICropDataset(records, img_size=img_size, pad_ratio=pad_ratio, train=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(dev.type == "cuda"))

    param_groups = []
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if bb_params:
        param_groups.append({"params": bb_params, "lr": lr_backbone})
    param_groups.append({"params": model.head.parameters(), "lr": lr_head})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    use_amp = dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: list[dict] = []
    for ep in range(epochs):
        model.train()
        tot_loss = 0.0
        tot_kd = 0.0
        correct = 0
        total = 0
        for imgs, targets in loader:
            imgs = imgs.to(dev, non_blocking=True)
            targets = targets.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs, targets=targets)
                loss = criterion(logits, targets)
                kd_val = 0.0
                if teacher is not None and num_old_classes > 0 and kd_lambda > 0:
                    with torch.no_grad():
                        t_logits = teacher(imgs)
                    loss_kd = kd_logit_loss(logits, t_logits, num_old_classes, temp=kd_temp)
                    loss = loss + kd_lambda * loss_kd
                    kd_val = float(loss_kd.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = targets.size(0)
            tot_loss += float(loss.item()) * bs
            tot_kd += kd_val * bs
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += bs
        scheduler.step()
        history.append({
            "epoch": ep + 1,
            "loss": round(tot_loss / max(1, total), 5),
            "kd_loss": round(tot_kd / max(1, total), 5),
            "train_acc": round(correct / max(1, total), 5),
        })
    return history


def run_continual_training(
    task_train_records: Sequence[Sequence[CropRecord]],
    task_eval_records: Sequence[Sequence[CropRecord]],
    task_class_lists: Sequence[Sequence[int]],
    *,
    mode: str = "replay_kd",
    arch: str = "convnext_tiny",
    pretrained: bool = True,
    include_background: bool = False,
    bg_records: Sequence[CropRecord] = (),
    img_size: int = 224,
    pad_ratio: float = 0.10,
    epochs_task1: int = 12,
    epochs_inc: int = 8,
    batch_size: int = 64,
    lr_task1: float = 3e-4,
    lr_inc_backbone: float = 5e-5,
    lr_inc_head: float = 3e-4,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
    scale: float = 20.0,
    margin: float = 0.2,
    exemplars_per_class: int = 20,
    kd_lambda: float = 1.0,
    kd_temp: float = 2.0,
    num_workers: int = 0,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[ContinualCropModel, dict]:
    """Run sequential continual learning across ``T`` tasks and return ``(model, summary)`` with
    the ``T x T`` task accuracy matrix and ``ACC / Forgetting / BWT`` metrics."""
    if mode not in ("prototype", "replay_kd"):
        raise ValueError(f"mode must be 'prototype' or 'replay_kd', got {mode!r}")
    torch.manual_seed(seed)
    t0 = time.time()

    n_tasks = len(task_class_lists)
    model = ContinualCropModel(
        arch=arch,
        num_classes=len(task_class_lists[0]),
        include_background=include_background,
        scale=scale,
        margin=margin,
        pretrained=pretrained,
    )

    acc_matrix: list[list[float | None]] = [[None] * n_tasks for _ in range(n_tasks)]
    exemplar_bank: list[CropRecord] = []
    task_logs: list[dict] = []

    for t_idx in range(n_tasks):
        cur_classes = list(task_class_lists[t_idx])
        cur_records = list(task_train_records[t_idx])
        num_old = model.num_fg_classes if t_idx > 0 else 0

        if t_idx == 0:
            train_recs = cur_records + (list(bg_records) if include_background else [])
            hist = _train_epochs(
                model, train_recs, epochs=epochs_task1, batch_size=batch_size,
                lr_backbone=lr_task1, lr_head=lr_task1, weight_decay=weight_decay,
                label_smoothing=label_smoothing, img_size=img_size, pad_ratio=pad_ratio,
                num_workers=num_workers, device=device,
            )
            if mode == "prototype":
                for p in model.backbone.parameters():
                    p.requires_grad = False
            elif exemplars_per_class > 0:
                embs, _ = extract_embeddings(
                    model, cur_records, img_size=img_size, pad_ratio=pad_ratio,
                    batch_size=batch_size, num_workers=num_workers, device=device,
                )
                exemplar_bank.extend(select_exemplars_herding(cur_records, embs, per_class=exemplars_per_class))
        else:
            embs, lbls = extract_embeddings(
                model, cur_records, img_size=img_size, pad_ratio=pad_ratio,
                batch_size=batch_size, num_workers=num_workers, device=device,
            )
            new_protos = compute_prototypes_from_embeddings(embs, lbls, cur_classes)
            if mode == "prototype":
                model.expand_fg_classes(new_protos)
                hist = []
            else:
                teacher = copy.deepcopy(model).eval()
                for p in teacher.parameters():
                    p.requires_grad = False
                model.expand_fg_classes(new_protos)
                # Remap bg_records label to new bg_class_index if background slot shifted
                remapped_bg = (
                    [CropRecord(r.image_path, r.box_xyxy, model.bg_class_index, r.image_id) for r in bg_records]
                    if include_background and model.bg_class_index is not None else []
                )
                train_recs = cur_records + exemplar_bank + remapped_bg
                hist = _train_epochs(
                    model, train_recs, epochs=epochs_inc, batch_size=batch_size,
                    lr_backbone=lr_inc_backbone, lr_head=lr_inc_head, weight_decay=weight_decay,
                    label_smoothing=label_smoothing, img_size=img_size, pad_ratio=pad_ratio,
                    num_workers=num_workers, device=device,
                    teacher=teacher, num_old_classes=num_old, kd_lambda=kd_lambda, kd_temp=kd_temp,
                )
                if exemplars_per_class > 0:
                    embs_after, _ = extract_embeddings(
                        model, cur_records, img_size=img_size, pad_ratio=pad_ratio,
                        batch_size=batch_size, num_workers=num_workers, device=device,
                    )
                    exemplar_bank.extend(
                        select_exemplars_herding(cur_records, embs_after, per_class=exemplars_per_class)
                    )

        for k in range(t_idx + 1):
            acc_matrix[t_idx][k] = round(
                eval_crop_accuracy(
                    model, task_eval_records[k], img_size=img_size, pad_ratio=pad_ratio,
                    batch_size=batch_size, num_workers=num_workers, device=device,
                ),
                5,
            )
        task_logs.append({
            "task": t_idx + 1,
            "new_classes": len(cur_classes),
            "seen_fg_classes": model.num_fg_classes,
            "train_crops": len(cur_records),
            "exemplars_stored": len(exemplar_bank),
            "task_accs": acc_matrix[t_idx][: t_idx + 1],
            "history": hist,
        })
        print(
            f"[ContinualRoI:{mode}] Task {t_idx + 1}/{n_tasks} (seen={model.num_fg_classes}) | "
            f"eval_accs={[round(a * 100, 2) for a in acc_matrix[t_idx][: t_idx + 1]]}",
            flush=True,
        )

    cl_metrics = compute_cl_metrics(acc_matrix)
    summary = {
        "mode": mode,
        "arch": arch,
        "include_background": include_background,
        "img_size": img_size,
        "pad_ratio": pad_ratio,
        "scale": scale,
        "margin": margin,
        "exemplars_per_class": exemplars_per_class,
        "num_tasks": n_tasks,
        "final_fg_classes": model.num_fg_classes,
        "acc_matrix": acc_matrix,
        "cl_metrics": cl_metrics,
        "train_seconds": round(time.time() - t0, 2),
        "tasks": task_logs,
    }
    return model, summary


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, help="autocheckout-data folder (default: $AC_DATA)")
    ap.add_argument("--img-root", type=Path, help="image root containing val2019")
    ap.add_argument("--selection", type=str, default="margin_k_B500_r3",
                    help="AL selection under relearn/active/selections (empty = full Val-A pool)")
    ap.add_argument("--stage", type=int, default=5, help="number of sequential tasks to learn (default 5)")
    ap.add_argument("--mode", choices=("prototype", "replay_kd"), default="replay_kd")
    ap.add_argument("--arch", default="convnext_tiny",
                    choices=("convnext_tiny", "resnet18", "resnet50", "dinov2_vits14", "tiny_test"))
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--pad-ratio", type=float, default=0.10)
    ap.add_argument("--epochs-task1", type=int, default=12)
    ap.add_argument("--epochs-inc", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--exemplars-per-class", type=int, default=20)
    ap.add_argument("--kd-lambda", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    data = Layout.at(args.data)
    img_root = args.img_root or data.rpc
    if not (img_root / "val2019").is_dir() and (data.root / "archive" / "val2019").is_dir():
        img_root = data.root / "archive"

    val_full = load_json(data.eval / "val_full.json")
    cfg = TaskConfig.load(data.eval / "task_config.json")
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

    assert_no_overlap(train_ids, calib_ids, label="Val-A-calib")
    assert_no_overlap(train_ids, valb_ids, label="Val-B")

    seen = cfg.seen_classes(args.stage)
    all_train = build_crop_records(val_full, img_root, train_ids, num_classes=seen, seed=args.seed)
    all_calib = build_crop_records(val_full, img_root, calib_ids, num_classes=seen, seed=args.seed)

    task_class_lists = [sorted(c.label for c in cfg.data_tasks[t].classes) for t in range(args.stage)]
    task_train_records = [[r for r in all_train if r.label in set(cls)] for cls in task_class_lists]
    task_eval_records = [[r for r in all_calib if r.label in set(cls)] for cls in task_class_lists]

    model, summary = run_continual_training(
        task_train_records, task_eval_records, task_class_lists,
        mode=args.mode, arch=args.arch, pretrained=not args.no_pretrained,
        img_size=args.img_size, pad_ratio=args.pad_ratio,
        epochs_task1=args.epochs_task1, epochs_inc=args.epochs_inc,
        batch_size=args.batch_size, exemplars_per_class=args.exemplars_per_class,
        kd_lambda=args.kd_lambda, num_workers=args.num_workers,
        device=args.device, seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / "continual_roi.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "arch": args.arch,
        "num_fg_classes": model.num_fg_classes,
        "include_background": model.include_background,
        "scale": model.head.scale,
        "margin": model.head.margin,
        "img_size": args.img_size,
        "pad_ratio": args.pad_ratio,
        "mode": args.mode,
    }, ckpt_path)
    summary["checkpoint"] = str(ckpt_path)
    save_json(args.out / "cl_summary.json", summary, indent=2)
    print(f"[ContinualRoI] Done: {summary['cl_metrics']} -> {ckpt_path}", flush=True)
    return summary


if __name__ == "__main__":
    main()
