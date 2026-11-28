"""M6.2 RoI-Refine evaluator: runs crop classifier on ambiguous queries of ``logits_val.npz``,
tunes ``(margin_thresh, alpha)`` + counting thresholds strictly on Val-A-calib (with 2-half stability check),
then evaluates once on Val-B with sign test against the unrefined detector.
"""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from autocheckout.datapaths import Layout
from autocheckout.io import load_json, save_json
from autocheckout.logit_dump import LogitDump, load_logits
from autocheckout.predictions import save_predictions
from autocheckout.roi_refine import (
    ambiguous_query_mask,
    build_roi_classifier,
    build_transforms,
    crop_box,
    fuse_logits,
    resolve_image_path,
)
from autocheckout.taskcfg import TaskConfig
from tools.al_score import STAGE, UNLEARNED_IOU, count_on, group_labels, score as al_score_fn


def sign_test_p(right_base: list[int], right_new: list[int]) -> tuple[int, int, float]:
    sb, sn = set(right_base), set(right_new)
    fixed, broken = len(sn - sb), len(sb - sn)
    n = fixed + broken
    if n == 0:
        return 0, 0, 1.0
    k = min(fixed, broken)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return fixed, broken, float(p)


@torch.no_grad()
def infer_roi_logits(
    dump: LogitDump,
    val_full: dict,
    img_root: Path,
    model: torch.nn.Module,
    *,
    max_margin_thresh: float,
    min_prob: float = 0.15,
    pad_ratio: float = 0.10,
    img_size: int = 224,
    batch_size: int = 128,
    device: str = "cuda",
) -> tuple[np.ndarray, int]:
    """Compute crop classifier logits for every query in ``dump`` that is ambiguous under ``max_margin_thresh``.
    Non-ambiguous queries keep zeros (never fused)."""
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = model.to(dev).eval()
    tf = build_transforms(img_size=img_size, train=False)

    mask = ambiguous_query_mask(dump.logits, margin_thresh=max_margin_thresh, min_prob=min_prob)
    roi_logits = np.zeros(dump.logits.shape, dtype=np.float32)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return roi_logits, 0

    id_to_file = {int(im["id"]): im["file_name"] for im in val_full["images"]}
    # Group by image index so each high-res image is opened once.
    by_img: dict[int, list[int]] = {}
    for img_idx, q_idx in coords:
        by_img.setdefault(int(img_idx), []).append(int(q_idx))

    batch_tensors: list[torch.Tensor] = []
    batch_coords: list[tuple[int, int]] = []

    def flush() -> None:
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors, dim=0).to(dev)
        out = model(x).float().cpu().numpy()
        for (ii, qi), vec in zip(batch_coords, out, strict=True):
            roi_logits[ii, qi, : min(vec.shape[0], roi_logits.shape[-1])] = vec[: roi_logits.shape[-1]]
        batch_tensors.clear()
        batch_coords.clear()

    for img_idx, q_indices in by_img.items():
        img_id = int(dump.image_id[img_idx])
        path = resolve_image_path(img_root, id_to_file[img_id])
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            for qi in q_indices:
                crop = crop_box(rgb, dump.boxes[img_idx, qi], pad_ratio=pad_ratio)
                batch_tensors.append(tf(crop))
                batch_coords.append((img_idx, qi))
                if len(batch_tensors) >= batch_size:
                    flush()
    flush()
    return roi_logits, len(coords)


def evaluate_roi_refine(
    dump: LogitDump,
    roi_logits: np.ndarray,
    val_full: dict,
    cfg: TaskConfig,
    calib_ids: list[int],
    valb_ids: list[int],
    *,
    stage: int = STAGE,
    unlearned_iou: float | None = UNLEARNED_IOU,
    margins: tuple[float, ...] = (0.15, 0.25, 0.35, 0.50),
    alphas: tuple[float, ...] = (0.25, 0.40, 0.55, 0.70),
    topk: int = 3,
    min_prob: float = 0.15,
    robust_calib: bool = True,
    seed: int = 0,
) -> tuple[dict, object]:
    """Grid-search ``(margin_thresh, alpha)`` strictly on ``calib_ids``, verify 2-half stability, then score Val-B."""
    eff_unlearned = unlearned_iou if stage < len(cfg.data_tasks) else None
    groups = group_labels(cfg, stage)
    base_preds = dump.predictions(producer="roi_refine_base")
    base_res = al_score_fn(
        val_full, cfg, base_preds, calib_ids, valb_ids, groups=groups, unlearned_iou=eff_unlearned, robust=robust_calib
    )
    base_cal = al_score_fn(
        val_full, cfg, base_preds, calib_ids, calib_ids, groups=groups, unlearned_iou=eff_unlearned, robust=robust_calib
    )

    best_cfg = {
        "margin_thresh": 0.0,
        "alpha": 0.0,
        "calib_cAcc": base_cal["calib_cAcc"],
        "calib_ACD": base_cal["ACD"],
    }
    grid_results = []
    for m in margins:
        for a in alphas:
            fused = fuse_logits(dump.logits, roi_logits, margin_thresh=m, alpha=a, topk=topk, min_prob=min_prob)
            cand_dump = LogitDump(dump.image_id, dump.query, fused, dump.boxes, meta=dump.meta)
            cand_preds = cand_dump.predictions(producer="roi_refine")
            # Score on calib ONLY during search (target=calib_ids so Val-B is never touched).
            cal = al_score_fn(
                val_full, cfg, cand_preds, calib_ids, calib_ids,
                groups=groups, unlearned_iou=eff_unlearned, robust=robust_calib,
            )
            entry = {"margin_thresh": m, "alpha": a, "calib_cAcc": cal["calib_cAcc"], "calib_ACD": cal["ACD"]}
            grid_results.append(entry)
            if (cal["calib_cAcc"] > best_cfg["calib_cAcc"] + 1e-12) or (
                abs(cal["calib_cAcc"] - best_cfg["calib_cAcc"]) <= 1e-12
                and cal["ACD"] < best_cfg["calib_ACD"] - 1e-12
            ):
                best_cfg = dict(entry)

    # 2-half stability check on calib for the chosen (margin, alpha)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(calib_ids))
    mid = len(calib_ids) // 2
    halves = [sorted(calib_ids[int(i)] for i in perm[:mid]), sorted(calib_ids[int(i)] for i in perm[mid:])]

    fused_best = fuse_logits(
        dump.logits, roi_logits, margin_thresh=best_cfg["margin_thresh"], alpha=best_cfg["alpha"],
        topk=topk, min_prob=min_prob,
    )
    best_dump = LogitDump(dump.image_id, dump.query, fused_best, dump.boxes, meta=dump.meta)
    best_preds = best_dump.predictions(producer="roi_refine")
    best_res = al_score_fn(
        val_full, cfg, best_preds, calib_ids, valb_ids, groups=groups, unlearned_iou=eff_unlearned, robust=robust_calib
    )

    half_deltas = []
    for fit_h, chk_h in ((halves[0], halves[1]), (halves[1], halves[0])):
        b_h = al_score_fn(
            val_full, cfg, base_preds, fit_h, chk_h, groups=groups, unlearned_iou=eff_unlearned, robust=robust_calib
        )["cAcc_m2"]
        r_h = al_score_fn(
            val_full, cfg, best_preds, fit_h, chk_h, groups=groups, unlearned_iou=eff_unlearned, robust=robust_calib
        )["cAcc_m2"]
        half_deltas.append({"base_cAcc": b_h, "refined_cAcc": r_h, "delta": r_h - b_h})

    fixed, broken, pval = sign_test_p(base_res["right_images"], best_res["right_images"])
    delta_valb = best_res["cAcc_m2"] - base_res["cAcc_m2"]
    stable_halves = all(h["delta"] > 0 for h in half_deltas)
    # Also report Val-B with base thresholds to separate model gain from threshold shift.
    base_thr_vec = list(base_res["thresholds"].values())
    valb_at_base_thr = count_on(
        val_full, cfg, best_preds, valb_ids, base_thr_vec, groups=groups[0], unlearned_iou=eff_unlearned
    )

    decision = "accept" if (delta_valb >= 0.005 and pval < 0.05 and stable_halves) else "reject"
    report = {
        "best_config": {**best_cfg, "topk": topk, "min_prob": min_prob, "robust_calib": robust_calib,
                        "stage": stage, "unlearned_iou": eff_unlearned},
        "base": {k: v for k, v in base_res.items() if k != "right_images"},
        "refined": {k: v for k, v in best_res.items() if k != "right_images"},
        "refined_at_base_thresholds": {k: v for k, v in valb_at_base_thr.items() if k != "right_images"},
        "delta_valb": delta_valb,
        "fixed_vs_base": fixed,
        "broken_vs_base": broken,
        "sign_p": pval,
        "halves": half_deltas,
        "stable_halves": stable_halves,
        "decision": decision,
        "grid_search": grid_results,
    }
    return report, best_preds


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, help="autocheckout-data folder (default: $AC_DATA)")
    ap.add_argument("--img-root", type=Path, help="image root containing val2019")
    ap.add_argument("--logits", type=Path, required=True, help="logits_val.npz from n0_dump")
    ap.add_argument("--checkpoint", type=Path, required=True, help="roi_classifier.pth from roi_refine_train")
    ap.add_argument("--name", default="m6_roi_r4")
    ap.add_argument("--stage", type=int, default=STAGE)
    ap.add_argument("--unlearned-iou", type=float, default=UNLEARNED_IOU)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--min-prob", type=float, default=0.15)
    ap.add_argument("--standard-calib", action="store_true", help="use legacy smallest-tie-break calib instead of robust")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    data = Layout.at(args.data)
    img_root = args.img_root or (data.root / "rpc")
    if not (img_root / "val2019").is_dir() and (data.root / "archive" / "val2019").is_dir():
        img_root = data.root / "archive"

    val_full = load_json(data.eval / "val_full.json")
    cfg = TaskConfig.load(data.eval / "task_config.json")
    calib_ids = sorted(int(i) for i in load_json(data.relearn / "active/splits/val_a_split.json")["calib"])
    valb_ids = sorted(int(im["id"]) for im in load_json(data.relearn / "meta/val_b.json")["images"])
    eval_ids = sorted(set(calib_ids) | set(valb_ids))

    full_dump = load_logits(args.logits)
    keep = np.isin(full_dump.image_id, eval_ids)
    dump = LogitDump(full_dump.image_id[keep], full_dump.query[keep], full_dump.logits[keep], full_dump.boxes[keep], meta=full_dump.meta)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_roi_classifier(arch=ckpt["arch"], num_classes=int(ckpt["num_classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])

    margins = (0.15, 0.25, 0.35, 0.50)
    roi_logits, n_crops = infer_roi_logits(
        dump, val_full, img_root, model,
        max_margin_thresh=max(margins), min_prob=args.min_prob,
        pad_ratio=float(ckpt.get("pad_ratio", 0.10)),
        img_size=int(ckpt.get("img_size", 224)),
        device=args.device,
    )
    print(f"[RoI-Refine] Inferred {n_crops} ambiguous crops across {len(dump.image_id)} images", flush=True)

    report, refined_preds = evaluate_roi_refine(
        dump, roi_logits, val_full, cfg, calib_ids, valb_ids,
        stage=args.stage, unlearned_iou=args.unlearned_iou,
        margins=margins, topk=args.topk, min_prob=args.min_prob, robust_calib=not args.standard_calib,
    )
    report["name"] = args.name
    report["ambiguous_crops_inferred"] = n_crops
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out / f"{args.name}.json", report, indent=2)
    save_predictions(args.out / f"{args.name}_pred_val.npz", refined_preds)
    print(
        f"[RoI-Refine] {args.name}: base={report['base']['cAcc_m2']*100:.2f}% -> "
        f"refined={report['refined']['cAcc_m2']*100:.2f}% ({report['delta_valb']*100:+.2f}pp, "
        f"+{report['fixed_vs_base']}/-{report['broken_vs_base']}, p={report['sign_p']:.4f}) => {report['decision']}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    main()
