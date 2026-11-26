"""K0 (docs/ai/planning/2026-11-26-plan-M1-M6-kiem-prototype.md): are the class prototypes of a checkpoint good
anchors for the prototype margin loss (M1)?

For the queries the Hungarian matcher pairs with a ground-truth box, compares three ways of naming the class: the
classifier (argmax of the logits), the nearest prototype and the nearest classifier row (both by cosine). Also
reports which classes have a prototype and which prototypes sit closest together. Verdict: ``prototype`` when the
prototypes name the class nearly as well as the classifier, else ``class_embed`` when the classifier rows do, else
``neither``.

    python -m tools.proto_check --checkpoint <task_final.pth> --val <val_full.json> --img-root <raw root> \\
        --split <val_a_split.json> --split-key calib --n 300 --device cpu --out results/proto_check/r4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from autocheckout.datapaths import REPO

GATE_TOL = 0.05
FEW_QUERIES = 5
CLOSEST_PAIRS = 20


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="task_final.pth with its prototypes")
    parser.add_argument("--val", type=Path, required=True, help="COCO file of the images (e.g. eval/val_full.json)")
    parser.add_argument("--img-root", type=Path, required=True, help="folder the file_name entries are relative to")
    parser.add_argument("--split", type=Path, required=True, help="JSON with the image ids to use (Val-A split)")
    parser.add_argument("--split-key", default="calib")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tol", type=float, default=GATE_TOL, help="accuracy gap to the classifier allowed")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def coverage(counts: dict, task_sizes: list[int], seen: int, few: int = FEW_QUERIES) -> dict:
    """Classes of the learned tasks with / without a prototype; ``few``: with fewer than ``few`` cached queries."""
    per_task, offset = [], 0
    for task, size in enumerate(task_sizes, start=1):
        if offset >= seen:
            break
        classes = range(offset, min(offset + size, seen))
        per_task.append({"task": task, "classes": len(classes),
                         "with_prototype": sum(counts.get(c, 0) > 0 for c in classes)})
        offset += size
    return {"seen_classes": seen, "classes_with_prototype": sum(counts.get(c, 0) > 0 for c in range(seen)),
            "empty": [c for c in range(seen) if counts.get(c, 0) == 0],
            "few": [c for c in range(seen) if 0 < counts.get(c, 0) < few], "per_task": per_task}


def normalized_anchors(vectors: dict, valid: list[int]) -> tuple[torch.Tensor, list[int]]:
    """Unit-length anchors of the ``valid`` classes (in that order) and their class ids."""
    ids = sorted(valid)
    stacked = torch.stack([vectors[c].detach().float().cpu() for c in ids])
    return F.normalize(stacked, dim=1), ids


def closest_pairs(anchors: torch.Tensor, ids: list[int], k: int = CLOSEST_PAIRS) -> list[dict]:
    """The ``k`` pairs of different classes with the highest cosine, highest first."""
    similarity = anchors @ anchors.T
    rows, cols = torch.triu_indices(len(ids), len(ids), offset=1)
    values = similarity[rows, cols]
    order = values.argsort(descending=True)[:k]
    return [{"a": ids[int(rows[i])], "b": ids[int(cols[i])], "cosine": round(float(values[i]), 4)} for i in order]


def anchor_scores(features: torch.Tensor, labels: torch.Tensor, anchors: torch.Tensor, ids: list[int]) -> dict:
    """Nearest-anchor accuracy and margin cos(z, a_c) - max_{k != c} cos(z, a_k) of matched queries whose class has
    an anchor; the others are counted as skipped."""
    position = {c: i for i, c in enumerate(ids)}
    keep = torch.tensor([int(c) in position for c in labels], dtype=torch.bool)
    if not keep.any():
        return {"matched": 0, "skipped": len(labels), "accuracy": 0.0, "margin_median": 0.0, "margins": []}
    target = torch.tensor([position[int(c)] for c in labels[keep]])
    similarity = F.normalize(features[keep].float(), dim=1) @ anchors.T
    correct = similarity.gather(1, target[:, None]).squeeze(1)
    others = similarity.scatter(1, target[:, None], float("-inf")).max(dim=1).values
    margins = correct - others
    return {"matched": int(keep.sum()), "skipped": int((~keep).sum()),
            "accuracy": float((similarity.argmax(dim=1) == target).float().mean()),
            "margin_median": float(margins.median()), "margins": margins.tolist()}


def verdict(acc_cls: float, acc_proto: float, acc_weight: float, tol: float = GATE_TOL) -> str:
    """Anchor of M1 (K0 gate of the plan)."""
    if acc_proto >= acc_cls - tol:
        return "prototype"
    if acc_weight >= acc_cls - tol:
        return "class_embed"
    return "neither"


def matched_queries(model, samples, device, seen: int):
    """Decoder output, true class and classifier class of every query the matcher pairs with a ground truth."""
    from two_pass import prompted_outputs

    features, labels, predicted = [], [], []
    for index, (_, batch) in enumerate(samples, start=1):
        pixels, mask = batch["pixel_values"].to(device), batch["pixel_mask"].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
        with torch.no_grad():
            out = prompted_outputs(model, pixels, mask, use_prompts=True, local_query=model.config.local_query)
            pairs = model.matcher({"logits": out.logits, "pred_boxes": out.pred_boxes}, targets)
        for b, (query_ids, target_ids) in enumerate(pairs):
            features.append(out.last_hidden_state[b, query_ids].cpu())
            labels.append(targets[b]["class_labels"][target_ids].cpu())
            predicted.append(out.logits[b, query_ids, :seen].argmax(dim=-1).cpu())
        print(f"{index:4d}/{len(samples)} ảnh, {sum(len(x) for x in labels)} query đã ghép", flush=True)
    return torch.cat(features), torch.cat(labels), torch.cat(predicted)


def confused_pair_similarity(labels, predicted, anchors, ids) -> dict:
    """Do the classes the classifier confuses have closer prototypes than an average pair?"""
    position = {c: i for i, c in enumerate(ids)}
    similarity = anchors @ anchors.T
    confused = [float(similarity[position[int(t)], position[int(p)]]) for t, p in zip(labels, predicted)
                if int(t) != int(p) and int(t) in position and int(p) in position]
    rows, cols = torch.triu_indices(len(ids), len(ids), offset=1)
    every = similarity[rows, cols]
    return {"confusions": len(confused),
            "cosine_median_confused": round(float(torch.tensor(confused).median()), 4) if confused else None,
            "cosine_median_all_pairs": round(float(every.median()), 4) if len(every) else None}


def main(argv=None) -> dict:
    sys.path.insert(0, str(REPO / "pdp"))
    from tools.p1_check import load_images, load_model

    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    task_sizes = list(checkpoint["runtime"]["model_config"]["task_num_classes"])
    seen = sum(task_sizes[:checkpoint["task_id"]])
    counts = checkpoint["class_cache_count"]
    valid = [c for c in range(seen) if counts.get(c, 0) > 0]

    model, processor = load_model(args.checkpoint, device)
    args.warmup = 0
    features, labels, predicted = matched_queries(model, load_images(args, processor), device, seen)

    prototypes, proto_ids = normalized_anchors(checkpoint["class_prototypes"], valid)
    rows = model.class_embed[-1].weight.detach().cpu()
    weights, weight_ids = normalized_anchors({c: rows[c] for c in range(seen)}, list(range(seen)))
    by_proto = anchor_scores(features, labels, prototypes, proto_ids)
    by_weight = anchor_scores(features, labels, weights, weight_ids)
    acc_cls = float((predicted == labels).float().mean()) if len(labels) else 0.0

    summary = {
        "checkpoint": str(args.checkpoint), "images": args.n, "matched": len(labels),
        "coverage": coverage(counts, task_sizes, seen),
        "acc_cls": round(acc_cls, 4), "acc_proto": round(by_proto["accuracy"], 4),
        "acc_weight": round(by_weight["accuracy"], 4),
        "proto_skipped": by_proto["skipped"],
        "margin_median_proto": round(by_proto["margin_median"], 4),
        "margin_median_weight": round(by_weight["margin_median"], 4),
        "confused_pairs": confused_pair_similarity(labels, predicted, prototypes, proto_ids),
        "closest_prototype_pairs": closest_pairs(prototypes, proto_ids),
        "tol": args.tol, "verdict": verdict(acc_cls, by_proto["accuracy"], by_weight["accuracy"], args.tol),
    }
    (args.out / "proto_check_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "closest_prototype_pairs"}, indent=1,
                     ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
