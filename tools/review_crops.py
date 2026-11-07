"""B3 / B1: contact sheets to look at by eye. For the counted boxes of a model on Val-B (same counting as
``tools/error_breakdown.py``):

  --extra LABEL         boxes of LABEL on no GT object (extra_bg), next to correct LABEL objects
  --pair A B            GT A counted as B and GT B counted as A, next to correct A and correct B objects

    python -m tools.review_crops --val .../val_b/val_full.json --pred .../pred_val.npz --policy .../al_margin_k_B500_r2.json \\
        --task-config configs/relearn/task_config_relearn.json --images .../autocheckout-data/rpc --out results/review_crops \\
        --extra 141 --pair 162 186
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from autocheckout.cl_metrics import box_iou
from autocheckout.counting import dedup_detections
from autocheckout.io import load_json
from autocheckout.predictions import load_predictions
from autocheckout.taskcfg import TaskConfig
from tools.error_breakdown import _match, thresholds_of

CELL, PER_ROW = 200, 10


def counted_boxes(val, preds, thresholds, nms_iou):
    ids = sorted(i['id'] for i in val['images'])
    kept = dedup_detections(preds.subset(np.isin(preds.image_id, ids) & (preds.label < len(thresholds))), nms_iou)
    keep = kept.score >= thresholds[kept.label]
    by = defaultdict(list)
    for i in np.flatnonzero(keep):
        by[int(kept.image_id[i])].append((int(kept.label[i]), kept.boxes[i], float(kept.score[i])))
    return by


def collect(val, by, extra, pair):
    gt = defaultdict(list)
    for a in val['annotations']:
        x, y, w, h = a['bbox']
        gt[a['image_id']].append((a['category_id'], [x, y, x + w, y + h]))
    cases = defaultdict(list)  # name -> (image id, box, caption)
    for image_id, g in gt.items():
        gl, gb = np.array([c for c, _ in g]), np.array([b for _, b in g], dtype=float)
        counted = by.get(image_id, [])
        pb = np.array([b for _, b, _ in counted], dtype=float).reshape(-1, 4)
        pairs = _match(gb, pb)
        taken = set(pairs.values())
        for gi, ji in pairs.items():
            lab, pred, score = int(gl[gi]), counted[ji][0], counted[ji][2]
            if pair and {lab, pred} == set(pair) and lab != pred:
                cases[f'GT {lab} -> counted {pred}'].append((image_id, gb[gi], f'{score:.2f}'))
            elif pair and lab == pred and lab in pair:
                cases[f'correct {lab}'].append((image_id, gb[gi], f'{score:.2f}'))
            elif extra is not None and lab == pred == extra:
                cases[f'correct {extra}'].append((image_id, gb[gi], f'{score:.2f}'))
        if extra is not None:
            for j, (lab, box, score) in enumerate(counted):
                if lab == extra and j not in taken and (not len(gb) or box_iou(box[None], gb).max() < 0.1):
                    cases[f'extra {extra} on background'].append((image_id, box, f'{score:.2f}'))
    return cases


def sheet(rows, root, val_images, limit, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))[:limit]
    cells = []
    for k in order:
        image_id, box, cap = rows[k]
        img = Image.open(root / val_images[image_id]).convert('RGB')
        x0, y0, x1, y1 = box
        pad = max(x1 - x0, y1 - y0) * 0.6 + 20
        crop = img.crop((max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(img.width, int(x1 + pad)),
                         min(img.height, int(y1 + pad))))
        d = ImageDraw.Draw(crop)
        d.rectangle([x0 - max(0, int(x0 - pad)), y0 - max(0, int(y0 - pad)), x1 - max(0, int(x0 - pad)),
                     y1 - max(0, int(y0 - pad))], outline=(255, 0, 0), width=3)
        crop = crop.resize((CELL, CELL))
        ImageDraw.Draw(crop).text((4, 4), f'#{image_id} s={cap}', fill=(255, 255, 0))
        cells.append(crop)
    n = (len(cells) + PER_ROW - 1) // PER_ROW
    out = Image.new('RGB', (PER_ROW * CELL, max(n, 1) * CELL))
    for i, c in enumerate(cells):
        out.paste(c, ((i % PER_ROW) * CELL, (i // PER_ROW) * CELL))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--val', type=Path, required=True)
    ap.add_argument('--pred', type=Path, required=True)
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--task-config', type=Path, required=True)
    ap.add_argument('--images', type=Path, required=True, help='folder holding val2019/')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--extra', type=int)
    ap.add_argument('--pair', type=int, nargs=2)
    ap.add_argument('--limit', type=int, default=40)
    args = ap.parse_args(argv)
    val = load_json(args.val)
    thresholds, nms_iou = thresholds_of(load_json(args.policy), TaskConfig.load(args.task_config))
    by = counted_boxes(val, load_predictions(args.pred), thresholds, nms_iou)
    cases = collect(val, by, args.extra, args.pair)
    names = {i['id']: i['file_name'] for i in val['images']}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in sorted(cases.items()):
        f = args.out / (name.replace(' ', '_').replace('>', '') + '.jpg')
        sheet(rows, args.images, names, args.limit if 'correct' not in name else 20).save(f, quality=85)
        print(f'{len(rows):4d}  {name}  -> {f}')


if __name__ == '__main__':
    main()
