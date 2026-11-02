"""A4: training data of one active-learning selection, as a prebuilt variant the improve mode reads
(``update_model._prebuilt_data``): ``<release>/variants/al_<selection>/``.

The fixed part (synthetic images, replay, ``train_rows.json``) is copied from the base variant unchanged; only the
real images change: the selected Val-A images, every object labelled, each repeated ``feedback_repeat`` times as in
the base. Selected images missing from ``images/real/val2019`` are copied from ``archive/val2019``. The build stops
when a Val-B, calib or test image would enter the training file.

    python -m tools.al_build_variant --release ../relearn_fix_v2 --selection margin_B500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from autocheckout.io import load_json, md5_file, save_json


def fixed_part_md5(coco: dict) -> str:
    """Content hash of the non-real images and their boxes, ids left out (same for every variant of one base)."""
    boxes = defaultdict(list)
    for a in coco['annotations']:
        boxes[a['image_id']].append((a['category_id'], [round(v, 2) for v in a['bbox']]))
    items = sorted((i['file_name'], sorted(boxes[i['id']])) for i in coco['images'] if i.get('source') != 'real')
    return hashlib.md5(json.dumps(items).encode()).hexdigest()


def build(release: Path, selection: str, base: str = 'pilot', copy_images: bool = True) -> dict:
    sel = load_json(release / f'active/selections/{selection}.json')
    split = load_json(release / 'active/splits/val_a_split.json')
    base_dir = release / 'variants' / base
    base_train, base_manifest = load_json(base_dir / 'train_task_5_fix.json'), load_json(base_dir / 'manifest.json')
    repeat = int(base_manifest['params']['feedback_repeat'])
    val_a, val_b = load_json(release / 'meta/val_a.json'), load_json(release / 'meta/val_b.json')
    test_names = {Path(i['file_name']).name for i in load_json(release / 'eval/test_full.json')['images']}

    picked = list(dict.fromkeys(sel['image_ids']))
    blocked = {i['id'] for i in val_b['images']} | set(split['calib'])
    images_a = {i['id']: i for i in val_a['images']}
    if set(picked) & blocked or not set(picked) <= set(split['pool']):
        raise ValueError(f'{selection}: images outside the pool (Val-B or calib) in the selection')
    names = {i: Path(images_a[i]['file_name']).name for i in picked}
    if set(names.values()) & test_names:
        raise ValueError(f'{selection}: test images in the selection')

    boxes = defaultdict(list)
    for a in val_a['annotations']:
        boxes[a['image_id']].append(a)
    images, annotations = [], []

    def add(image: dict, anns: list[dict]) -> None:
        new_id = len(images) + 1
        images.append({**image, 'id': new_id})
        for a in anns:
            annotations.append({k: a[k] for k in ('category_id', 'bbox', 'area', 'iscrowd')}
                               | {'id': len(annotations) + 1, 'image_id': new_id})

    for _ in range(repeat):
        for i in picked:
            src = images_a[i]
            add({'file_name': f'real/val2019/{names[i]}', 'width': src['width'], 'height': src['height'],
                 'level': src.get('level', ''), 'orig_id': i, 'source': 'real'}, boxes[i])
    base_boxes = defaultdict(list)
    for a in base_train['annotations']:
        base_boxes[a['image_id']].append(a)
    for image in base_train['images']:
        if image.get('source') != 'real':
            add({k: v for k, v in image.items() if k != 'id'}, base_boxes[image['id']])
    train = {'images': images, 'annotations': annotations, 'categories': base_train['categories']}
    if fixed_part_md5(train) != fixed_part_md5(base_train):
        raise AssertionError('fixed part changed')

    copied = 0
    if copy_images:
        target = release / 'images/real/val2019'
        target.mkdir(parents=True, exist_ok=True)
        for name in names.values():
            if not (target / name).is_file():
                shutil.copy2(release / 'archive/val2019' / name, target / name)
                copied += 1

    out = release / 'variants' / f'al_{selection}'
    rows = load_json(base_dir / 'train_rows.json')
    save_json(out / 'train_task_5_fix.json', train)
    save_json(out / 'train_rows.json', rows)
    per_source = defaultdict(int)
    for image in images:
        per_source[image['source']] += 1
    row_set, objects_per_row = set(rows), defaultdict(int)
    for a in annotations:
        if a['category_id'] in row_set:
            objects_per_row[a['category_id']] += 1
    manifest = {'variant': f'al_{selection}', 'rule': 'active learning: selected Val-A images x repeat + fixed part '
                                                      f'of {base}; see tools/al_build_variant.py',
                'base': base, 'base_train_md5': md5_file(base_dir / 'train_task_5_fix.json'),
                'fixed_part_md5': fixed_part_md5(train), 'selection': selection,
                'selection_md5': md5_file(release / f'active/selections/{selection}.json'),
                'strategy': sel['strategy'], 'budget': sel['budget'], 'seed': sel['seed'],
                'round': sel.get('opts', {}).get('round', 1),
                'params': {**base_manifest['params'], 'feedback_repeat': repeat}, 'rows': rows,
                'images': len(images), 'objects': len(annotations), 'images_per_source': dict(per_source),
                'labelled_images': len(picked), 'labelled_boxes': sum(len(boxes[i]) for i in picked),
                'objects_per_weak_sku': {str(r): objects_per_row[r] for r in rows},
                'min_objects_any_row': min(objects_per_row[r] for r in rows),
                'val_b_images_in_train': 0, 'calib_images_in_train': 0, 'images_copied': copied}
    save_json(out / 'manifest.json', manifest)
    return manifest


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--release', type=Path, required=True)
    ap.add_argument('--selection', nargs='+', required=True)
    ap.add_argument('--base', default='pilot')
    args = ap.parse_args(argv)
    return [build(args.release, s, args.base) for s in args.selection]


if __name__ == '__main__':
    for m in main():
        print(json.dumps({k: m[k] for k in ('variant', 'images', 'objects', 'images_per_source', 'labelled_images',
                                            'labelled_boxes', 'min_objects_any_row', 'images_copied')}))
