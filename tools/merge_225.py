"""A7 (AN/cyclegan/03_QUYET_DINH_VA_PLAN.md section 1.6): the train file of one task from several synthetic parts.

Parts: the RPC composites (synth_v2 release, 1840 px), the same scenes rendered by CycleGAN at 800 px (render_cg.py,
same relative file names) and the VN composites (their own synth_v2 release). Each part keeps its own folder (the first
path part of every file name, linked to the part's root under one image folder, as new_skus._link_images) and its own
id range (``offset``: image and object ids of the part + offset). A part with ``scale`` != 1 has its boxes and areas
scaled and its masks dropped (PDP reads bbox, area, category_id and iscrowd only; a mask would need re-encoding at the
new size). Stops on colliding ids, an id past its range, a box outside its image or two names for one category.

  python -m tools.merge_225 --task 1 --out <merged release> --task-config configs/tasks_225_vn10-3-res25_seed0.json \\
      --part rpc_syn:<rpc release>:1:0 --part rpc_render:<render root>:0.43478:100000000 \\
      --labels-of rpc_render:<rpc release> --part vn_syn:<vn release>:1:200000000
The merged release holds tasks/synth_v2/train_task_<t>.json and its _gt_full audit file, manifest.json (sha256 per
file, as full_syn.link_tasks checks) and task_config.json; its images are <name>/... links to each part root.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from autocheckout.io import load_json, save_json

OFFSET_STEP = 100_000_000


def merge_parts(parts: list[tuple[str, dict, float, int]]) -> dict:
    """COCO of the parts (name, coco, scale, offset), in the given order."""
    images, annotations, categories = [], [], {}
    for name, coco, scale, offset in parts:
        for c in coco['categories']:
            if categories.setdefault(c['id'], c)['name'] != c['name']:
                raise ValueError(f'category {c["id"]} named {categories[c["id"]]["name"]!r} and {c["name"]!r}')
        sizes = {}
        for image in coco['images']:
            if not 0 <= image['id'] < OFFSET_STEP:
                raise ValueError(f'{name}: image id {image["id"]} does not fit under an id offset of {OFFSET_STEP}')
            w, h = round(image['width'] * scale), round(image['height'] * scale)
            sizes[image['id']] = (w, h)
            images.append({**image, 'id': image['id'] + offset, 'file_name': f'{name}/{image["file_name"]}',
                           'width': w, 'height': h, 'part': name})
        for ann in coco['annotations']:
            if not 0 <= ann['id'] < OFFSET_STEP:
                raise ValueError(f'{name}: object id {ann["id"]} does not fit under an id offset of {OFFSET_STEP}')
            x, y, bw, bh = (v * scale for v in ann['bbox'])
            w, h = sizes[ann['image_id']]
            if x < -0.5 or y < -0.5 or x + bw > w + 0.5 or y + bh > h + 0.5:
                raise ValueError(f'{name}: object {ann["id"]} lies outside its {w}x{h} image')
            new = {**ann, 'id': ann['id'] + offset, 'image_id': ann['image_id'] + offset, 'bbox': [x, y, bw, bh],
                   'area': ann['area'] * scale * scale}
            if scale != 1.0:
                new.pop('segmentation', None)
            annotations.append(new)
    for key, rows in (('image', images), ('object', annotations)):
        ids = [r['id'] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f'{key} ids collide across the parts')
    return {'images': images, 'annotations': annotations,
            'categories': [categories[k] for k in sorted(categories)]}


def main(argv: list[str] | None = None) -> None:
    import hashlib
    import shutil

    from autocheckout.new_skus import _link_images

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--task', type=int, required=True)
    ap.add_argument('--part', action='append', required=True, help='name:root:scale:offset')
    ap.add_argument('--labels-of', action='append', default=[],
                    help='name:release root whose files a part without its own uses (the render part)')
    ap.add_argument('--out', type=Path, required=True,
                    help='merged release: tasks/synth_v2/, manifest.json, task_config.json, <name> -> each part root')
    ap.add_argument('--task-config', type=Path, required=True, help='the 225-SKU config (same as the eval set)')
    args = ap.parse_args(argv)
    labels_of = dict(item.split(':', 1) for item in args.labels_of)
    links, files = {}, {}
    for kind in ('', '_gt_full'):
        parts = []
        for item in args.part:
            name, root, scale, offset = item.rsplit(':', 3)
            source = Path(labels_of.get(name, root)) / 'tasks/synth_v2' / f'train_task_{args.task}{kind}.json'
            parts.append((name, load_json(source), float(scale), int(offset)))
            links[name] = Path(root)
        files[f'train_task_{args.task}{kind}.json'] = merge_parts(parts)
    merged = files[f'train_task_{args.task}.json']
    if [i['id'] for i in merged['images']] != [i['id'] for i in files[f'train_task_{args.task}_gt_full.json']['images']]:
        raise SystemExit('train and gt_full files list different images')
    missing = [i['file_name'] for i in merged['images']
               if not (links[i['part']] / i['file_name'].split('/', 1)[1]).is_file()]
    if missing:
        raise SystemExit(f'{len(missing)} images missing, e.g. {missing[:3]}')
    _link_images(args.out, [], named=links)
    digest = {}
    for name, data in files.items():
        save_json(args.out / 'tasks/synth_v2' / name, data)
        digest[name] = hashlib.sha256((args.out / 'tasks/synth_v2' / name).read_bytes()).hexdigest()
    shutil.copyfile(args.task_config, args.out / 'task_config.json')
    path = args.out / 'manifest.json'
    manifest = load_json(path) if path.is_file() else {'parts': {}, 'tasks': {}}
    manifest['parts'].update({name: str(root) for name, root in links.items()})
    manifest['tasks'][str(args.task)] = {'images': len(merged['images']), 'objects': len(merged['annotations']),
                                         'by_part': {n: sum(i['part'] == n for i in merged['images']) for n in links},
                                         'files': digest}
    save_json(path, manifest, indent=1)
    print(f'task {args.task}: {len(merged["images"])} images {manifest["tasks"][str(args.task)]["by_part"]}, '
          f'{len(merged["annotations"])} objects -> {args.out}')


if __name__ == '__main__':
    main()
