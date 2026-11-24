"""A7 (AN/cyclegan/03): evaluation set of the joint RPC + VN campaign, in the labels of the 225-SKU config.

  val_full.json   Val-A (calib + pool) + Val-B + VN val: one prediction pass per stage
  val_task_<t>    the images of Val-A and VN val with an object of task t, its objects only (the per-epoch val of
                  training, which picks the FSA epoch): never Val-B
  test_full.json  test2019 + VN test (scored once)
  joint_split.json  the image ids of every role (full_syn settings ``joint_split``): thresholds on calib + vn_val,
                  reports on val_b / vn_val, test on test_rpc / test_vn (joint_vn.py)
VN photos (real counter photos, never trained on) are split by the user-approved 52/88 resplit (doc 02 section 5.2),
locked by its md5. VN objects map to the config by their VNxx code (old task-6 files use labels 200-224).

  python -m tools.eval_set_225 --task-config configs/tasks_225_vn10-3-res25_seed0.json --rpc-val <eval/val_full.json>
      --rpc-test <eval/test_full.json> --vn <task6/val_task_6.json> <task6/test_task_6.json> --vn-split <resplit.json>
      --out <eval_225> --rpc-root <dir with val2019/ test2019/> --vn-root <VN package root with checkout/>
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path

from autocheckout.io import load_json, save_json
from autocheckout.taskcfg import TaskConfig
from tools.make_task_json import build_task_coco, images_with

VN_SPLIT_MD5 = '0dac2bd866ab47d51c100b17819861f8'  # AN/cyclegan/scripts_kiem_tra/vn_resplit_candidate_keepold.json
VN_CODE = re.compile(r'^VN(\d\d)\b')


def _to_rpc(coco: dict) -> dict:
    """Objects with RPC ids: an RPC file's categories carry rpc_category_id; a VN file's names start with VNxx."""
    to_rpc = {}
    for c in coco['categories']:
        code = VN_CODE.match(c['name'])
        to_rpc[c['id']] = 200 + int(code.group(1)) if code else c['rpc_category_id']
    return {**coco, 'annotations': [{**a, 'category_id': to_rpc[a['category_id']]} for a in coco['annotations']]}


def _union(parts: list[dict]) -> dict:
    images = [i for p in parts for i in p['images']]
    annotations = [a for p in parts for a in p['annotations']]
    for key, rows in (('image', images), ('object', annotations)):
        ids = [r['id'] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f'{key} ids collide between the RPC and VN files')
    return {'images': images, 'annotations': annotations, 'categories': []}


def build(*, rpc_val: dict, rpc_test: dict, vn: list[dict], vn_split: dict, calib: list[int], pool: list[int],
          val_b: list[int], cfg: TaskConfig) -> tuple[dict[str, dict], dict[str, list[int]]]:
    """({file name: COCO in model labels}, {role: image ids})."""
    if set(calib) & set(val_b) or set(pool) & (set(calib) | set(val_b)):
        raise ValueError('calib, pool and Val-B overlap')
    vn_all = _union([_to_rpc(p) for p in vn])
    by_name = {i['file_name']: i['id'] for i in vn_all['images']}
    both = set(vn_split['val']) & set(vn_split['test'])
    if both:
        raise ValueError(f'VN photos in both val and test: {sorted(both)[:3]}')
    if set(by_name) != set(vn_split['val']) | set(vn_split['test']):
        raise ValueError(f'VN photos not in the split: {sorted(set(by_name) ^ (set(vn_split["val"]) | set(vn_split["test"])))[:3]}')
    roles = {'calib': sorted(calib), 'pool': sorted(pool), 'val_b': sorted(val_b),
             'vn_val': sorted(by_name[n] for n in vn_split['val']), 'test_rpc': sorted(i['id'] for i in rpc_test['images']),
             'test_vn': sorted(by_name[n] for n in vn_split['test'])}
    rpc_val = _to_rpc(rpc_val)
    if {i['id'] for i in rpc_val['images']} != set(calib) | set(pool) | set(val_b):
        raise ValueError('RPC val is not exactly calib + pool + Val-B')
    val = _union([rpc_val, vn_all])
    val = {**val, 'images': [i for i in val['images'] if i['id'] not in set(roles['test_vn'])]}
    val['annotations'] = [a for a in val['annotations'] if a['image_id'] not in set(roles['test_vn'])]
    test = _union([_to_rpc(rpc_test), vn_all])
    test = {**test, 'images': [i for i in test['images'] if i['id'] not in set(roles['vn_val'])]}
    test['annotations'] = [a for a in test['annotations'] if a['image_id'] not in set(roles['vn_val'])]
    files = {'val_full.json': build_task_coco(val, cfg), 'test_full.json': build_task_coco(test, cfg)}
    selection = set(calib) | set(pool) | set(roles['vn_val'])
    for task in cfg.data_tasks:
        rpc = {c.rpc_category_id for c in task.classes}
        files[f'val_task_{task.task_id}.json'] = build_task_coco(val, cfg, images_with(val, rpc) & selection, rpc)
    return files, roles


def main(argv: list[str] | None = None) -> None:
    from autocheckout.datapaths import Layout
    from autocheckout.io import md5_file
    from autocheckout.new_skus import _link_images

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--task-config', type=Path, required=True)
    ap.add_argument('--rpc-val', type=Path, required=True)
    ap.add_argument('--rpc-test', type=Path, required=True)
    ap.add_argument('--vn', type=Path, nargs='+', required=True)
    ap.add_argument('--vn-split', type=Path, required=True)
    ap.add_argument('--data', type=Path, help='data folder of the Val-A split and Val-B (default ../autocheckout-data)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rpc-root', type=Path, required=True, help='folder with val2019/ and test2019/')
    ap.add_argument('--vn-root', type=Path, required=True, help='VN package folder with checkout/')
    args = ap.parse_args(argv)
    if md5_file(args.vn_split) != VN_SPLIT_MD5:
        raise SystemExit(f'{args.vn_split}: md5 is not the locked VN split {VN_SPLIT_MD5}')
    data = Layout.at(args.data)
    split = load_json(data.relearn / 'active/splits/val_a_split.json')
    val_b = [i['id'] for i in load_json(data.relearn / 'meta/val_b.json')['images']]
    cfg = TaskConfig.load(args.task_config)
    files, roles = build(rpc_val=load_json(args.rpc_val), rpc_test=load_json(args.rpc_test),
                         vn=[load_json(p) for p in args.vn], vn_split=load_json(args.vn_split),
                         calib=split['calib'], pool=split['pool'], val_b=val_b, cfg=cfg)
    for name, coco in files.items():
        save_json(args.out / name, coco)
    save_json(args.out / 'joint_split.json', roles)
    shutil.copyfile(args.task_config, args.out / 'task_config.json')
    _link_images(args.out / 'images', [], named={'val2019': args.rpc_root / 'val2019', 'test2019': args.rpc_root /
                                                  'test2019', 'checkout': args.vn_root / 'checkout'})
    manifest = {'vn_split_md5': VN_SPLIT_MD5, 'roles': {k: len(v) for k, v in roles.items()}, 'files': {}}
    for path in sorted(args.out.glob('*.json')):
        if path.name != 'manifest.json':
            coco = load_json(path)
            manifest['files'][path.name] = {'images': len(coco.get('images', [])),
                                            'objects': len(coco.get('annotations', [])), 'md5': md5_file(path),
                                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    save_json(args.out / 'manifest.json', manifest, indent=1)
    missing = [i['file_name'] for name in ('val_full.json', 'test_full.json') for i in files[name]['images']
               if not (args.out / 'images' / i['file_name']).is_file()]
    print({k: len(v) for k, v in roles.items()}, f'missing images {len(missing)}', missing[:3])


if __name__ == '__main__':
    main()
