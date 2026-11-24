"""Data of the improve mode (docs/project/feedback-finetune-plan-2026-11-01.md, I2-I7): val split into a feedback half
and a scoring half, the feedback images, the confusion pairs, the synthetic images of the retrained SKUs, the replay,
and the merged release the training run reads. Every object of every training image is labelled (F4).

Functions take and return COCO dicts; only ``build_release`` and ``build_task_dir`` write files.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from autocheckout.groups import coco_group_keys
from autocheckout.io import save_json
from autocheckout.predictions import Predictions
from tools.sku_regression import evaluate

Coco = dict[str, Any]


def subset(coco: Coco, image_ids: Iterable[int]) -> Coco:
    """The images of ``image_ids`` and their annotations, other keys kept."""
    ids = set(image_ids)
    return {**coco, 'images': [i for i in coco['images'] if i['id'] in ids],
            'annotations': [a for a in coco['annotations'] if a['image_id'] in ids]}


def scene_groups(val: Coco) -> dict[int, str]:
    """Scene group of every image: ``coco_group_keys`` on RPC file names (same basket shot again), else one group per
    image (fixtures whose names are not RPC names)."""
    try:
        return coco_group_keys(val)
    except ValueError:
        return {image['id']: f"i{image['id']}" for image in val['images']}


def split_val(val: Coco, seed: int) -> tuple[Coco, Coco]:
    """I2: Val-A (feedback) and Val-B (scoring), half of the images each, whole scene groups on one side, balanced
    level by level. Within a level the groups are shuffled, then each goes to the half with fewer images."""
    keys, level = scene_groups(val), {i['id']: i.get('level', '') for i in val['images']}
    groups: dict[str, list[int]] = defaultdict(list)
    for image_id, key in keys.items():
        groups[key].append(image_id)
    by_level: dict[str, list[str]] = defaultdict(list)
    for key in sorted(groups):
        by_level[level[groups[key][0]]].append(key)
    rng, halves = random.Random(seed), ([], [])
    for name in sorted(by_level):
        keys_of_level = by_level[name]
        rng.shuffle(keys_of_level)
        counts = [0, 0]
        for key in sorted(keys_of_level, key=lambda k: -len(groups[k])):
            side = 0 if counts[0] <= counts[1] else 1
            halves[side].extend(groups[key])
            counts[side] += len(groups[key])
    return subset(val, halves[0]), subset(val, halves[1])


def wrong_images(val: Coco, counted: Predictions, labels: Iterable[int]) -> list[int]:
    """I3: images whose counts are not exactly right (the bills a cashier would correct)."""
    correct = evaluate(val, counted, labels)['correct']
    return sorted(i for i, ok in correct.items() if not ok)


def confusion_pairs(val: Coco, counted: Predictions, labels: Iterable[int]) -> list[tuple[int, int, int]]:
    """I4: unordered pairs (a, b, count) of an object of one SKU counted as the other, both directions added, most
    frequent first."""
    pairs = Counter()
    for (truth, predicted), count in evaluate(val, counted, labels)['swapped'].items():
        pairs[tuple(sorted((truth, predicted)))] += count
    return [(a, b, n) for (a, b), n in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))]


def pair_rows(pairs: Sequence[tuple[int, int, int]], top: int | None = None, min_count: int | None = None,
              extra: Iterable[int] = ()) -> list[int]:
    """I5: the labels of both sides of the ``top`` pairs, or of the pairs with >= ``min_count`` confusions, plus
    ``extra``."""
    chosen = pairs[:top] if top is not None else [p for p in pairs if min_count is None or p[2] >= min_count]
    return sorted({label for a, b, _ in chosen for label in (a, b)} | set(extra))


def resolve_aliases(coco: Coco, alias_to_label: dict[int, int]) -> Coco:
    """I6: the labels of a relearn release (aliases 200..) back to the original labels."""
    return {**coco, 'annotations': [{**a, 'category_id': alias_to_label.get(a['category_id'], a['category_id'])}
                                    for a in coco['annotations']]}


def images_with(coco: Coco, labels: Iterable[int]) -> Coco:
    """I6: the images holding at least one object of ``labels``."""
    labels = set(labels)
    return subset(coco, {a['image_id'] for a in coco['annotations'] if a['category_id'] in labels})


def select_by_quota(coco: Coco, quota: dict[int, int], seed: int, exclude: Iterable[int] = ()) -> Coco:
    """I6 with a cap: the images that bring the classes of ``quota`` up to their number of objects. Images go in a
    shuffled order (seed), those holding more distinct classes of ``quota`` first (often both sides of a confused
    pair); an image is taken while it still adds an object to a class below its quota. Images not in ``exclude``."""
    excluded = set(exclude)
    objects: dict[int, list[int]] = defaultdict(list)
    for a in coco['annotations']:
        objects[a['image_id']].append(a['category_id'])
    candidates = sorted(i['id'] for i in coco['images']
                        if i['id'] not in excluded and any(c in quota for c in objects[i['id']]))
    random.Random(seed).shuffle(candidates)
    candidates.sort(key=lambda i: -len({c for c in objects[i] if c in quota}))
    count, chosen = Counter(), []
    for image_id in candidates:
        if any(count[c] < quota.get(c, 0) for c in objects[image_id]):
            chosen.append(image_id)
            count.update(objects[image_id])
    return subset(coco, chosen)


def select_replay(coco: Coco, exclude: Iterable[int], n_images: int, min_per_class: int, labels: Iterable[int],
                  seed: int) -> Coco:
    """I7: ``n_images`` images not in ``exclude``: first, in a shuffled order, every image that still adds an object
    of a class with fewer than ``min_per_class`` objects; then random images up to ``n_images``."""
    labels, excluded = set(labels), set(exclude)
    objects: dict[int, list[int]] = defaultdict(list)
    for a in coco['annotations']:
        objects[a['image_id']].append(a['category_id'])
    candidates = sorted(i['id'] for i in coco['images'] if i['id'] not in excluded)
    random.Random(seed).shuffle(candidates)
    count, chosen = Counter(), []
    for image_id in candidates:
        if len(chosen) >= n_images:
            break
        if any(c in labels and count[c] < min_per_class for c in objects[image_id]):
            chosen.append(image_id)
            count.update(objects[image_id])
    taken = set(chosen)
    rest = [i for i in candidates if i not in taken]
    chosen += rest[:max(0, n_images - len(chosen))]
    return subset(coco, chosen)


def build_release(root: Path, sources: dict[str, Path], parts: Sequence[tuple[str, Coco, int]],
                  categories: list[dict[str, Any]]) -> tuple[Coco, dict[str, Any]]:
    """The training file of one improve run and its image folder: ``root/<source>`` links to each source's image
    root, ``file_name`` gets that prefix, images and annotations get new ids (``orig_id`` and ``source`` kept). A part
    ``(source, coco, repeat)`` is written ``repeat`` times (I3: feedback images x3)."""
    root.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        link = root / name
        if link.is_symlink() and link.resolve() != Path(path).resolve():
            raise ValueError(f'{link} points to {link.resolve()}, expected {path}')
        if not link.exists():
            link.symlink_to(Path(path).resolve())
    images, annotations, counts = [], [], Counter()
    for source, coco, repeat in parts:
        if source not in sources:
            raise ValueError(f'unknown source {source}')
        anns = defaultdict(list)
        for a in coco['annotations']:
            anns[a['image_id']].append(a)
        for _ in range(repeat):
            for image in coco['images']:
                new_id = len(images) + 1
                images.append({**image, 'id': new_id, 'file_name': f"{source}/{image['file_name']}",
                               'orig_id': image['id'], 'source': source})
                for a in anns[image['id']]:
                    annotations.append({**a, 'id': len(annotations) + 1, 'image_id': new_id, 'orig_id': a['id']})
                counts[source] += 1
    train = {'images': images, 'annotations': annotations, 'categories': categories}
    manifest = {'images_per_source': dict(counts), 'images': len(images), 'objects': len(annotations),
                'objects_per_label': dict(sorted(Counter(a['category_id'] for a in annotations).items()))}
    return train, manifest


def build_task_dir(target: Path, train: Coco, val_full: Coco, val_b: Coco, test_full: Path) -> Path:
    """Annotation folder of an improve run: train_task_5_fix.json, val_full.json (all of val, so one prediction pass
    serves Val-A and Val-B), val_task_5.json (Val-B, the per-epoch check) and a link to test_full.json."""
    target.mkdir(parents=True, exist_ok=True)
    save_json(target / 'train_task_5_fix.json', train)
    save_json(target / 'val_full.json', val_full)
    save_json(target / 'val_task_5.json', val_b)
    link = target / 'test_full.json'
    if not link.is_symlink():
        link.symlink_to(Path(test_full).resolve())
    return target
