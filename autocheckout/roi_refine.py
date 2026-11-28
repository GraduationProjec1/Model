"""M6.2 RoI-Refine: post-hoc high-resolution crop classifier for ambiguous detector queries.

Never modifies detector weights (0% forgetting). Only refines queries whose detector top-1 vs top-2
sigmoid probability margin is below ``margin_thresh`` and whose max probability is at least ``min_prob``.
Fusion is optionally restricted to the detector's top-K classes per query so an out-of-distribution crop
prediction cannot override a clear non-candidate class.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class CropRecord:
    image_path: str
    box_xyxy: tuple[float, float, float, float]
    label: int
    image_id: int


@dataclass(frozen=True)
class RefineConfig:
    margin_thresh: float = 0.35
    alpha: float = 0.5
    temp: float = 1.0
    topk: int = 3
    min_prob: float = 0.15
    pad_ratio: float = 0.10
    img_size: int = 224
    arch: str = "convnext_tiny"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> RefineConfig:
        return cls(**json.loads(text))


def assert_no_overlap(train_ids: Iterable[int], forbidden_ids: Iterable[int], label: str = "eval") -> None:
    """Strict anti-leakage check: raises ``ValueError`` if any training image id appears in ``forbidden_ids``."""
    overlap = set(int(i) for i in train_ids) & set(int(i) for i in forbidden_ids)
    if overlap:
        sample = sorted(overlap)[:10]
        raise ValueError(f"RoI-Refine data leakage: {len(overlap)} train images overlap {label} (e.g. {sample})")


def crop_box(image: Image.Image, box_xyxy: Sequence[float], pad_ratio: float = 0.10) -> Image.Image:
    """Crop ``[x1, y1, x2, y2]`` from ``image`` with relative padding, safely clamped to image bounds."""
    w_img, h_img = image.size
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    px = bw * pad_ratio
    py = bh * pad_ratio
    left = max(0, min(w_img - 1, int(np.floor(x1 - px))))
    top = max(0, min(h_img - 1, int(np.floor(y1 - py))))
    right = max(left + 1, min(w_img, int(np.ceil(x2 + px))))
    bottom = max(top + 1, min(h_img, int(np.ceil(y2 + py))))
    return image.crop((left, top, right, bottom))


def resolve_image_path(img_root: Path, file_name: str) -> Path:
    """Resolve COCO ``file_name`` under ``img_root`` (supporting ``val2019/`` or direct paths)."""
    direct = img_root / file_name
    if direct.is_file():
        return direct
    for sub in ("val2019", "train2019", "test2019", "images"):
        cand = img_root / sub / file_name
        if cand.is_file():
            return cand
    return direct


def build_crop_records(
    coco_dict: dict[str, Any],
    img_root: Path,
    allowed_image_ids: Iterable[int] | None = None,
    *,
    num_classes: int = 200,
    cat_to_label: dict[int, int] | None = None,
    max_per_class: int | None = None,
    seed: int = 0,
) -> list[CropRecord]:
    """Extract GT ``CropRecord`` list from a COCO dict, optionally filtered by image ids and capped per class."""
    allowed = None if allowed_image_ids is None else {int(i) for i in allowed_image_ids}
    id_to_file = {int(im["id"]): im["file_name"] for im in coco_dict["images"]}
    by_class: dict[int, list[CropRecord]] = defaultdict(list)

    for ann in coco_dict["annotations"]:
        img_id = int(ann["image_id"])
        if allowed is not None and img_id not in allowed:
            continue
        raw_cat = int(ann["category_id"])
        label = cat_to_label[raw_cat] if cat_to_label is not None else raw_cat
        if not (0 <= label < num_classes):
            continue
        x, y, w, h = (float(v) for v in ann["bbox"])
        if w <= 1.0 or h <= 1.0:
            continue
        img_path = str(resolve_image_path(img_root, id_to_file[img_id]))
        by_class[label].append(CropRecord(img_path, (x, y, x + w, y + h), label, img_id))

    rng = np.random.default_rng(seed)
    records: list[CropRecord] = []
    for cls in sorted(by_class):
        items = by_class[cls]
        if max_per_class is not None and len(items) > max_per_class:
            idx = rng.choice(len(items), size=max_per_class, replace=False)
            items = [items[int(i)] for i in sorted(idx)]
        records.extend(items)
    return records


def build_transforms(img_size: int = 224, train: bool = False) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class RoICropDataset(Dataset):
    """On-the-fly high-res crop dataset from ``CropRecord`` entries."""

    def __init__(self, records: Sequence[CropRecord], img_size: int = 224, pad_ratio: float = 0.10, train: bool = False):
        self.records = list(records)
        self.pad_ratio = pad_ratio
        self.transform = build_transforms(img_size=img_size, train=train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self.records[idx]
        with Image.open(rec.image_path) as im:
            rgb = im.convert("RGB")
            crop = crop_box(rgb, rec.box_xyxy, pad_ratio=self.pad_ratio)
        return self.transform(crop), rec.label


def build_roi_classifier(arch: str = "convnext_tiny", num_classes: int = 200, pretrained: bool = True) -> nn.Module:
    """Build a lightweight crop classifier. ``arch='tiny_test'`` is a fast offline CNN for unit tests."""
    if arch == "tiny_test":
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )
    if arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        net = models.convnext_tiny(weights=weights)
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, num_classes)
        return net
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    raise ValueError(f"Unsupported RoI-Refine arch: {arch!r}")


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def ambiguous_query_mask(
    det_logits: np.ndarray,
    *,
    margin_thresh: float = 0.35,
    min_prob: float = 0.15,
) -> np.ndarray:
    """Boolean mask ``[...,]`` over queries where ``p1 >= min_prob`` and ``(p1 - p2) < margin_thresh``."""
    if margin_thresh <= 0.0:
        return np.zeros(det_logits.shape[:-1], dtype=bool)
    probs = _sigmoid_np(det_logits.astype(np.float32))
    if probs.shape[-1] < 2:
        return np.zeros(det_logits.shape[:-1], dtype=bool)
    part = np.partition(probs, -2, axis=-1)
    p1 = part[..., -1]
    p2 = part[..., -2]
    return (p1 >= min_prob) & ((p1 - p2) < margin_thresh)


def fuse_logits(
    det_logits: np.ndarray,
    roi_logits: np.ndarray,
    *,
    margin_thresh: float = 0.35,
    alpha: float = 0.5,
    temp: float = 1.0,
    topk: int = 3,
    min_prob: float = 0.15,
) -> np.ndarray:
    """Fuse detector logits and crop classifier logits on ambiguous queries only (does not mutate inputs).

    When ``alpha == 0`` or ``margin_thresh <= 0``, returns a copy identical to ``det_logits``.
    When ``topk > 0``, only the detector's top-K classes per ambiguous query are updated; all other
    classes keep ``det_logits`` unchanged, preventing wild label flips outside the detector's candidates.
    """
    if temp <= 0:
        raise ValueError(f"temp must be > 0, got {temp}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    det = np.asarray(det_logits, dtype=np.float32)
    roi = np.asarray(roi_logits, dtype=np.float32)
    if det.shape != roi.shape:
        raise ValueError(f"det_logits {det.shape} and roi_logits {roi.shape} must have the same shape")
    out = det.copy()
    if alpha == 0.0 or margin_thresh <= 0.0:
        return out

    mask = ambiguous_query_mask(det, margin_thresh=margin_thresh, min_prob=min_prob)
    if not np.any(mask):
        return out

    scaled_roi = roi / float(temp)
    flat_det = det[mask]
    flat_roi = scaled_roi[mask]
    if topk > 0 and topk < det.shape[-1]:
        # Restrict fusion to the detector's top-K classes per ambiguous query.
        top_idx = np.argpartition(flat_det, -topk, axis=-1)[:, -topk:]
        rows = np.arange(flat_det.shape[0])[:, None]
        fused_sub = flat_det.copy()
        # Center roi logits on the top-K subset around the detector's top-K mean so score scale stays aligned
        # with counting thresholds while changing the relative ranking among the top-K candidates.
        det_k = flat_det[rows, top_idx]
        roi_k = flat_roi[rows, top_idx]
        roi_k_aligned = roi_k - roi_k.mean(axis=-1, keepdims=True) + det_k.mean(axis=-1, keepdims=True)
        fused_sub[rows, top_idx] = (1.0 - alpha) * det_k + alpha * roi_k_aligned
        out[mask] = fused_sub
    else:
        out[mask] = (1.0 - alpha) * flat_det + alpha * flat_roi
    return out
