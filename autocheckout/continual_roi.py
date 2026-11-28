"""Stage 2 Continual Crop Recognizer for Decoupled Continual Object Detection.

Supports:
- L2-normalized ``CosineClassifier`` with optional ArcFace angular margin during training and
  ``add_classes(new_weights)`` that expands the classifier without mutating old class logits.
- Dedicated ``background`` class slot to reject Stage-1 false-positive / shifted boxes.
- Two continual learning modes:
  * Mode A (Online Prototype): freeze backbone, compute L2-normalized mean prototypes for new classes,
    append directly to ``CosineClassifier`` (exact 0% forgetting on old classes).
  * Mode B (Exemplar Replay + Logit KD): iCaRL-style herding exemplar bank (e.g. 20 crops/SKU) +
    knowledge distillation on old-class logits to separate fine-grained new-vs-old pairs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from autocheckout.roi_refine import CropRecord, RoICropDataset, resolve_image_path


class CosineClassifier(nn.Module):
    """L2-normalized cosine classifier with optional ArcFace margin and continual class expansion."""

    def __init__(self, in_features: int, num_classes: int, scale: float = 20.0, margin: float = 0.0):
        super().__init__()
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        if not (0.0 <= margin < 1.0):
            raise ValueError(f"margin must be in [0, 1), got {margin}")
        self.in_features = in_features
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    @property
    def num_classes(self) -> int:
        return int(self.weight.shape[0])

    def forward(self, features: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        z = F.normalize(features, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        cos = F.linear(z, w)
        if targets is not None and self.training and self.margin > 0.0:
            cos_clamped = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            theta = torch.acos(cos_clamped)
            target_cos = torch.cos(theta + self.margin)
            one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).to(cos.dtype)
            cos = cos * (1.0 - one_hot) + target_cos * one_hot
        return cos * self.scale

    @torch.no_grad()
    def add_classes(self, new_weights: torch.Tensor) -> None:
        """Append ``[K, in_features]`` class weight vectors without altering existing class rows."""
        if new_weights.ndim != 2 or new_weights.shape[1] != self.in_features:
            raise ValueError(
                f"new_weights must have shape [K, {self.in_features}], got {tuple(new_weights.shape)}"
            )
        nw = F.normalize(new_weights.detach().to(device=self.weight.device, dtype=self.weight.dtype), dim=-1)
        combined = torch.cat([self.weight.detach(), nw], dim=0)
        self.weight = nn.Parameter(combined)

    @torch.no_grad()
    def set_class_weights(self, start_idx: int, new_weights: torch.Tensor) -> None:
        """Overwrite rows ``[start_idx : start_idx + K]`` with L2-normalized prototypes."""
        k = int(new_weights.shape[0])
        if start_idx < 0 or start_idx + k > self.num_classes:
            raise ValueError(f"slice [{start_idx}:{start_idx + k}] out of range for {self.num_classes} classes")
        nw = F.normalize(new_weights.detach().to(device=self.weight.device, dtype=self.weight.dtype), dim=-1)
        self.weight[start_idx : start_idx + k].copy_(nw)


def _build_backbone(arch: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    if arch == "tiny_test":
        net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        return net, 16
    if arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        net = models.convnext_tiny(weights=weights)
        dim = net.classifier[2].in_features
        net.classifier[2] = nn.Identity()
        return net, dim
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, dim
    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, dim
    if arch == "dinov2_vits14":
        net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=pretrained)
        return net, 384
    raise ValueError(f"Unsupported continual RoI backbone arch: {arch!r}")


class ContinualCropModel(nn.Module):
    """Backbone + L2-normalized ``CosineClassifier`` with optional trailing ``background`` class."""

    def __init__(
        self,
        arch: str = "convnext_tiny",
        num_classes: int = 100,
        *,
        include_background: bool = False,
        scale: float = 20.0,
        margin: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.arch = arch
        self.include_background = include_background
        self.backbone, self.feat_dim = _build_backbone(arch, pretrained=pretrained)
        total_slots = num_classes + (1 if include_background else 0)
        self.head = CosineClassifier(self.feat_dim, total_slots, scale=scale, margin=margin)

    @property
    def num_fg_classes(self) -> int:
        return self.head.num_classes - (1 if self.include_background else 0)

    @property
    def bg_class_index(self) -> int | None:
        return self.head.num_classes - 1 if self.include_background else None

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.head(feats, targets=targets)

    @torch.no_grad()
    def expand_fg_classes(self, new_prototypes: torch.Tensor) -> None:
        """Append ``K`` new foreground classes while keeping the ``background`` slot (if any) at the end."""
        if not self.include_background:
            self.head.add_classes(new_prototypes)
            return
        w = self.head.weight.detach()
        fg = w[:-1]
        bg = w[-1:]
        nw = F.normalize(new_prototypes.detach().to(device=w.device, dtype=w.dtype), dim=-1)
        self.head.weight = nn.Parameter(torch.cat([fg, nw, bg], dim=0))


@torch.no_grad()
def extract_embeddings(
    model: ContinualCropModel,
    records: Sequence[CropRecord],
    *,
    img_size: int = 224,
    pad_ratio: float = 0.10,
    batch_size: int = 64,
    num_workers: int = 0,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return L2-normalized embeddings ``[N, D]`` and integer labels ``[N]`` for ``records``."""
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = model.to(dev).eval()
    ds = RoICropDataset(records, img_size=img_size, pad_ratio=pad_ratio, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for imgs, targets in loader:
        feats = F.normalize(model.extract_features(imgs.to(dev)), dim=-1)
        embs.append(feats.float().cpu().numpy())
        labels.append(targets.numpy())
    if not embs:
        return np.zeros((0, model.feat_dim), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return np.concatenate(embs, axis=0), np.concatenate(labels, axis=0)


def compute_prototypes_from_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_ids: Sequence[int],
) -> torch.Tensor:
    """Compute L2-normalized class prototypes ``[len(class_ids), D]`` from embeddings."""
    dim = embeddings.shape[1]
    protos = np.zeros((len(class_ids), dim), dtype=np.float32)
    for row, cid in enumerate(class_ids):
        mask = labels == int(cid)
        if np.any(mask):
            mean = embeddings[mask].mean(axis=0)
            norm = float(np.linalg.norm(mean))
            protos[row] = mean / norm if norm > 1e-12 else mean
    return torch.from_numpy(protos)


def select_exemplars_herding(
    records: Sequence[CropRecord],
    embeddings: np.ndarray,
    per_class: int = 20,
) -> list[CropRecord]:
    """iCaRL herding: for each class, iteratively pick up to ``per_class`` crops whose running mean
    best approximates the full class embedding mean."""
    if per_class <= 0 or len(records) == 0:
        return []
    by_cls: dict[int, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        by_cls[int(rec.label)].append(idx)

    selected: list[CropRecord] = []
    for cls in sorted(by_cls):
        indices = by_cls[cls]
        if len(indices) <= per_class:
            selected.extend(records[i] for i in indices)
            continue
        cls_emb = embeddings[indices].astype(np.float64)
        mean = cls_emb.mean(axis=0)
        chosen_local: list[int] = []
        used = np.zeros(len(indices), dtype=bool)
        running_sum = np.zeros_like(mean)
        for step in range(1, per_class + 1):
            # Target: (running_sum + x_i) / step close to mean => minimize ||step*mean - running_sum - x_i||^2
            target = step * mean - running_sum
            dists = np.linalg.norm(cls_emb - target[None, :], axis=1)
            dists[used] = np.inf
            best = int(np.argmin(dists))
            used[best] = True
            chosen_local.append(best)
            running_sum += cls_emb[best]
        selected.extend(records[indices[i]] for i in chosen_local)
    return selected


def kd_logit_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    num_old_classes: int,
    temp: float = 2.0,
) -> torch.Tensor:
    """Temperature-scaled KL distillation on the first ``num_old_classes`` logits."""
    if num_old_classes <= 0:
        return student_logits.sum() * 0.0
    s_logp = F.log_softmax(student_logits[:, :num_old_classes] / temp, dim=-1)
    t_prob = F.softmax(teacher_logits[:, :num_old_classes] / temp, dim=-1)
    return F.kl_div(s_logp, t_prob, reduction="batchmean") * (temp * temp)


def _box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def collect_background_crops(
    pred_boxes_by_img: dict[int, np.ndarray],
    gt_coco: dict[str, Any],
    img_root: Path,
    bg_label: int,
    *,
    iou_thresh: float = 0.30,
    max_per_image: int = 3,
) -> list[CropRecord]:
    """Extract Stage-1 false-positive crops (``max IoU < iou_thresh`` with all GT) as ``background`` samples."""
    id_to_file = {int(im["id"]): im["file_name"] for im in gt_coco["images"]}
    gt_by_img: dict[int, list[list[float]]] = defaultdict(list)
    for ann in gt_coco["annotations"]:
        x, y, w, h = (float(v) for v in ann["bbox"])
        gt_by_img[int(ann["image_id"])].append([x, y, x + w, y + h])

    records: list[CropRecord] = []
    for img_id, pred_boxes in sorted(pred_boxes_by_img.items()):
        if img_id not in id_to_file or len(pred_boxes) == 0:
            continue
        p_arr = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
        gt_arr = np.asarray(gt_by_img.get(img_id, []), dtype=np.float32).reshape(-1, 4)
        if len(gt_arr) > 0:
            max_iou = _box_iou_matrix(p_arr, gt_arr).max(axis=1)
        else:
            max_iou = np.zeros(len(p_arr), dtype=np.float32)
        neg_idx = np.where(max_iou < iou_thresh)[0][:max_per_image]
        img_path = str(resolve_image_path(img_root, id_to_file[img_id]))
        for idx in neg_idx:
            x1, y1, x2, y2 = (float(v) for v in p_arr[idx])
            if (x2 - x1) > 1.0 and (y2 - y1) > 1.0:
                records.append(CropRecord(img_path, (x1, y1, x2, y2), bg_label, int(img_id)))
    return records


def compute_cl_metrics(acc_matrix: Sequence[Sequence[float | None]]) -> dict[str, float]:
    """Compute standard CL metrics from lower-triangular ``T x T`` accuracy matrix ``A[t][k]``
    (accuracy on task ``k`` after learning task ``t``): ``ACC``, ``Forgetting``, ``BWT``."""
    t_count = len(acc_matrix)
    if t_count == 0:
        return {"ACC": 0.0, "Forgetting": 0.0, "BWT": 0.0}
    last = [float(acc_matrix[-1][k]) for k in range(t_count) if acc_matrix[-1][k] is not None]
    acc = float(np.mean(last)) if last else 0.0
    if t_count == 1:
        return {"ACC": acc, "Forgetting": 0.0, "BWT": 0.0}
    forgetting_vals: list[float] = []
    bwt_vals: list[float] = []
    for k in range(t_count - 1):
        hist = [float(acc_matrix[t][k]) for t in range(k, t_count) if acc_matrix[t][k] is not None]
        if len(hist) >= 2:
            forgetting_vals.append(max(hist[:-1]) - hist[-1])
            bwt_vals.append(hist[-1] - hist[0])
    return {
        "ACC": acc,
        "Forgetting": float(np.mean(forgetting_vals)) if forgetting_vals else 0.0,
        "BWT": float(np.mean(bwt_vals)) if bwt_vals else 0.0,
    }
