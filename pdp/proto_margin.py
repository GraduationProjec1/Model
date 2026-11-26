"""M1 (docs/ai/planning/2026-11-26-plan-M1-M6-kiem-prototype.md §3): prototype margin loss.

Focal loss treats every wrong class alike; this loss asks the decoder output of a matched query to be closer to the
anchor of its class than to the anchors of the most similar other classes (the SKUs it gets swapped with), by a
margin. Anchors are unit vectors fixed for the run: the class prototypes (mean decoder output of the correctly
predicted queries, saved in the checkpoint) or the classifier rows (--proto_anchor class_embed, K0 verdict).
"""

import torch
import torch.nn.functional as F


def anchor_matrix(vectors, valid, seen):
    """Unit anchors [len(valid), D] of the ``valid`` classes (sorted), detached, and slot [seen]: the anchor row of
    each class, -1 for a class without an anchor."""
    ids = sorted(valid)
    anchors = F.normalize(torch.stack([vectors[c].detach().float().cpu() for c in ids]), dim=1)
    slot = torch.full((seen,), -1, dtype=torch.long)
    slot[ids] = torch.arange(len(ids))
    return anchors, slot


def margin_loss(features, labels, anchors, slot, *, margin, temp, topk):
    """Mean over the queries whose class has an anchor of
    -log softmax([(cos(z, a_c) - margin), cos(z, a_k) for the ``topk`` closest k != c] / temp)[0].
    Zero (with a zero gradient) when no query has an anchor."""
    slot = slot.to(labels.device)
    inside = labels < len(slot)
    rows = torch.full_like(labels, -1)
    rows[inside] = slot[labels[inside]]
    keep = rows >= 0
    if not keep.any():
        return features.sum() * 0
    target = rows[keep]
    similarity = F.normalize(features[keep], dim=1) @ anchors.to(features.dtype).T
    positive = similarity.gather(1, target[:, None]) - margin
    k = min(topk, len(anchors) - 1)
    negatives = similarity.scatter(1, target[:, None], float('-inf')).topk(k, dim=1).values
    logits = torch.cat([positive, negatives], dim=1) / temp
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))
