"""tools/proto_check.py (K0 of docs/ai/planning/2026-11-26-plan-M1-M6-kiem-prototype.md): are the class prototypes
of a checkpoint good anchors for the prototype margin loss (M1)?"""

import json

import pytest
import torch

from pdp_helpers import make_toy_dataset, make_trainer, pdp_args, use_tiny_detr


def test_coverage_lists_empty_classes_per_task():
    from tools.proto_check import coverage

    counts = {0: 5, 1: 0, 2: 3, 3: 2, 4: 0}
    result = coverage(counts, task_sizes=[3, 2], seen=5, few=3)
    assert result["classes_with_prototype"] == 3 and result["seen_classes"] == 5
    assert result["empty"] == [1, 4]
    assert result["per_task"] == [{"task": 1, "classes": 3, "with_prototype": 2},
                                  {"task": 2, "classes": 2, "with_prototype": 1}]
    assert result["few"] == [3]


def test_normalized_anchors_skip_empty_prototypes_and_never_give_nan():
    from tools.proto_check import normalized_anchors

    prototypes = {0: torch.tensor([3.0, 0.0]), 1: torch.zeros(2), 2: torch.tensor([0.0, -2.0])}
    anchors, ids = normalized_anchors(prototypes, valid=[0, 2])
    assert ids == [0, 2]
    assert torch.allclose(anchors, torch.tensor([[1.0, 0.0], [0.0, -1.0]]))
    assert torch.isfinite(anchors).all()


def test_closest_pairs_are_sorted_and_self_similarity_is_one():
    from tools.proto_check import closest_pairs

    anchors = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]), dim=1)
    assert torch.allclose((anchors @ anchors.T).diagonal(), torch.ones(3))
    pairs = closest_pairs(anchors, ids=[10, 11, 12], k=2)
    assert [(p["a"], p["b"]) for p in pairs] == [(10, 11), (11, 12)]
    assert pairs[0]["cosine"] >= pairs[1]["cosine"]


def test_anchor_accuracy_and_margin():
    from tools.proto_check import anchor_scores

    anchors = torch.eye(3)
    features = torch.tensor([[1.0, 0.2, 0.0], [0.0, 0.1, 1.0], [0.9, 1.0, 0.0]])
    labels = torch.tensor([0, 2, 0])
    result = anchor_scores(features, labels, anchors, ids=[0, 1, 2])
    assert result["matched"] == 3 and result["accuracy"] == pytest.approx(2 / 3)
    assert result["margin_median"] > 0
    assert result["margins"][2] < 0


def test_anchor_scores_ignore_labels_without_anchor():
    from tools.proto_check import anchor_scores

    result = anchor_scores(torch.eye(2), torch.tensor([0, 5]), torch.eye(2), ids=[0, 1])
    assert result["matched"] == 1 and result["skipped"] == 1 and result["accuracy"] == 1.0


@pytest.mark.parametrize(("cls", "proto", "weight", "expected"), [
    (0.80, 0.78, 0.60, "prototype"),
    (0.80, 0.60, 0.79, "class_embed"),
    (0.80, 0.60, 0.50, "neither"),
])
def test_verdict_follows_the_k0_gate(cls, proto, weight, expected):
    from tools.proto_check import verdict

    assert verdict(cls, proto, weight, tol=0.05) == expected


@pytest.fixture
def checkpoint_and_images(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    toy = make_toy_dataset(tmp_path / "toy")
    trainer = make_trainer(pdp_args(tmp_path), task_id=1)
    generator = torch.Generator().manual_seed(0)
    for c in (0, 1):
        trainer.class_prototypes[c] = torch.randn(trainer.model.config.d_model, generator=generator)
        trainer.class_cache_count[c] = 4
    path = tmp_path / "task_final.pth"
    trainer.save_task_final(str(path))
    val = toy["tasks"] / "val_full.json"
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"calib": [image["id"] for image in json.loads(val.read_text())["images"]]}))
    return path, val, toy["images"], split


@pytest.mark.slow
def test_main_writes_a_summary_on_a_tiny_checkpoint(tmp_path, checkpoint_and_images):
    from tools import proto_check

    path, val, images, split = checkpoint_and_images
    out = tmp_path / "out"
    summary = proto_check.main(["--checkpoint", str(path), "--val", str(val), "--img-root", str(images),
                                "--split", str(split), "--split-key", "calib", "--n", "3", "--device", "cpu",
                                "--out", str(out)])
    assert summary["coverage"]["seen_classes"] == 3
    assert summary["coverage"]["empty"] == [2]
    assert summary["images"] == 3 and summary["matched"] > 0
    assert summary["verdict"] in {"prototype", "class_embed", "neither"}
    for key in ("acc_cls", "acc_proto", "acc_weight"):
        assert 0.0 <= summary[key] <= 1.0
    saved = json.loads((out / "proto_check_summary.json").read_text())
    assert saved["verdict"] == summary["verdict"]
