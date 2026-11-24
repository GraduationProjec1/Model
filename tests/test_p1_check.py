"""tools/p1_check.py (G0 of docs/project/ke-hoach-chay-25-11-2026.md): P1 speed and same outputs on a real checkpoint.

A tiny checkpoint and toy images stand in for r3_real and Val-A.
"""

import json

import pytest

from pdp_helpers import make_toy_dataset, make_trainer, pdp_args, use_tiny_detr

pytestmark = pytest.mark.slow


@pytest.fixture
def checkpoint_and_images(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    toy = make_toy_dataset(tmp_path / "toy")
    path = tmp_path / "task_final.pth"
    make_trainer(pdp_args(tmp_path), task_id=1).save_task_final(str(path))
    val = toy["tasks"] / "val_full.json"
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"pool": [image["id"] for image in json.loads(val.read_text())["images"]][:3]}))
    return path, val, toy["images"], split


def run(tmp_path, checkpoint_and_images, *extra):
    from tools import p1_check

    path, val, images, split = checkpoint_and_images
    return p1_check.main(["--checkpoint", str(path), "--val", str(val), "--img-root", str(images),
                          "--split", str(split), "--n", "2", "--warmup", "1", "--device", "cpu",
                          "--out", str(tmp_path / "out"), *extra])


def test_reuse_gives_the_same_outputs_and_writes_a_summary(tmp_path, checkpoint_and_images):
    summary = run(tmp_path, checkpoint_and_images)
    assert summary["images"] == 2
    assert summary["passed"] is True and summary["topk_identical"] == 2 and summary["max_abs_diff"] <= 1e-4
    assert summary["calls_new"]["backbone"] == 1 and summary["calls_old"]["backbone"] == 2
    saved = json.loads((tmp_path / "out/p1_check_summary.json").read_text())
    assert saved["passed"] is True and (tmp_path / "out/p1_check.csv").is_file()


def test_fails_when_the_outputs_differ_beyond_the_tolerance(tmp_path, checkpoint_and_images, monkeypatch):
    from tools import p1_check

    real = p1_check.compare

    def shifted(old, new, topk):
        result = real(old, new, topk)
        return {**result, "max_abs_diff": 1.0}

    monkeypatch.setattr(p1_check, "compare", shifted)
    summary = run(tmp_path, checkpoint_and_images)
    assert summary["passed"] is False


def test_main_exit_code_follows_the_result(tmp_path, checkpoint_and_images, monkeypatch):
    from tools import p1_check

    monkeypatch.setattr(p1_check, "main", lambda argv=None: {"passed": False})
    with pytest.raises(SystemExit) as stop:
        p1_check.cli()
    assert stop.value.code == 2
