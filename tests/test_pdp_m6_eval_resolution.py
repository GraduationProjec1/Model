"""M6 (docs/ai/planning/2026-11-26-plan-M1-M6-kiem-prototype.md §2): --eval_resolution predicts a trained
checkpoint at another image size, without retraining; every other use of the checkpoint keeps its resolution."""

import json

import pytest
import torch

import runtime
from pdp_helpers import make_trainer, pdp_args, run_main, use_tiny_detr


@pytest.fixture
def saved_runtime(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    args = pdp_args(tmp_path, "--image_size", "96")
    path = tmp_path / "task_final.pth"
    make_trainer(args, task_id=1).save_task_final(str(path))
    return args, torch.load(path, map_location="cpu", weights_only=False)["runtime"]


def test_off_by_default_keeps_the_saved_processor(saved_runtime):
    args, saved = saved_runtime
    assert args.eval_resolution == 0
    args._runtime_snapshot = saved
    assert runtime.processor(args, None).size == saved["processor_config"]["size"]


def test_predict_only_at_another_resolution_resizes_the_processor(saved_runtime):
    args, saved = saved_runtime
    args.predict_only, args.eval_resolution, args.image_size, args.max_image_size = 1, 800, 800, 800
    args._runtime_snapshot = saved
    runtime.validate_runtime(args, saved)
    assert runtime.processor(args, None).size == {"shortest_edge": 800, "longest_edge": 800}


def test_another_resolution_without_the_flag_still_conflicts(saved_runtime):
    args, saved = saved_runtime
    args.image_size = args.max_image_size = 800
    with pytest.raises(ValueError, match="resolution"):
        runtime.validate_runtime(args, saved)


def test_eval_resolution_is_refused_when_training(tmp_path):
    import main as pdp_main

    args = pdp_args(tmp_path, "--eval_resolution", "800")
    with pytest.raises(SystemExit, match="--eval_resolution"):
        pdp_main.main(args)


@pytest.mark.slow
def test_predict_only_writes_predictions_at_the_new_size(tmp_path, monkeypatch):
    import main as pdp_main
    from autocheckout.predictions import load_predictions

    args = run_main(tmp_path, monkeypatch, "--n_tasks", "1", "--pred_splits", "val")
    task = tmp_path / "run" / "task_1"
    args.predict_only, args.eval_resolution, args.output_dir = 1, 128, str(tmp_path / "run")
    pdp_main.main(args)
    session = json.loads((task / "run_info.json").read_text())["sessions"][-1]
    assert session["processor_size"] == {"shortest_edge": 128, "longest_edge": 128}
    assert load_predictions(task / "pred_val.npz").meta["split"] == "val"
    saved = torch.load(task / "task_final.pth", map_location="cpu", weights_only=False)
    assert saved["runtime"]["processor_config"]["size"]["shortest_edge"] != 128
