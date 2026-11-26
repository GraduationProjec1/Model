"""tools/predict_resolution.py (M6 of docs/ai/planning/2026-11-26-plan-M1-M6-kiem-prototype.md §2.2): predictions
of a trained run at another image size, with the arguments the run saved in its run_info.json."""

import json

import pytest
import torch

from pdp_helpers import run_main

pytestmark = pytest.mark.slow


def test_predicts_a_finished_run_at_another_resolution(tmp_path, monkeypatch):
    from autocheckout.predictions import load_predictions
    from tools import predict_resolution

    run_main(tmp_path, monkeypatch, "--n_tasks", "1", "--pred_splits", "val")
    source = tmp_path / "run" / "task_1"
    run_info = json.loads((source / "run_info.json").read_text())
    run_info["sessions"][-1]["args"]["train_rows"] = str(tmp_path / "gone/train_rows.json")
    (source / "run_info.json").write_text(json.dumps(run_info))
    out = tmp_path / "res128"
    task = predict_resolution.main(["--run-info", str(source / "run_info.json"),
                                    "--checkpoint", str(source / "task_final.pth"),
                                    "--resolution", "128", "--out", str(out)])
    assert task == out / "task_1"
    session = json.loads((task / "run_info.json").read_text())["sessions"][-1]
    assert session["mode"] == "predict"
    assert session["processor_size"] == {"shortest_edge": 128, "longest_edge": 128}
    assert load_predictions(task / "pred_val.npz").meta["task_id"] == 1
    assert (task / "task_final.pth").resolve() == (source / "task_final.pth").resolve()
    assert not (task / "pred_test.npz").exists()
    saved = torch.load(source / "task_final.pth", map_location="cpu", weights_only=False)
    assert saved["runtime"]["processor_config"]["size"]["shortest_edge"] != 128


def test_refuses_to_overwrite_another_checkpoint(tmp_path):
    from tools import predict_resolution

    task = tmp_path / "out" / "task_1"
    task.mkdir(parents=True)
    (task / "task_final.pth").write_bytes(b"other")
    run_info = tmp_path / "run_info.json"
    run_info.write_text(json.dumps({"sessions": [{"args": {"start_task": 1, "n_tasks": 1}}]}))
    other = tmp_path / "task_final.pth"
    other.write_bytes(b"mine")
    with pytest.raises(SystemExit, match="task_final.pth"):
        predict_resolution.main(["--run-info", str(run_info), "--checkpoint", str(other), "--resolution", "800",
                                 "--out", str(tmp_path / "out")])
