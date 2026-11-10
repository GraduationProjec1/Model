"""tools/run_split.py: the chia 12+13 SKU campaign (docs/design/2026-11-09-chia-12-13-sku/01-plan.md)."""

import json

import pytest

from autocheckout import full_syn
from autocheckout.datapaths import Layout
from tools import run_split


def test_real_settings_point_at_the_12x_data_and_the_r1_seed(tmp_path):
    data = Layout.at(tmp_path / "data")
    settings = run_split.real_settings("configs/training/full_syn_12x.json", data)
    assert settings["campaign_id"] == "full-syn-12x" and settings["dry_run"] is False
    assert settings["release_root"] == str(data.root / "synth_v2_12x")
    assert settings["eval_root"] == str(data.root / "eval_12x")
    assert settings["seed_campaign"] == str(data.root / "models/full-syn-r1")
    assert settings["seed_task_config"] == str(data.root / "eval/task_config.json")
    assert not any(k.startswith("_") for k in settings)
    # same training settings as r1 (models/full-syn-r1/plan.json), only the campaign differs
    for key, value in dict(round=1, resolution=640, batch_size=8, effective_batch=8, fsa_batch_size=4, num_workers=8,
                           tf32=True, fsa_epochs=8, epochs=16, checkpoint_every_minutes=2, audit_images=1000,
                           pdp_experiment=full_syn.PDP).items():
        assert settings[key] == value
    dry = tmp_path / "dry.json"
    dry.write_text(json.dumps({"campaign_id": "x", "dry_run": True}))
    with pytest.raises(ValueError, match="dry-run"):
        run_split.real_settings(str(dry), data)


@pytest.mark.parametrize("mine,accept,decision", [(0.71, False, "go"), (0.70, False, "stop"), (0.70, True, "go")])
def test_mid_gate_compares_125_skus_with_r1_task_2(tmp_path, monkeypatch, mine, accept, decision):
    plan = {"root": str(tmp_path), "runs": str(tmp_path / "runs"), "pdp_experiment": full_syn.PDP}
    (tmp_path / "stages").mkdir()
    calls = []

    def fake(data, name, pred, stage, out, *, split, groups="r1", target="valb", grid_min=None):
        calls.append((name, stage, split, groups, target))
        return {"cAcc_m2": mine if split else 0.7323, "target_images": 2271}

    monkeypatch.setattr(run_split, "_score", fake)
    if decision == "stop":
        with pytest.raises(full_syn.GateStop, match="accept-mid-gate"):
            run_split.mid_gate(plan, Layout.at(tmp_path / "data"), accept)
    else:
        gate = run_split.mid_gate(plan, Layout.at(tmp_path / "data"), accept)
        assert gate["accepted_by_flag"] == accept
        calls.clear()
        assert run_split.mid_gate(plan, Layout.at(tmp_path / "data"), False) == gate  # kept on resume
        assert calls == []
    saved = json.loads((tmp_path / "stages/mid_gate_125.json").read_text())
    assert saved["decision"] == decision and saved["max_drop"] == run_split.MID_DROP
    if calls:
        assert ("12x_t3_pool", 3, True, "r1", "pool") in calls and ("r1_t2_pool", 2, False, "r1", "pool") in calls


def test_milestones_score_both_groupings_and_grids_next_to_r1(tmp_path, monkeypatch):
    plan = {"root": str(tmp_path), "runs": str(tmp_path / "runs"), "pdp_experiment": full_syn.PDP}
    for stage in (3, 5):
        (tmp_path / f"runs/{full_syn.PDP}/task_{stage}").mkdir(parents=True)
        (tmp_path / f"runs/{full_syn.PDP}/task_{stage}/pred_val.npz").write_bytes(b"")
    (tmp_path / "split12").mkdir()

    def fake(data, name, pred, stage, out, *, split, groups="r1", target="valb", grid_min=None):
        return {"cAcc_m2": 0.5 if split else 0.6, "target_images": 100, "at_floor": ["task_1"] if grid_min else []}

    monkeypatch.setattr(run_split, "_score", fake)
    result = run_split.milestones(plan, Layout.at(tmp_path / "data"))
    rows = result["rows"]
    assert [(r["stage"], r["r1_stage"], r["skus"], r["grid"]) for r in rows] == [
        (3, 2, 125, "grid10"), (3, 2, 125, "grid02"), (5, 3, 150, "grid10"), (5, 3, 150, "grid02")]
    assert all(r["r1"] == 0.6 and r["12x_r1"] == 0.5 and r["12x_task"] == 0.5 for r in rows)
    table = (tmp_path / "split12/MILESTONES.md").read_text()
    assert "| 125 | 3 | grid10 | 60.00% | 50.00% | 50.00% | 100 / 100 | - |" in table
    assert "task_1" in table


@pytest.mark.slow
def test_dry_run_goes_through_the_whole_chain(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"campaign_id": "full-syn-12x", "artifact_root": str(tmp_path)}))
    result = run_split.dry_run(str(settings))
    assert result["tasks"] == [2, 3, 4, 5] and "equal" in result["task1"]
    root = tmp_path / "dry_run/split12-dry"
    assert (root / "smoke_split/smoke_report.json").is_file()
    assert json.loads((root / "stages/task1.json").read_text())["gate"]["decision"] == "go"
    assert run_split.dry_run(str(settings))["tasks"] == [2, 3, 4, 5]  # resumes, trains nothing again
