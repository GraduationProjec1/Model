"""W1: optional W&B logger (live loss/AP while a task trains). Off by default: only the CSV log is written."""

import json

import pytest
from lightning.pytorch.loggers import CSVLogger

import main as pdp_main
from pdp_helpers import main_args


def _args(tmp_path, extra=()):
    args = main_args(["--accelerator", "cpu", *extra])
    args.output_dir = str(tmp_path / "al_margin" / "task_2")
    args.task = "2"
    (tmp_path / "al_margin" / "task_2").mkdir(parents=True)
    return args


def test_default_is_csv_only(tmp_path, monkeypatch):
    monkeypatch.delenv("AC_WANDB_PROJECT", raising=False)
    args = _args(tmp_path)
    assert args.wandb_project == ""
    loggers = pdp_main.make_loggers(args)
    assert len(loggers) == 1 and isinstance(loggers[0], CSVLogger)
    assert not (tmp_path / "al_margin" / "task_2" / "wandb_run.json").exists()


def test_env_turns_it_on(monkeypatch):
    monkeypatch.setenv("AC_WANDB_PROJECT", "autocheckout")
    assert main_args(["--accelerator", "cpu"]).wandb_project == "autocheckout"


def test_wandb_logger_names_run_and_keeps_id_for_resume(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    from lightning.pytorch.loggers import WandbLogger

    monkeypatch.setenv("WANDB_MODE", "offline")  # no account / network in tests
    args = _args(tmp_path, ["--wandb_project", "autocheckout"])
    loggers = pdp_main.make_loggers(args)
    assert isinstance(loggers[0], CSVLogger) and isinstance(loggers[1], WandbLogger)
    wb = loggers[1]
    saved = json.loads((tmp_path / "al_margin" / "task_2" / "wandb_run.json").read_text())
    assert saved["project"] == "autocheckout" and saved["name"] == "al_margin/task_2" and saved["group"] == "al_margin"
    assert wb._wandb_init["id"] == saved["id"] and wb._wandb_init["resume"] == "allow"
    assert wb._wandb_init["config"]["task"] == "2"
    assert "log_file" not in wb._wandb_init["config"]  # only plain values go to W&B

    # resumed session (same task folder) continues the same W&B run instead of starting a new one
    again = pdp_main.make_loggers(args)[1]
    assert again._wandb_init["id"] == saved["id"]


def test_missing_wandb_package_fails_before_training(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_wandb(name, *a, **k):
        if name == "wandb" or name.startswith("wandb."):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_wandb)
    args = _args(tmp_path, ["--wandb_project", "autocheckout"])
    with pytest.raises(SystemExit, match="pip install wandb"):
        pdp_main.make_loggers(args)
