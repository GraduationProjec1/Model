"""R4/V5: the benchmark runs a full PDP training step (teacher + PPG + student) and two-pass inference."""

import pytest
import torch

import benchmark
import main as pdp_main
from pdp_helpers import IMAGE_SIZE, make_toy_dataset, small_processor, use_tiny_detr, write_task_config

pytestmark = pytest.mark.slow


def bench_args(tmp_path, *extra):
    config = write_task_config(tmp_path / "tasks.json", sizes=(100, 25), reserved=0)
    args = benchmark.parse_args([
        "--task_config", str(config), "--n_classes", "126", "--repo_name", "", "--n_tasks", "2",
        "--use_prompts", "1", "--local_query", "1", "--num_prompts", "4", "--prompt_len", "2",
        "--freeze", "backbone,encoder,decoder", "--new_params", "class_embed,prompts", "--lr", "1e-4",
        "--batch_size", "2", "--bench_steps", "3", "--bench_train_size", str(IMAGE_SIZE),
        "--bench_sizes", str(IMAGE_SIZE), "--accelerator", "cpu", *extra])
    pdp_main.setup_task_info(args)
    args.log_file = open(tmp_path / "log.txt", "w")  # noqa: SIM115
    return args


def test_train_and_infer_modes(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    args = bench_args(tmp_path)
    train = benchmark.bench_train(args, torch.device("cpu"))
    assert train["seconds_per_image"] > 0 and train["batch_size"] == 2
    infer = benchmark.bench_infer(args, torch.device("cpu"))
    assert infer[f"ms_per_image_{IMAGE_SIZE}"] > 0
    storage = benchmark.storage_per_class(args)
    assert storage["prototype_memory_bytes_per_class"] == 100 * 1024  # 100 KiB, as in the plan (V5)


# --- P1/P2 (docs/design/2026-11-22-toi-uu-2-luot, Task 1): stage timings, data loading, image ids -------------
def test_random_batch_has_unique_image_ids():
    generator = torch.Generator().manual_seed(0)
    cpu = torch.device("cpu")
    first = benchmark.random_batch(2, IMAGE_SIZE, 3, cpu, generator, first_id=1)
    second = benchmark.random_batch(2, IMAGE_SIZE, 3, cpu, generator, first_id=3)
    ids = [target["image_id"].item() for target in first["labels"] + second["labels"]]
    assert ids == [1, 2, 3, 4]


def test_stages_mode_reports_backbone_encoder_decoder(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    args = bench_args(tmp_path)
    result = benchmark.bench_stages(args, torch.device("cpu"))
    assert result["mode"] == "stages"
    parts = [result[key] for key in ("backbone_ms", "encoder_ms", "decoder_ms")]
    assert all(part > 0 for part in parts)
    # the three stages are timed inside the same first pass, so they cannot exceed it
    assert sum(parts) <= result["pass1_ms"]
    assert result["pass2_ms"] > 0
    # each reported number is rounded to 3 decimals (off by up to 0.0005), so this sum is off by up to 0.0015
    assert result["removed_ms_infer"] == pytest.approx(result["backbone_ms"] + result["encoder_ms"], abs=2e-3)
    assert 0 < result["predicted_infer_saving"] < 1
    for key in ("removed_ms_train_epoch1", "removed_ms_train_cached", "removed_ms_train_no_teacher"):
        assert result[key] > 0


def test_train_mode_repeats_the_same_batches_each_epoch(tmp_path, monkeypatch):
    use_tiny_detr(monkeypatch)
    args = bench_args(tmp_path, "--bench_epochs", "2")
    result = benchmark.bench_train(args, torch.device("cpu"))
    assert result["steps"] == 2 * args.bench_steps
    assert result["seconds_per_image"] > 0


def test_data_mode_reports_loader_time(tmp_path):
    data = make_toy_dataset(tmp_path / "data")
    args = bench_args(tmp_path, "--train_img_dir", str(data["images"]), "--task_ann_dir", str(data["tasks"]),
                      "--num_workers", "0", "--bench_steps", "2")
    result = benchmark.bench_data(args, processor=small_processor())
    assert result["mode"] == "data"
    assert result["batches"] == 2
    assert result["seconds_per_image"] > 0
