"""tools/al_score.py: Val-B scoring with Val-A-calib thresholds, generalised to any stage and task config
(chia 12+13 SKU, docs/design/2026-11-09-chia-12-13-sku, plan 3.5)."""

import os
from pathlib import Path

import pytest

from autocheckout.io import save_json
from autocheckout.predictions import save_predictions
from autocheckout.taskcfg import TaskConfig
from tests.coco_helpers import make_ann, make_gt, make_image, make_preds, three_task_config
from tools import al_score
from tools.eval_count import CALIBRATION_GRID

BOX = [0, 0, 10, 10]
XYXY = [0, 0, 10, 10]


def split_config() -> TaskConfig:
    """three_task_config with tasks 2 and 3 split in two: [2, 1, 1, 1, 1], same labels and RPC ids."""
    data = three_task_config().to_dict()
    classes = [c for t in data["tasks"] for c in t["classes"]]
    sizes, tasks, offset = (2, 1, 1, 1, 1), [], 0
    for task_id, size in enumerate(sizes, 1):
        tasks.append({"task_id": task_id, "offset": offset, "reserved": False, "classes": classes[offset:offset + size]})
        offset += size
    return TaskConfig.from_dict({"name": "toy-split", "seed": 0, "tasks": tasks})


def test_groups_by_task_or_by_the_r1_tasks():
    cfg, ref = split_config(), three_task_config()
    assert al_score.group_labels(cfg, 3) == ([[0, 1], [2], [3]], ["task_1", "task_2", "task_3"])
    assert al_score.group_labels(cfg, 3, "r1", ref) == ([[0, 1], [2, 3]], ["r1_task_1", "r1_task_2"])
    assert al_score.group_labels(cfg, 2, "r1", ref) == ([[0, 1], [2]], ["r1_task_1", "r1_task_2"])
    assert al_score.group_labels(cfg, 5, "r1", ref)[0] == [[0, 1], [2, 3], [4, 5]]
    assert al_score.group_labels(ref, 3, "r1", ref) == (al_score.group_labels(ref, 3)[0],
                                                        ["r1_task_1", "r1_task_2", "r1_task_3"])
    with pytest.raises(ValueError, match="r1"):
        al_score.group_labels(cfg, 3, "r1")
    other = TaskConfig.from_dict({**ref.to_dict(), "name": "other", "tasks": ref.to_dict()["tasks"][:2]})
    with pytest.raises(ValueError, match="missing"):
        al_score.group_labels(cfg, 5, "r1", other)


def test_threshold_grid_keeps_the_default_and_can_start_lower():
    assert al_score.threshold_grid() is CALIBRATION_GRID
    low = al_score.threshold_grid(0.02)
    assert low[0] == 0.02 and low[-1] == 0.9 and len(low) == 89 and set(CALIBRATION_GRID) <= set(low)
    with pytest.raises(ValueError):
        al_score.threshold_grid(0.2)


def fixture():
    """Image 1: one label-0 object, found. Image 2: label-0 found + a label-4 object (not learned at stage 3) that
    the model calls label 2. Image 3: only a label-5 object, called label 3."""
    images = [make_image(i) for i in (1, 2, 3)]
    anns = [make_ann(1, 1, 0, BOX), make_ann(2, 2, 0, BOX), make_ann(3, 2, 4, [50, 50, 10, 10]),
            make_ann(4, 3, 5, BOX)]
    gt = make_gt(images, anns, num_labels=6)
    preds = make_preds([(1, 0, 0, 0.9, XYXY), (2, 0, 0, 0.9, XYXY), (2, 1, 2, 0.8, [50, 50, 60, 60]),
                        (3, 0, 3, 0.7, XYXY)])
    return gt, preds


def test_m2_drops_unlearned_objects_and_images_without_learned_objects():
    gt, preds = fixture()
    cfg = split_config()
    groups = al_score.group_labels(cfg, 3)[0]
    plain = al_score.count_on(gt, cfg, preds, [1, 2, 3], [0.5, 0.5, 0.5], groups=groups)
    assert plain["target_images"] == 3 and plain["right_images"] == [1]
    m2 = al_score.count_on(gt, cfg, preds, [1, 2, 3], [0.5, 0.5, 0.5], groups=groups, unlearned_iou=0.5)
    assert m2["target_images"] == 2 and m2["right_images"] == [1, 2] and m2["cAcc_m2"] == 1.0


def test_score_chooses_on_calib_and_reports_the_grid_floor():
    gt, preds = fixture()
    cfg = split_config()
    default = al_score.score(gt, cfg, preds, [1, 2], [1, 2, 3], groups=al_score.group_labels(cfg, 3))
    assert set(default["thresholds"]) == {"task_1", "task_2", "task_3"} and "at_floor" not in default
    low = al_score.score(gt, cfg, preds, [1, 2], [1, 2, 3], groups=al_score.group_labels(cfg, 3, "r1",
                         three_task_config()), grid=al_score.threshold_grid(0.02), unlearned_iou=0.5)
    assert set(low["thresholds"]) == {"r1_task_1", "r1_task_2"} and low["grid"] == [0.02, 0.9]
    assert low["target_images"] == 2 and low["cAcc_m2"] == 1.0 and isinstance(low["at_floor"], list)


def data_folder(root: Path, cfg: TaskConfig, gt: dict) -> Path:
    save_json(root / "eval/val_full.json", gt)
    cfg.save(root / "eval/task_config.json")
    save_json(root / "relearn/active/splits/val_a_split.json", {"calib": [1, 2], "pool": [3]})
    save_json(root / "relearn/meta/val_b.json", {"images": [make_image(i) for i in (1, 2, 3)]})
    return root


def test_cli_default_output_is_unchanged_and_new_options_write_where_asked(tmp_path):
    gt, preds = fixture()
    data = data_folder(tmp_path / "data", three_task_config(), gt)
    save_predictions(tmp_path / "pred_val.npz", preds)
    result = al_score.main(["--data", str(data), "--pred", str(tmp_path / "pred_val.npz"), "--name", "m",
                            "--stage", "3"])
    assert set(result) == {"name", "pred", "pred_md5", "thresholds", "nms_iou", "calib_images", "calib_cAcc",
                           "target_images", "cAcc_m2", "ACD", "mCCD", "mCIoU", "right_images", "stage",
                           "task_config", "groups", "target"}
    assert (data / "relearn/active/scored/m.json").is_file()

    save_json(tmp_path / "split/eval_12x/val_full.json", gt)
    split_config().save(tmp_path / "split/eval_12x/task_config.json")
    out = tmp_path / "out/r1.json"
    result = al_score.main(["--data", str(data), "--pred", str(tmp_path / "pred_val.npz"), "--name", "s",
                            "--stage", "3", "--groups", "r1", "--eval-dir", str(tmp_path / "split/eval_12x"),
                            "--target", "pool", "--grid-min", "0.02", "--out", str(out)])
    assert out.is_file() and result["groups"] == "r1" and result["target"] == "pool"
    assert result["unlearned_iou"] == 0.5 and set(result["thresholds"]) == {"r1_task_1", "r1_task_2"}
    last = al_score.main(["--data", str(data), "--pred", str(tmp_path / "pred_val.npz"), "--name", "t",
                          "--stage", "5", "--eval-dir", str(tmp_path / "split/eval_12x"), "--out", str(out)])
    assert "unlearned_iou" not in last
    with pytest.raises(ValueError, match="stage"):
        al_score.main(["--data", str(data), "--pred", str(tmp_path / "pred_val.npz"), "--name", "x", "--stage", "6",
                       "--eval-dir", str(tmp_path / "split/eval_12x"), "--out", str(out)])


DATA = Path(os.environ.get("AC_DATA") or Path(__file__).resolve().parents[2] / "autocheckout-data")
R1_TASK5 = DATA / "models/full-syn-r1/runs/EXP-FULL-SYN/task_5/pred_val.npz"


@pytest.mark.skipif(not R1_TASK5.is_file() or not (DATA / "relearn/meta/val_b.json").is_file(),
                    reason="autocheckout-data with r1 and relearn/ not on this machine")
def test_r1_baseline_is_unchanged(tmp_path):
    """Plan 3.5 / Task 6: the default call still gives r1's 42.32% on Val-B with its five thresholds."""
    result = al_score.main(["--pred", str(R1_TASK5), "--name", "baseline_r1_check", "--out",
                            str(tmp_path / "r1.json")])
    assert round(result["cAcc_m2"], 4) == 0.4232 and result["target_images"] == 2897
    assert list(result["thresholds"].values()) == [0.13, 0.17, 0.17, 0.17, 0.23]
