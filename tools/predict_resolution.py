"""M6 (docs/ai/planning/2026-11-26-plan-M1-M6-kiem-prototype.md §2.2): predictions of a finished run at another
image size, without retraining.

Rebuilds the arguments the run saved in its run_info.json, links its task_final.pth into <out>/task_<t>/ and runs
pdp main with --predict_only 1 --eval_resolution <N> --pred_splits val. Score <out>/task_<t>/pred_val.npz with
tools.al_score like any run.

    python -m tools.predict_resolution --run-info <run>/task_5/run_info.json --checkpoint <run>/task_5/task_final.pth \\
        --resolution 800 --out <runs>/M6-r4-res800 [--ann-dir <data>/eval --img-root <data>/rpc]

N0/N1 (docs/ai/planning/2026-11-26-plan-N1-N5-hieu-chinh-task-va-loss.md): ``--resolution 0`` keeps the saved size,
``--dump-logits`` also writes logits_val.npz, ``--task-calib <json>`` predicts with a per-task logit calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autocheckout.datapaths import REPO


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-info", type=Path, required=True, help="run_info.json of the trained task")
    parser.add_argument("--checkpoint", type=Path, required=True, help="task_final.pth of that task")
    parser.add_argument("--resolution", type=int, required=True,
                        help="image size (shortest = longest edge); 0 = the size saved in the checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="output root; predictions in <out>/task_<t>/")
    parser.add_argument("--ann-dir", default="", help="folder of val_full.json and val_task_<t>.json "
                        "(default: the run's own)")
    parser.add_argument("--img-root", default="", help="image root of val_full.json (default: the run's own)")
    parser.add_argument("--dump-logits", action="store_true", help="N0: also write logits_val.npz")
    parser.add_argument("--task-calib", default="", help="N1: json of tools/task_calib_search.py")
    return parser.parse_args(argv)


def link_checkpoint(task_dir: Path, checkpoint: Path) -> None:
    target = task_dir / "task_final.pth"
    task_dir.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != checkpoint.resolve():
            raise SystemExit(f"{target} is another task_final.pth; choose another --out")
        return
    target.symlink_to(checkpoint.resolve())


def main(argv=None) -> Path:
    args = parse_args(argv)
    saved = json.loads(args.run_info.read_text())["sessions"][-1]["args"]
    task_id = int(saved["n_tasks"])
    task_dir = args.out / f"task_{task_id}"
    link_checkpoint(task_dir, args.checkpoint)

    sys.path.insert(0, str(REPO / "pdp"))
    import main as pdp_main

    parser = pdp_main.get_args_parser()
    defaults = {action.dest: action.default for action in parser._actions if action.dest != "help"}
    run = argparse.Namespace(**{**defaults, **saved})
    run.output_dir, run.start_task, run.n_tasks = str(args.out), task_id, task_id
    # predict only: nothing is trained, so the rows to retrain and the previous checkpoint of the run are not read
    run.predict_only, run.eval_resolution, run.pred_splits = 1, args.resolution, "val"
    run.dump_logits, run.task_calib = int(args.dump_logits), str(args.task_calib)
    run.prev_ckpt, run.train_rows = "", ""
    if args.ann_dir:
        run.task_ann_dir = run.pred_ann_dir = args.ann_dir
    if args.img_root:
        run.test_img_dir = args.img_root
    pdp_main.main(run)
    return task_dir


if __name__ == "__main__":
    main()
