"""Round 3 (docs/design/2026-11-07-vong-3, Task 2): the counting policy of a run as the counter uses it.

``label_thresholds``: the group thresholds of a ``tools/al_score.py`` result (chosen on Val-A-calib) spread over the
200 labels; ``review_score_by_label``: own SKU thresholds of a ``tools/count_policy_search.py`` policy, used by the
counter only to ASK the cashier (boxes are never dropped by them). Read by checkout-system ``model_store pack-task
--count-policy``.

    python -m tools.export_count_policy --scored <data>/relearn/active/scored/al_margin_k_B500_r2.json \\
        --task-config configs/relearn/task_config_relearn.json \\
        --review-from results/count_policy_search/al_margin_k_B500_r2_pool.json --review-policy 'nms=0.45+sku' \\
        --out results/count_policy_al_margin_k_B500_r2.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from autocheckout.counting import label_thresholds
from autocheckout.io import load_json, save_json
from autocheckout.taskcfg import TaskConfig

STAGE = 5


def export(scored: Path, task_config: str | Path, review_from: Path | None = None,
           review_policy: str | None = None) -> dict:
    d = load_json(scored)
    cfg = TaskConfig.load(task_config)
    groups = [list(cfg.task(t).labels) for t in range(1, STAGE + 1)]
    vector = label_thresholds(groups, [d['thresholds'][f'task_{t}'] for t in range(1, STAGE + 1)])
    out = {'label_thresholds': [float(x) for x in vector], 'nms_iou': float(d['nms_iou']),
           'source': f"{Path(scored).name}: group thresholds chosen on Val-A-calib"}
    if review_from is not None:
        policy = load_json(review_from)['policies'][review_policy]
        if policy['nms'] is None or abs(float(policy['nms']) - out['nms_iou']) > 1e-9:
            raise ValueError(f'{review_policy}: NMS {policy["nms"]} differs from the counting NMS {out["nms_iou"]}')
        out['review_score_by_label'] = {str(k): float(v) for k, v in policy['sku_thresholds'].items()}
        out['source'] += f"; review per SKU from {Path(review_from).name} [{review_policy}]"
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--scored', type=Path, required=True)
    ap.add_argument('--task-config', type=Path, required=True)
    ap.add_argument('--review-from', type=Path)
    ap.add_argument('--review-policy')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    out = export(args.scored, args.task_config, args.review_from, args.review_policy)
    save_json(args.out, out, indent=1)
    return out


if __name__ == '__main__':
    main()
