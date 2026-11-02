"""B2: the lower bound EXP-FULL-SYN-NAIVE (no private pool, no pseudo-labels), tasks 2..last from task 1 of FULL-SYN
r1, then its forgetting next to r1's (tools/forgetting_report.py, val). Needs a GPU; rerun the same command to resume.

    python -m tools.run_naive --settings configs/training/full_syn_naive.json --last 5    # ~20 h on a 4090
    python -m tools.run_naive --settings configs/training/full_syn_naive.json --last 3    # cheaper, ~10 h

The data paths come from the data folder (autocheckout/datapaths.py: ../autocheckout-data or $AC_DATA).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autocheckout import full_syn
from autocheckout.datapaths import Layout
from autocheckout.io import load_json
from tools import forgetting_report


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--settings', required=True)
    ap.add_argument('--last', type=int, default=full_syn.N_TASKS, choices=range(2, full_syn.N_TASKS + 1))
    ap.add_argument('--out', type=Path, default=Path('docs/al/ket-qua'))
    ap.add_argument('--data', type=Path, help='the data folder (default: ../autocheckout-data or $AC_DATA)')
    args = ap.parse_args(argv)
    from autocheckout.training_workflow import resolve

    data = Layout.at(args.data)
    settings = {k: v for k, v in load_json(resolve(args.settings)).items() if not k.startswith('_')}
    settings.update(release_root=str(data.synth), eval_root=str(data.eval), raw_root=str(data.rpc),
                    pretrained_root=str(data.pretrained), seed_campaign=str(data.r1))
    if settings.get('pdp_experiment') != full_syn.NAIVE:
        raise ValueError(f'the settings must run {full_syn.NAIVE}')
    plan = full_syn.prepare(settings)
    full_syn.stage_seed(plan)
    full_syn.stage_tasks(plan, last=args.last)
    naive = Path(plan['runs']) / full_syn.NAIVE / 'metrics_cl_val.json'
    seed = Path(plan['seed_root']) / 'runs' / full_syn.PDP / 'metrics_cl_val.json'
    return forgetting_report.main(['--out', str(args.out), '--run', f'r1_val={seed}', '--run', f'naive_val={naive}'])


if __name__ == '__main__':
    r = main()
    print(json.dumps({k: {m: v[m] for m in ('ACC_final', 'forgetting', 'BWT')} for k, v in r.items()}, indent=1))
