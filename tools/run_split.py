"""Chia 12+13 SKU (docs/design/2026-11-09-chia-12-13-sku/01-plan.md): task 1 of FULL-SYN r1, repartitioned to the
100 + 8 x 12/13 task config, then tasks 2..9 on the synthetic release synth_v2_12x, scored like r1. Needs a GPU except
--dry-run; rerun the same command to resume (finished stages are kept, an interrupted task continues).

    python -m tools.run_split --dry-run                 # CPU, a few minutes: the whole chain on the toy fixture
    python -m tools.run_split --smoke                   # GPU, ~15 min: data check + task 2 for 200 steps, stop, resume
    python -m tools.run_split                           # GPU, ~25-30 h on a 4090
    python -m tools.run_split --accept-mid-gate         # after the team chose to go on past the 125-SKU gate

Gates (plan 5): task 1 must predict exactly as in r1 (full_syn.stage_task1_predict); teacher precision >= 0.97 before
tasks 2-5 (task 2 also >= 0.5, as r1); at 125 SKUs (after task 3) cAcc-M2 on the Val-A pool, r1 task groups, may be at
most MID_DROP below r1 at task 2, otherwise the run stops until the team decides (--accept-mid-gate). At the end, the
four milestones 125/150/175/200 SKUs are scored on Val-B next to r1 (tools/al_score.py), into <campaign>/split12/.
The data paths come from the data folder (autocheckout/datapaths.py: ../autocheckout-data or $AC_DATA).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autocheckout import full_syn
from autocheckout.datapaths import Layout
from autocheckout.io import load_json, save_json

MID_STAGE, MID_R1_STAGE, MID_DROP = 3, 2, 0.03            # plan 5: 125 SKUs, Val-A pool, 3.0 points
PRECISION_FLOOR = {task: 0.97 for task in (2, 3, 4, 5)}   # plan 5 (R13): from task 6 on, only logged
MILESTONES = {3: 2, 5: 3, 7: 4, 9: 5}                     # 12x stage -> r1 stage with the same SKUs learned
GRIDS = {'grid10': None, 'grid02': 0.02}                  # R1: threshold grid from 0.10 (as r1) and from 0.02


def real_settings(path: str, data: Layout) -> dict:
    from autocheckout.training_workflow import resolve

    settings = {k: v for k, v in load_json(resolve(path)).items() if not k.startswith('_')}
    if settings.get('dry_run'):
        raise ValueError(f'{path} is a dry-run config; use --dry-run')
    settings.update(release_root=str(data.synth_12x), eval_root=str(data.eval_12x), raw_root=str(data.rpc),
                    pretrained_root=str(data.pretrained), seed_campaign=str(data.r1),
                    seed_task_config=str(data.eval / 'task_config.json'))
    return settings


def _score(data: Layout, name: str, pred: Path, stage: int, out: Path, *, split: bool, groups: str = 'r1',
           target: str = 'valb', grid_min: float | None = None) -> dict:
    """tools.al_score of one prediction file: a 12x stage (split=True, eval_12x labels) or an r1 stage."""
    from tools import al_score

    argv = ['--data', str(data.root), '--pred', str(pred), '--name', name, '--stage', str(stage),
            '--target', target, '--out', str(out / f'{name}.json')]
    if split:
        argv += ['--groups', groups, '--eval-dir', str(data.eval_12x)]
    if grid_min is not None:
        argv += ['--grid-min', str(grid_min)]
    result = al_score.main(argv)
    return {k: v for k, v in result.items() if k != 'right_images'}


def mid_gate(plan: dict, data: Layout, accept: bool) -> dict:
    """Plan 5: after task 3 (125 SKUs), cAcc-M2 on the Val-A pool with the r1 task groups, against r1 at task 2. A
    passed (or accepted) gate is kept, so a resumed run does not stop at it again."""
    done = Path(plan['root']) / 'stages/mid_gate_125.json'
    if done.is_file() and load_json(done)['decision'] == 'go':
        return load_json(done)
    run = Path(plan['runs']) / plan['pdp_experiment']
    out = Path(plan['root']) / 'split12'
    mine = _score(data, f'12x_t{MID_STAGE}_pool', run / f'task_{MID_STAGE}/pred_val.npz', MID_STAGE, out,
                  split=True, target='pool')
    ref = _score(data, f'r1_t{MID_R1_STAGE}_pool', data.r1 / f'runs/{full_syn.PDP}/task_{MID_R1_STAGE}/pred_val.npz',
                 MID_R1_STAGE, out, split=False, target='pool')
    drop = round(ref['cAcc_m2'] - mine['cAcc_m2'], 6)
    stop = drop > MID_DROP
    gate = {'name': '125 SKUs on the Val-A pool', 'decision': 'stop' if stop and not accept else 'go',
            'drop': drop, 'max_drop': MID_DROP, 'cAcc_m2': mine['cAcc_m2'], 'r1_cAcc_m2': ref['cAcc_m2'],
            'images': mine['target_images'], 'r1_images': ref['target_images'], 'accepted_by_flag': bool(stop and accept),
            'reason': f"12x {mine['cAcc_m2']:.4f} vs r1 {ref['cAcc_m2']:.4f} ({-drop:+.4f}, at most -{MID_DROP})"}
    save_json(done, gate, indent=2)
    print(f"GATE {gate['name']}: {gate['decision'].upper()} - {gate['reason']}")
    if gate['decision'] == 'stop':
        raise full_syn.GateStop(f"{gate['name']}: {gate['reason']}; the team decides, then rerun with "
                                '--accept-mid-gate to go on')
    return gate


def milestones(plan: dict, data: Layout) -> dict:
    """Val-B scores of the 12x stages 3/5/7/9 and of r1 stages 2-5 (same SKUs learned), both threshold grids; the
    12x ones with the r1 task groups (the decision) and with its own task groups (reported)."""
    run = Path(plan['runs']) / plan['pdp_experiment']
    out = Path(plan['root']) / 'split12'
    rows = []
    for stage, r1_stage in MILESTONES.items():
        pred = run / f'task_{stage}/pred_val.npz'
        if not pred.is_file():
            continue
        for grid, minimum in GRIDS.items():
            r1 = _score(data, f'r1_t{r1_stage}_valb_{grid}', data.r1 / f'runs/{full_syn.PDP}/task_{r1_stage}/pred_val.npz',
                        r1_stage, out, split=False, grid_min=minimum)
            row = {'stage': stage, 'skus': 100 + 25 * r1_stage - 25, 'r1_stage': r1_stage, 'grid': grid,
                   'r1': r1['cAcc_m2'], 'r1_images': r1['target_images'], 'r1_at_floor': r1.get('at_floor', [])}
            for groups in ('r1', 'task'):
                mine = _score(data, f'12x_t{stage}_valb_{groups}_{grid}', pred, stage, out, split=True,
                              groups=groups, grid_min=minimum)
                row[f'12x_{groups}'] = mine['cAcc_m2']
                row[f'12x_{groups}_images'] = mine['target_images']
                row[f'12x_{groups}_at_floor'] = mine.get('at_floor', [])
            rows.append(row)
    lines = ['# Chia 12+13 SKU: cAcc-M2 Val-B (ngưỡng chọn trên Val-A calib)', '',
             'Quyết định theo `01-plan.md` §5: mốc 200 SKU, nhóm ngưỡng r1, lưới 0,10 (sign test làm riêng).', '',
             '| SKU | 12x stage | lưới | r1 | 12x (nhóm r1) | 12x (nhóm task) | ảnh r1 / 12x | ngưỡng chạm sàn |',
             '|---|---|---|---|---|---|---|---|']
    for r in rows:
        floor = ', '.join(r['r1_at_floor'] + r['12x_r1_at_floor'] + r['12x_task_at_floor']) or '-'
        lines.append(f"| {r['skus']} | {r['stage']} | {r['grid']} | {100 * r['r1']:.2f}% | {100 * r['12x_r1']:.2f}% | "
                     f"{100 * r['12x_task']:.2f}% | {r['r1_images']} / {r['12x_r1_images']} | {floor} |")
    (out / 'MILESTONES.md').write_text('\n'.join(lines) + '\n')
    save_json(out / 'milestones.json', rows, indent=2)
    return {'rows': rows, 'table': str(out / 'MILESTONES.md')}


def dry_run(settings_path: str) -> dict:
    """The whole chain on the CPU fixture: a seed campaign with tasks [2, 2, 2] gives its task 1 to a campaign with
    tasks [2, 1, 1, 1, 1]; seed -> repartition -> split smoke -> task 1 predicted again -> tasks 2..5. Scoring on
    Val-A/Val-B needs the data folder and is not part of the dry run."""
    from autocheckout.training_workflow import resolve

    settings = load_json(resolve(settings_path))
    base = dict(round=1, artifact_root=settings.get('artifact_root', 'data/training'), dry_run=True, batch_size=1,
                effective_batch=2, resolution=640, fsa_epochs=8, epochs=16, audit_images=4)
    seed = full_syn.prepare({**base, 'campaign_id': 'split12-dry-seed', 'dry_task_sizes': [2, 2, 2]})
    full_syn.stage_fsa(seed)
    full_syn.stage_task1(seed)
    plan = full_syn.prepare({**base, 'campaign_id': 'split12-dry', 'seed_campaign': seed['root'],
                             'seed_task_config': seed['task_config']})
    full_syn.stage_seed(plan)
    full_syn.stage_smoke_split(plan, steps=1)
    task1 = full_syn.stage_task1_predict(plan)
    if task1['gate']['decision'] != 'go':
        raise SystemExit(f"dry run: task 1 changed after the repartition ({task1['gate']['reason']})")
    full_syn.stage_tasks(plan, last=MID_STAGE, min_precision=PRECISION_FLOOR)
    tasks = full_syn.stage_tasks(plan, min_precision=PRECISION_FLOOR)
    return {'campaign': plan['root'], 'task1': task1['gate']['reason'], 'tasks': sorted(tasks['tasks']),
            'summary': str(Path(plan['root']) / 'SUMMARY.md')}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--settings', default='configs/training/full_syn_12x.json')
    ap.add_argument('--data', type=Path, help='the data folder (default: ../autocheckout-data or $AC_DATA)')
    ap.add_argument('--dry-run', action='store_true', help='CPU: the whole chain on the toy fixture')
    ap.add_argument('--smoke', action='store_true', help='GPU: data check and task 2 for --smoke-steps, stop, resume')
    ap.add_argument('--smoke-steps', type=int, default=200)
    ap.add_argument('--last', type=int, help='last task to train (default: every task)')
    ap.add_argument('--accept-mid-gate', action='store_true', help='go on past a stop of the 125-SKU gate')
    args = ap.parse_args(argv)
    if args.dry_run:
        return dry_run(args.settings)
    data = Layout.at(args.data)
    plan = full_syn.prepare(real_settings(args.settings, data))
    full_syn.stage_seed(plan)
    if args.smoke:
        return full_syn.stage_smoke_split(plan, steps=args.smoke_steps)
    full_syn.stage_task1_predict(plan)
    last = args.last or full_syn._n_tasks(plan)
    full_syn.stage_tasks(plan, last=min(MID_STAGE, last), min_precision=PRECISION_FLOOR)
    result = {'campaign': plan['root']}
    if last > MID_STAGE:
        result['mid_gate'] = mid_gate(plan, data, args.accept_mid_gate)
        full_syn.stage_tasks(plan, last=last, min_precision=PRECISION_FLOOR)
    result['milestones'] = milestones(plan, data)
    result['summary'] = str(Path(plan['root']) / 'SUMMARY.md')
    return result


if __name__ == '__main__':
    print(json.dumps(main(), indent=1, default=str))
