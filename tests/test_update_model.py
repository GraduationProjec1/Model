"""autocheckout/update_model.py, mode improve (update-model plan, Task 3): gates, flags of EXP-FB-FT and a dry run."""

import json
from pathlib import Path

import pytest

from autocheckout import update_model as um
from autocheckout.predictions import load_predictions

REPO = Path(__file__).resolve().parent.parent


def comparison(fixed=30, broken=10, p=0.001, pairs=(100, 50), falls=0):
    return {'fixed': fixed, 'broken': broken, 'sign_test_p': p, 'falls': [{}] * falls,
            'pair_confusions_in_rows': {'base': pairs[0], 'new': pairs[1]}}


@pytest.mark.parametrize('name, cacc, cmp, ap, decision', [
    ('pilot', 0.44, comparison(), 0.96, 'go'),
    ('pilot', 0.4368, comparison(), 0.96, 'stop'),                      # pilot needs a strict gain
    ('e1', 0.452, comparison(), 0.96, 'stop'),                          # +0.0152 < +0.0165
    ('e1', 0.4534, comparison(), 0.96, 'go'),
    ('e1', 0.46, comparison(p=0.2), 0.96, 'stop'),                      # sign test
    ('e1', 0.46, comparison(pairs=(100, 80)), 0.96, 'stop'),            # pairs fell 20% only
    ('e1', 0.46, comparison(), 0.955, 'stop'),                          # AP50 of the other SKUs fell 1.4 points
    ('e1', 0.46, comparison(falls=1), 0.96, 'stop'),
    ('pilot', 0.44, comparison(falls=3), 0.96, 'go'),
])
def test_verdict_checks_every_condition_of_the_gate(name, cacc, cmp, ap, decision):
    assert um.verdict(um.GATES[name], 0.4368, cacc, cmp, 0.969, ap)[0] == decision


def test_round2_needs_no_sign_test_but_no_more_pair_confusions():
    gate = um.GATES['round2']
    assert um.verdict(gate, 0.46, 0.46, comparison(p=0.9, pairs=(50, 50)), 0.96, 0.958)[0] == 'go'
    assert um.verdict(gate, 0.46, 0.46, comparison(pairs=(50, 51)), 0.96, 0.96)[0] == 'stop'


def test_exp_fb_ft_sets_the_fixes_after_the_full_syn_flags(tmp_path):
    """F3-F6: the flags of EXP-FB-FT.sh come after those of EXP-FULL-SYN.sh, so argparse keeps them."""
    import subprocess

    env = dict(REPO=str(REPO), FSA_HF='/fsa', SYNTH_ROOT='/synth', PREV_CKPT='/r1/task_5/task_final.pth',
               TRAIN_ROWS='/rows.json', LR='2e-5', LR_OLD='5e-6', LR_DROP='3', PATH='/usr/bin:/bin')
    script = f'source {REPO}/configs/exp/native/EXP-FB-FT.sh; printf "%s\\0" "${{ARGS[@]}}"'
    out = subprocess.run(['bash', '-ec', script], env=env, capture_output=True, text=True,
                         check=True).stdout.split('\0')[:-1]
    last = {flag: out[i + 1] for i, flag in enumerate(out[:-1]) if flag.startswith('--')}
    assert last['--pseudo'] == 'none' and last['--init_new_prompts'] == '0'
    assert last['--new_params'] == 'class_embed,prompts.private'
    assert (last['--lr'], last['--lr_old'], last['--lr_drop']) == ('2e-5', '5e-6', '3')
    assert last['--prev_ckpt'] == '/r1/task_5/task_final.pth' and last['--train_rows'] == '/rows.json'
    assert last['--train_suffix'] == '_fix' and out.count('--prev_ckpt') == 1


def test_exp_fb_ft_refuses_a_run_without_model_or_rows():
    import subprocess

    result = subprocess.run(['bash', '-ec', f'source {REPO}/configs/exp/native/EXP-FB-FT.sh'],
                            env=dict(REPO=str(REPO), FSA_HF='/fsa', SYNTH_ROOT='/s', PATH='/usr/bin:/bin'),
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'PREV_CKPT' in result.stderr


@pytest.mark.slow
def test_improve_dry_run_runs_every_stage_once(tmp_path):
    import torch

    from autocheckout import full_syn
    base = dict(round=1, artifact_root=str(tmp_path), dry_run=True, batch_size=1, effective_batch=2,
                resolution=640, fsa_epochs=8, epochs=16, audit_images=4)
    seed = full_syn.prepare({**base, 'campaign_id': 'seed'})
    for stage in (full_syn.stage_fsa, full_syn.stage_task1, full_syn.stage_tasks):
        stage(seed)
    settings = dict(campaign_id='fix', round=1, artifact_root=str(tmp_path), dry_run=True,
                    seed_campaign=str(tmp_path / 'dry_run/seed'), min_per_class=1)
    plan = um.prepare(settings)
    split = um.stage_split(plan)
    assert split['val_a']['images'] + split['val_b']['images'] == 2
    baseline = um.stage_baseline(plan)
    assert baseline['rows']['pilot'] and 'cAcc_m2' in baseline

    seed5 = torch.load(plan['seed_task5'], map_location='cpu')['model']['class_embed.0.weight']
    pilot = um.stage_train(plan, 'pilot')
    run = Path(plan['runs']) / 'EXP-FB-FT-pilot'
    cli = json.loads((run / 'resolved_cli.json').read_text())['arguments']
    assert cli[cli.index('--pseudo') + 1] == 'none' and '--train_rows' in cli
    trained = torch.load(run / 'task_5/task_final.pth', map_location='cpu')['model']['class_embed.0.weight']
    frozen = [r for r in range(6) if r not in pilot['rows']]
    assert torch.equal(trained[frozen], seed5[frozen])               # rows outside train_rows, bit for bit
    assert not torch.equal(trained[pilot['rows']], seed5[pilot['rows']])
    assert pilot['gate']['decision'] in {'go', 'stop'} and pilot['gate']['enforced'] is False
    assert load_predictions(run / 'task_5/pred_val.npz').image_id.size  # predictions on all of val
    assert (Path(plan['root']) / 'scored/pilot/sku_regression.json').is_file()

    mtime = (run / 'task_5/task_final.pth').stat().st_mtime
    assert um.stage_train(plan, 'pilot')['finished_at'] == pilot['finished_at']
    assert (run / 'task_5/task_final.pth').stat().st_mtime == mtime
    e1 = um.stage_train(plan, 'e1')
    round2 = um.stage_round2(plan)
    assert round2['reference'] == 'e1' and e1['stage'] == 'e1'
    if e1['gate']['decision'] == 'go':  # e2 only after an e1 stop
        with pytest.raises(RuntimeError, match='e2'):
            um.stage_train(plan, 'e2')
    if any(r['gate']['decision'] == 'go' for r in (e1, round2)):
        with pytest.raises(RuntimeError, match='confirm'):
            um.stage_final_test(plan)
        final = um.stage_final_test(plan, confirm=True)
        assert final['test2019']['counting']
    assert um.prepare(settings) == plan
