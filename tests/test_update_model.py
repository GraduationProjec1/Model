"""autocheckout/update_model.py, mode improve (update-model plan, Task 3): gates, flags of EXP-FB-FT and a dry run."""

import json
from pathlib import Path

import pytest

from autocheckout import update_model as um
from autocheckout.predictions import load_predictions

REPO = Path(__file__).resolve().parent.parent


def comparison(fixed=30, broken=10, p=0.001, pairs=(100, 50), falls=0, largest=(60, 30)):
    top = None if largest is None else {'pair': [0, 1], 'base': largest[0], 'new': largest[1]}
    return {'fixed': fixed, 'broken': broken, 'sign_test_p': p, 'falls': [{}] * falls,
            'pair_confusions_in_rows': {'base': pairs[0], 'new': pairs[1]}, 'largest_pair_in_rows': top}


@pytest.mark.parametrize('name, cacc, cmp, ap, decision', [
    ('pilot', 0.44, comparison(), 0.96, 'go'),
    ('pilot', 0.4368, comparison(), 0.96, 'stop'),
    ('e1', 0.452, comparison(), 0.96, 'stop'),
    ('e1', 0.4534, comparison(), 0.96, 'go'),
    ('e1', 0.46, comparison(p=0.2), 0.96, 'stop'),
    ('e1', 0.46, comparison(pairs=(100, 80)), 0.96, 'stop'),
    ('e1', 0.46, comparison(pairs=(100, 75)), 0.96, 'go'),
    ('e1', 0.46, comparison(), 0.955, 'stop'),
    ('e1', 0.46, comparison(falls=11), 0.96, 'stop'),
    ('e1', 0.46, comparison(falls=10), 0.96, 'go'),
    ('pilot', 0.44, comparison(falls=9), 0.96, 'go'),
    ('pilot', 0.44, comparison(falls=17), 0.96, 'stop'),
])
def test_verdict_checks_every_condition_of_the_gate(name, cacc, cmp, ap, decision):
    assert um.verdict(um.GATES[name], 0.4368, cacc, cmp, 0.969, ap)[0] == decision


def test_round2_needs_no_sign_test_and_a_10_percent_pair_fall():
    """13/11/2026: round2 pair_drop 0 -> 0.10 (a later round starts from fewer, harder confusions)."""
    gate = um.GATES['round2']
    assert um.verdict(gate, 0.46, 0.46, comparison(p=0.9, pairs=(50, 45)), 0.96, 0.958)[0] == 'go'
    assert um.verdict(gate, 0.46, 0.46, comparison(pairs=(50, 46)), 0.96, 0.96)[0] == 'stop'
    assert um.verdict(gate, 0.46, 0.46, comparison(p=0.9, pairs=(50, 45), falls=11), 0.96, 0.96)[0] == 'stop'
    assert um.verdict(gate, 0.5406, 0.5678, comparison(p=1e-6, pairs=(201, 168), largest=(103, 99)),
                      0.96, 0.96)[0] == 'go'


def test_later_round_gates_follow_the_pilot_gate_with_a_10_percent_pair_fall():
    """13/11/2026 (doc 08 §2.4): the first round keeps 0.25; rounds that start from an improved model need 0.10."""
    later = {**um.GATES['pilot'], 'pair_drop': 0.10}
    for kind in ('r3_base', 'r3_strong', 'r3_prompt', 'r3_freeze', 'r4', 'r3_strong_s1'):
        assert um.GATES[kind] == later
    assert (um.GATES['pilot']['pair_drop'], um.GATES['pilot']['max_falls']) == (0.25, 10)
    assert all(um.GATES[kind]['pair_drop'] == 0.25 for kind in ('pilot', 'e1', 'e2'))
    assert all(gate['top_pair_rise'] == 0.05 for gate in um.GATES.values())


@pytest.mark.parametrize('kind, pairs, decision', [
    ('pilot', (100, 80), 'stop'),
    ('r3_strong', (100, 85), 'go'),
    ('r3_strong', (100, 92), 'stop'),
    ('r4', (313, 281), 'go'),
])
def test_pair_drop_by_kind_of_run(kind, pairs, decision):
    assert um.verdict(um.GATES[kind], 0.5582, 0.60, comparison(pairs=pairs), 0.96, 0.96)[0] == decision


@pytest.mark.parametrize('largest, decision', [
    ((99, 108), 'stop'),
    ((99, 104), 'stop'),
    ((99, 103), 'go'),
    ((99, 100), 'go'),
    ((99, 92), 'go'),
])
def test_largest_pair_may_not_rise_more_than_5_percent(largest, decision):
    """13/11/2026: r3_train cut its pair confusions 9.7 % but made the largest pair (162<->186 milk) worse."""
    cmp = comparison(pairs=(443, 313), largest=largest)
    decision_, reason = um.verdict(um.GATES['r3_strong'], 0.5582, 0.60, cmp, 0.96, 0.96)
    assert decision_ == decision
    assert ('largest pair' in reason) == (decision == 'stop')


def test_largest_pair_is_reported_as_not_measurable_without_a_pair_in_the_rows():
    decision, reason = um.verdict(um.GATES['pilot'], 0.4368, 0.46, comparison(largest=None), 0.96, 0.96)
    assert decision == 'go' and 'top pair n/a' in reason


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
    assert torch.equal(trained[frozen], seed5[frozen])
    assert not torch.equal(trained[pilot['rows']], seed5[pilot['rows']])
    assert pilot['gate']['decision'] in {'go', 'stop'} and pilot['gate']['enforced'] is False
    assert load_predictions(run / 'task_5/pred_val.npz').image_id.size
    assert (Path(plan['root']) / 'scored/pilot/sku_regression.json').is_file()

    mtime = (run / 'task_5/task_final.pth').stat().st_mtime
    assert um.stage_train(plan, 'pilot')['finished_at'] == pilot['finished_at']
    assert (run / 'task_5/task_final.pth').stat().st_mtime == mtime
    e1 = um.stage_train(plan, 'e1')
    round2 = um.stage_round2(plan)
    assert round2['reference'] == 'e1' and e1['stage'] == 'e1'
    if e1['gate']['decision'] == 'go':
        with pytest.raises(RuntimeError, match='e2'):
            um.stage_train(plan, 'e2')
    if any(r['gate']['decision'] == 'go' for r in (e1, round2)):
        with pytest.raises(RuntimeError, match='confirm'):
            um.stage_final_test(plan)
        final = um.stage_final_test(plan, confirm=True)
        assert final['test2019']['counting']
    assert um.prepare(settings) == plan


@pytest.mark.parametrize('name, kind', [
    ('al_margin_k_B500', 'pilot'), ('al_margin_k_B500_r2', 'round2'),
    ('al_margin_k_B500_r3_rows', 'r3_base'), ('al_margin_k_B500_r3_train', 'r3_strong'),
    ('al_margin_k_B500_r3_real', 'r3_strong'), ('al_margin_k_B500_r3_all', 'r3_strong'),
    ('al_margin_k_B500_r3_prompt', 'r3_prompt'), ('al_margin_k_B500_r3_freeze', 'r3_freeze'),
])
def test_round3_kind_and_recipes(name, kind):
    assert um._kind(name) == kind and kind in um.VARIANTS and kind in um.GATES


def test_round3_recipes():
    assert um.VARIANTS['r3_base'] == um.VARIANTS['round2']
    strong = um.VARIANTS['r3_strong']
    assert (strong['epochs'], strong['lr'], strong['lr_old'], strong['augment']) == (6, 2e-5, 5e-6, 1)
    assert um.VARIANTS['r3_prompt'] == {**strong, 'tune_old_private': 1}
    assert um.VARIANTS['r3_freeze'] == {**strong, 'lr_old': 0.0}
    for old in ('pilot', 'e1', 'e2', 'round2'):
        assert 'augment' not in um.VARIANTS[old] and 'tune_old_private' not in um.VARIANTS[old]


def test_round3_unknown_tag_is_refused():
    with pytest.raises(ValueError):
        um._kind('al_margin_k_B500_r3_kd')


def test_round3_starts_from_round2(tmp_path):
    runs = tmp_path / 'runs'
    (tmp_path / 'stages').mkdir()
    (tmp_path / 'stages' / 'al_margin_k_B500_r2.json').write_text(json.dumps({'stage': 'al_margin_k_B500_r2'}))
    plan = {'root': str(tmp_path), 'runs': str(runs), 'seed_task5': '/r1.pth'}
    prev, ref = um._al_start(plan, 'al_margin_k_B500_r3_train')
    assert prev == str(runs / 'EXP-FB-FT-al_margin_k_B500_r2' / 'task_5/task_final.pth')
    assert ref == 'al_margin_k_B500_r2'
    with pytest.raises(RuntimeError):
        um._al_start(plan, 'al_margin_b500_r3_rows')


def test_exp_fb_ft_round3_flags(tmp_path):
    import subprocess

    base = dict(REPO=str(REPO), FSA_HF='/fsa', SYNTH_ROOT='/synth', PREV_CKPT='/r2.pth', TRAIN_ROWS='/rows.json',
                PATH='/usr/bin:/bin')
    script = f'source {REPO}/configs/exp/native/EXP-FB-FT.sh; printf "%s\\0" "${{ARGS[@]}}"'

    def last(env):
        out = subprocess.run(['bash', '-ec', script], env=env, capture_output=True, text=True,
                             check=True).stdout.split('\0')[:-1]
        return {flag: out[i + 1] for i, flag in enumerate(out[:-1]) if flag.startswith('--')}

    old = last(base)
    assert old['--augment'] == '0' and old['--tune_old_private'] == '0'
    new = last({**base, 'AUGMENT': '1', 'TUNE_OLD_PRIVATE': '1', 'LR_OLD': '0'})
    assert (new['--augment'], new['--tune_old_private'], new['--lr_old']) == ('1', '1', '0')
    assert new.get('--augment_flip', '0') == '0'


def test_fit_only_for_runs_with_an_al_selection(tmp_path, monkeypatch):
    import tools.fit_check as fc

    assert um._fit({}, tmp_path, {'variant': 'pilot'}) == {}
    seen = {}
    monkeypatch.setattr(fc, 'fit_scores', lambda val, cfg, preds, split, ids: seen.update(split=split, ids=ids)
                        or {'cAcc_trained': 0.5, 'cAcc_pool_untrained': 0.4})
    monkeypatch.setattr(um, 'load_predictions', lambda path: path)
    monkeypatch.setattr(um, '_cfg', lambda plan: None)
    active = tmp_path / 'rel/active'
    (active / 'splits').mkdir(parents=True)
    (active / 'selections').mkdir()
    (tmp_path / 'eval').mkdir()
    (tmp_path / 'eval/val_full.json').write_text('{}')
    (active / 'splits/val_a_split.json').write_text('{"calib": [1], "pool": [2, 3]}')
    (active / 'selections/margin_k_B500_r2.json').write_text('{"image_ids": [2]}')
    fit = um._fit({'eval_root': str(tmp_path / 'eval')}, tmp_path / 'run',
                  {'selection': 'margin_k_B500_r2', 'prebuilt_release': str(tmp_path / 'rel')})
    assert fit == {'cAcc_trained': 0.5, 'cAcc_pool_untrained': 0.4} and seen['ids'] == [2]


def test_verdict_skips_the_ap50_of_the_other_skus_when_every_row_is_retrained():
    """r3_all retrains all 200 rows: AP50 of the other SKUs is NaN. The condition is reported as not measurable, not
    passed silently; a real fall still stops."""
    nan = float('nan')
    decision, reason = um.verdict(um.GATES['pilot'], 0.4368, 0.46, comparison(), nan, nan)
    assert decision == 'go' and 'AP50 other n/a' in reason
    assert um.verdict(um.GATES['pilot'], 0.4368, 0.46, comparison(), 0.969, 0.955)[0] == 'stop'


def test_the_gate_of_an_improve_run_is_a_label_and_never_stops(tmp_path, capsys):
    """12/11/2026 (doc 08 §3): an improve run is trained and scored before its gate, so a stop saves nothing and
    run_al.sh went on past it anyway. The decision is kept in the stage file; nothing is raised."""
    plan = dict(root=str(tmp_path), campaign_id='c', round=1, release_config_sha256=None, input_sha256={})
    gate = um._criteria('improve x', 'stop', 'cAcc fell', pair_drop=0.25)
    assert gate['enforced'] is False and gate['decision'] == 'stop' and gate['pair_drop'] == 0.25
    assert 'not enforced' in capsys.readouterr().out
    result = um._finish(plan, 'x', {'stage': 'x'}, gate)
    assert result['gate']['decision'] == 'stop'
    assert json.loads((tmp_path / 'stages/x.json').read_text())['gate']['enforced'] is False


def test_round4_kind_recipe_and_gate():
    """al_<selection>_r4: công thức r3_strong, gate như pilot với pair_drop 0,10 (chỉ là nhãn)."""
    assert um._kind('al_margin_k_B500_r4') == 'r4'
    assert um.VARIANTS['r4'] == um.VARIANTS['r3_strong'] and um.GATES['r4'] == um.GATES['r3_strong']


def test_round4_starts_from_the_round3_real_model(tmp_path):
    runs = tmp_path / 'runs'
    plan = {'root': str(tmp_path), 'runs': str(runs), 'seed_task5': '/r1.pth'}
    with pytest.raises(RuntimeError, match='train al_margin_k_B500_r3_real first'):
        um._al_start(plan, 'al_margin_k_B500_r4')
    (tmp_path / 'stages').mkdir()
    (tmp_path / 'stages' / 'al_margin_k_B500_r3_real.json').write_text('{}')
    prev, ref = um._al_start(plan, 'al_margin_k_B500_r4')
    assert prev == str(runs / 'EXP-FB-FT-al_margin_k_B500_r3_real' / 'task_5/task_final.pth')
    assert ref == 'al_margin_k_B500_r3_real'


def test_round3_seed_tag_repeats_a_run_with_another_seed(tmp_path):
    """al_..._r3_real_s1: dữ liệu và công thức của real, chỉ đổi seed train; xuất phát từ r2 như real."""
    assert um._kind('al_margin_k_B500_r3_real_s1') == 'r3_strong_s1'
    assert um.VARIANTS['r3_strong_s1'] == {**um.VARIANTS['r3_strong'], 'seed': 1}
    assert um.GATES['r3_strong_s1'] == um.GATES['r3_strong']
    assert um.r3_data_tag('al_margin_k_B500_r3_real_s1') == 'real'
    assert um.r3_data_tag('al_margin_k_B500_r3_all') == 'all'
    runs = tmp_path / 'runs'
    (tmp_path / 'stages').mkdir()
    (tmp_path / 'stages' / 'al_margin_k_B500_r2.json').write_text('{}')
    plan = {'root': str(tmp_path), 'runs': str(runs), 'seed_task5': '/r1.pth'}
    prev, ref = um._al_start(plan, 'al_margin_k_B500_r3_real_s1')
    assert ref == 'al_margin_k_B500_r2' and prev.endswith('EXP-FB-FT-al_margin_k_B500_r2/task_5/task_final.pth')
    with pytest.raises(ValueError):
        um._kind('al_margin_k_B500_r3_real_s9')


def test_exp_fb_ft_seed_flag_defaults_to_the_old_seed(tmp_path):
    import subprocess

    base = dict(REPO=str(REPO), FSA_HF='/fsa', SYNTH_ROOT='/synth', PREV_CKPT='/r2.pth', TRAIN_ROWS='/rows.json',
                PATH='/usr/bin:/bin')
    script = f'source {REPO}/configs/exp/native/EXP-FB-FT.sh; printf "%s\\0" "${{ARGS[@]}}"'

    def seeds(env):
        out = subprocess.run(['bash', '-ec', script], env=env, capture_output=True, text=True,
                             check=True).stdout.split('\0')[:-1]
        return [out[i + 1] for i, flag in enumerate(out[:-1]) if flag == '--seed']

    assert seeds(base)[-1] == '0'
    assert seeds({**base, 'SEED': '1'})[-1] == '1'


def test_train_environment_carries_the_seed_of_the_recipe():
    assert um._train_seed(um.VARIANTS['r3_strong']) == 0
    assert um._train_seed(um.VARIANTS['r3_strong_s1']) == 1


def _drive_campaign(tmp_path):
    """Campaign như install_r4.sh kéo về: inputs/<run>/ không có test_full.json (symlink, rclone không upload)."""
    eval_root = tmp_path / 'eval'
    eval_root.mkdir()
    (eval_root / 'test_full.json').write_text('{"images": []}')
    root = tmp_path / 'camp'
    (root / 'inputs/al_margin_k_B500_r3_real').mkdir(parents=True)
    (root / 'stages').mkdir()
    for stage in ('al_margin_k_B500_r2', 'al_margin_k_B500_r3_real'):
        (root / 'stages' / f'{stage}.json').write_text('{"cAcc_m2": 0.65}')
    return {'root': str(root), 'runs': str(root / 'runs'), 'eval_root': str(eval_root), 'dry_run': False,
            'seed_task5': str(tmp_path / 'r1.pth'), 'release_root': str(tmp_path / 'release')}


def test_link_test_full_restores_the_link_rclone_did_not_upload(tmp_path):
    plan = _drive_campaign(tmp_path)
    link = um._link_test_full(plan, 'al_margin_k_B500_r3_real')
    assert link == Path(plan['root']) / 'inputs/al_margin_k_B500_r3_real/test_full.json'
    assert link.is_symlink() and link.resolve() == (Path(plan['eval_root']) / 'test_full.json').resolve()
    assert um._link_test_full(plan, 'al_margin_k_B500_r3_real') == link


def test_link_test_full_replaces_a_broken_link(tmp_path):
    plan = _drive_campaign(tmp_path)
    link = Path(plan['root']) / 'inputs/al_margin_k_B500_r3_real/test_full.json'
    link.symlink_to(tmp_path / 'old_machine/eval/test_full.json')
    um._link_test_full(plan, 'al_margin_k_B500_r3_real')
    assert link.resolve() == (Path(plan['eval_root']) / 'test_full.json').resolve()


def test_link_test_full_refuses_a_missing_test2019_file(tmp_path):
    plan = _drive_campaign(tmp_path)
    (Path(plan['eval_root']) / 'test_full.json').unlink()
    with pytest.raises(FileNotFoundError, match='test_full.json'):
        um._link_test_full(plan, 'al_margin_k_B500_r3_real')


def test_final_test_of_a_variant_links_test_full_before_predicting(tmp_path, monkeypatch):
    plan = _drive_campaign(tmp_path)

    class Predicting(Exception):
        pass

    def train(plan_, record, attempt, name, task_dir, *args, **kwargs):
        assert (task_dir / 'test_full.json').is_file()
        raise Predicting

    monkeypatch.setattr(um, '_train', train)
    monkeypatch.setattr(um, '_attempt', _no_attempt)
    with pytest.raises(Predicting):
        um.stage_final_test(plan, confirm=True, variant='al_margin_k_B500_r3_real')


class _no_attempt:
    def __init__(self, plan, stage):
        pass

    def __enter__(self):
        return {}, Path('.')

    def __exit__(self, *exc):
        return False
