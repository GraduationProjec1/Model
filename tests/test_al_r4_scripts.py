"""Đợt GPU 1 (docs/design/2026-11-12-dot-gpu-1): scripts/run_al.sh và case al_r4 của scripts/vast_train_and_sync.sh
chạy hết B0 -> B4 -> B1 trong một lệnh: một run lỗi không chặn run sau, nhưng lỗi phải hiện ở mã thoát; chạy lại đúng
lệnh cũ thì bỏ qua phần đã xong (kể cả test2019 đã chấm).

Script được chép sang một repo tạm (REPO = thư mục cha của scripts/), python và rclone/git là bản giả, nên test không
ghi vào data/ thật và không chạm Drive.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import REPO_ROOT

R3_REAL, R3_ALL = 'al_margin_k_B500_r3_real', 'al_margin_k_B500_r3_all'
S1, R4 = 'al_margin_k_B500_r3_real_s1', 'al_margin_k_B500_r4'
R4_PQ = 'al_margin_k_B500_r4_pq'

FAKE_PYTHON = r"""#!/bin/bash
echo "$*" >> "$CALLS"
[[ $1 == -m ]] || exit 0
module=$2; shift 2
stage=""; variant=""; name=""; test_pred=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage) stage=$2; shift ;;
        --variant) variant=$2; shift ;;
        --name) name=$2; shift ;;
        --test-pred) test_pred=$2; shift ;;
    esac
    shift
done
camp=$REPO_UNDER_TEST/data/training/update-model/relearn-fix-v2
case $module in
    autocheckout.update_model)
        if [[ $stage == final_test ]]; then
            [[ " ${FAIL_TEST:-} " == *" $variant "* ]] && { echo "Traceback: final_test $variant"; exit 1; }
        elif [[ $stage == al ]]; then
            [[ " ${FAIL_AL:-} " == *" $variant "* ]] && { echo "Traceback: al $variant"; exit 1; }
            mkdir -p "$camp/stages" "$camp/runs/EXP-FB-FT-$variant/task_5"
            touch "$camp/stages/$variant.json" "$camp/runs/EXP-FB-FT-$variant/task_5/pred_val.npz"
        fi ;;
    tools.p1_check)
        [[ -n ${FAIL_P1:-} ]] && { echo "P1 lệch"; exit 2; } ;;
    tools.al_score)
        mkdir -p "$AC_DATA/relearn/active/scored" "$AC_DATA/relearn/active/scored_test"
        touch "$AC_DATA/relearn/active/scored/$name.json"
        if [[ -n $test_pred ]]; then
            [[ -f $AC_DATA/relearn/active/scored_test/$name.json ]] && { echo "FileExistsError"; exit 1; }
            touch "$AC_DATA/relearn/active/scored_test/$name.json"
        fi ;;
esac
exit 0
"""


def _repo(tmp_path: Path, *scripts: str) -> Path:
    repo = tmp_path / 'repo'
    (repo / 'scripts').mkdir(parents=True)
    for script in scripts:
        shutil.copy(REPO_ROOT / 'scripts' / script, repo / 'scripts' / script)
    data = tmp_path / 'data'
    for v in (R3_REAL, R3_ALL, S1, R4):
        (data / 'relearn/variants' / v).mkdir(parents=True)
        (data / 'relearn/variants' / v / 'train_task_5_fix.json').write_text('{}')
    return repo


def _env(tmp_path: Path, repo: Path, **extra: str) -> dict:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / 'python').write_text(FAKE_PYTHON)
    (bin_dir / 'git').write_text('#!/bin/bash\nexit 0\n')
    for tool in ('python', 'git'):
        (bin_dir / tool).chmod(0o755)
    path = f'{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin'
    return {'PATH': path, 'HOME': str(tmp_path), 'CALLS': str(tmp_path / 'calls.txt'),
            'AC_DATA': str(tmp_path / 'data'), 'REPO_UNDER_TEST': str(repo), **extra}


def _run_al(tmp_path: Path, *args: str, **extra: str) -> subprocess.CompletedProcess:
    repo = tmp_path / 'repo' if (tmp_path / 'repo').exists() else _repo(tmp_path, 'run_al.sh')
    return subprocess.run(['bash', str(repo / 'scripts/run_al.sh'), *args], env=_env(tmp_path, repo, **extra),
                          capture_output=True, text=True)


def _calls(tmp_path: Path, module: str) -> list[str]:
    path = tmp_path / 'calls.txt'
    lines = path.read_text().splitlines() if path.exists() else []
    return [line for line in lines if f'-m {module}' in line]


def test_b0_rerun_skips_the_test2019_score_already_written(tmp_path):
    """B0 bị ngắt sau khi chấm r3_real: chạy lại đúng lệnh thì r3_real bỏ qua al_score, r3_all vẫn được chấm."""
    scored_test = tmp_path / 'data/relearn/active/scored_test'
    _repo(tmp_path, 'run_al.sh')
    scored_test.mkdir(parents=True)
    (scored_test / f'{R3_REAL}.json').write_text('{}')
    result = _run_al(tmp_path, 'test', R3_REAL, R3_ALL, '--confirm-test')
    assert result.returncode == 0, result.stdout + result.stderr
    scores = _calls(tmp_path, 'tools.al_score')
    assert [R3_ALL in c for c in scores] == [True]
    assert (scored_test / f'{R3_ALL}.json').is_file()


def test_b0_failure_of_one_model_still_scores_the_other_and_exits_nonzero(tmp_path):
    result = _run_al(tmp_path, 'test', R3_REAL, R3_ALL, '--confirm-test', FAIL_TEST=R3_REAL)
    assert result.returncode != 0
    assert (tmp_path / f'data/relearn/active/scored_test/{R3_ALL}.json').is_file()
    assert not (tmp_path / f'data/relearn/active/scored_test/{R3_REAL}.json').exists()
    assert f'[LỖI] {R3_REAL}' in result.stdout + result.stderr


def test_a_failed_training_does_not_stop_the_next_run_but_fails_the_command(tmp_path):
    result = _run_al(tmp_path, S1, R4, FAIL_AL=S1)
    assert result.returncode != 0
    trained = [c for c in _calls(tmp_path, 'autocheckout.update_model') if '--stage al' in c]
    assert len(trained) == 2 and R4 in trained[1]
    scores = _calls(tmp_path, 'tools.al_score')
    assert len(scores) == 1 and R4 in scores[0]
    assert f'[LỖI] {S1}' in result.stdout + result.stderr


def test_training_runs_score_and_exit_zero(tmp_path):
    result = _run_al(tmp_path, S1, R4)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(_calls(tmp_path, 'tools.al_score')) == 2 and len(_calls(tmp_path, 'tools.al_report')) == 1


def test_a_stale_prediction_without_a_stage_is_not_scored(tmp_path):
    """pred_val.npz của lần chạy dở trước, stage chưa có: không chấm (điểm của run chưa xong)."""
    repo = _repo(tmp_path, 'run_al.sh')
    run = repo / f'data/training/update-model/relearn-fix-v2/runs/EXP-FB-FT-{S1}/task_5'
    run.mkdir(parents=True)
    (run / 'pred_val.npz').write_text('')
    result = _run_al(tmp_path, S1, FAIL_AL=S1)
    assert result.returncode != 0 and _calls(tmp_path, 'tools.al_score') == []



def test_round4_pq_reuses_the_round4_data_and_trains_under_its_own_name(tmp_path):
    """E1 của plan A: không dựng dữ liệu mới khi đã có variant _r4; train, chấm với tên _r4_pq."""
    result = _run_al(tmp_path, R4_PQ)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _calls(tmp_path, 'tools.al_round2') == [] and _calls(tmp_path, 'tools.al_build_variant') == []
    trained = [c for c in _calls(tmp_path, 'autocheckout.update_model') if '--stage al' in c]
    assert len(trained) == 1 and f'--variant {R4_PQ}' in trained[0]
    scores = _calls(tmp_path, 'tools.al_score')
    assert len(scores) == 1 and f'--name {R4_PQ}' in scores[0]


def test_round4_pq_builds_the_round4_data_when_missing(tmp_path):
    repo = _repo(tmp_path, 'run_al.sh')
    shutil.rmtree(tmp_path / 'data/relearn/variants' / R4)
    result = _run_al(tmp_path, R4_PQ)
    assert result.returncode == 0, result.stdout + result.stderr
    [build] = _calls(tmp_path, 'tools.al_round2')
    assert '--from al_margin_k_B500_r3_real' in build
    assert repo.exists()


FAKE_RUN_AL = r"""#!/bin/bash
echo "run_al $* REUSE_FEATURES=${REUSE_FEATURES:-unset}" >> "$CALLS"
for pattern in ${FAIL_STEP:-}; do [[ " $* " == *" $pattern "* ]] && exit 1; done
exit 0
"""


def _vast(tmp_path: Path, task: str = 'al_r4', skip: tuple = (), **extra: str) -> tuple[subprocess.CompletedProcess, Path]:
    repo = _repo(tmp_path, 'vast_train_and_sync.sh')
    (repo / 'scripts/run_al.sh').write_text(FAKE_RUN_AL)
    camp = repo / 'data/training/update-model/relearn-fix-v2'
    data = tmp_path / 'data'
    need = [camp / 'plan.json', camp / 'stages/baseline.json',
            data / 'relearn/active/selections/margin_k_B500_r2.json',
            data / 'relearn/active/selections/margin_k_B500_r3.json',
            data / 'relearn/active/splits/val_a_split.json', data / 'relearn/variants/e1/train_task_5_fix.json',
            data / f'relearn/variants/{R3_REAL}/train_rows.json']
    for v in ('al_margin_k_B500_r2', R3_REAL, R3_ALL):
        need += [camp / f'stages/{v}.json', camp / f'runs/EXP-FB-FT-{v}/task_5/task_final.pth',
                 camp / f'runs/EXP-FB-FT-{v}/task_5/pred_val.npz', camp / f'scored/{v}/calibration_count_m2.json',
                 camp / f'scored/{v}/calibration_count_groups.json']
    need += [data / 'eval/val_full.json', data / 'rpc/val2019/x.jpg']
    for f in need:
        if f.name in skip:
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text('{}')
    env = _env(tmp_path, repo, SKIP_MD5='1', AUTO_DESTROY='0', SYNC_EVERY_MIN='0', **extra)
    result = subprocess.run(['bash', str(repo / 'scripts/vast_train_and_sync.sh'), task], env=env,
                            capture_output=True, text=True)
    metadata = next((repo / 'data/training/update-model').glob(f'vast_logs_{task}_*/run_metadata.txt'))
    return result, metadata


def test_al_r4_records_exit_0_when_every_step_passes(tmp_path):
    result, metadata = _vast(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Exit Code: 0' in metadata.read_text()


def test_al_r4_runs_every_step_after_a_failed_b0_and_records_the_failure(tmp_path):
    result, metadata = _vast(tmp_path, FAIL_STEP='test', R4_FREEZE='1')
    steps = [line for line in (tmp_path / 'calls.txt').read_text().splitlines() if line.startswith('run_al')]
    assert len(steps) == 3 and S1 in steps[1] and 'al_margin_k_B500_r3_freeze' in steps[2]
    assert result.returncode != 0
    text = metadata.read_text()
    assert 'Exit Code: 0' not in text and '[LỖI]' in text and 'B0' in text


def _steps(tmp_path: Path) -> list[str]:
    return [line for line in (tmp_path / 'calls.txt').read_text().splitlines() if line.startswith('run_al')]


def test_attn_e1_checks_p1_then_trains_r4_pq_with_reuse_and_breaks_down_the_errors(tmp_path):
    result, metadata = _vast(tmp_path, 'attn_e1')
    assert result.returncode == 0, result.stdout + result.stderr
    [check] = _calls(tmp_path, 'tools.p1_check')
    assert f'EXP-FB-FT-{R3_REAL}/task_5/task_final.pth' in check and '--device cuda' in check
    assert '--img-root ' + str(tmp_path / 'data/rpc') in check
    assert _steps(tmp_path) == [f'run_al {R4_PQ} REUSE_FEATURES=1']
    [breakdown] = _calls(tmp_path, 'tools.error_breakdown')
    assert f'EXP-FB-FT-{R4_PQ}/task_5/pred_val.npz' in breakdown and f'scored/{R4_PQ}.json' in breakdown
    assert 'Exit Code: 0' in metadata.read_text()


def test_attn_e1_trains_without_reuse_when_p1_changes_the_outputs(tmp_path):
    result, metadata = _vast(tmp_path, 'attn_e1', FAIL_P1='1')
    assert _steps(tmp_path) == [f'run_al {R4_PQ} REUSE_FEATURES=0']
    assert result.returncode != 0
    text = metadata.read_text()
    assert '[LỖI]' in text and 'G0' in text


def test_attn_e1_stops_before_any_gpu_work_when_an_input_is_missing(tmp_path):
    result, _ = _vast(tmp_path, 'attn_e1', skip=('val_full.json',))
    assert result.returncode == 3
    calls = tmp_path / 'calls.txt'
    text = calls.read_text() if calls.exists() else ''
    assert 'tools.p1_check' not in text and 'run_al' not in text


@pytest.mark.parametrize('task, variant', [('attn_e2', 'al_margin_k_B500_r4_pqx'), ('attn_e3', 'al_margin_k_B500_r4_x'),
                                           ('attn_e4', 'al_margin_k_B500_r4_pqc'), ('attn_e5', 'al_margin_k_B500_r4_pqxc')])
def test_attn_b_runs_train_their_own_variant(tmp_path, task, variant):
    result, metadata = _vast(tmp_path, task)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _steps(tmp_path) == [f'run_al {variant} REUSE_FEATURES=1']
    [breakdown] = _calls(tmp_path, 'tools.error_breakdown')
    assert f'EXP-FB-FT-{variant}/task_5/pred_val.npz' in breakdown
