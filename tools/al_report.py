"""A7: one table and one curve of the active-learning runs. Per run: cAcc-M2 on Val-B with Val-A-calib thresholds
(tools/al_score.py, the main number), the improve-mode numbers of ``stages/<run>.json`` (Val-B-calibrated cAcc,
AP50 of the SKUs not retrained, confusions inside the 17 rows, gate) and the labelling cost from the manifest.
Writes ``<data>/relearn/active/report/{al_results.md, al_results.json, al_curve.png}``.

    python -m tools.al_report
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from autocheckout.datapaths import REPO, Layout
from autocheckout.io import load_json, save_json

DEFAULT_CAMPAIGN = REPO / 'data/training/update-model/relearn-fix-v2'


def collect(release: Path, campaign: Path) -> list[dict]:
    rows = []
    for scored in sorted((release / 'active/scored').glob('*.json')):
        s = load_json(scored)
        name = s['name']
        row = {'name': name, 'cAcc_calib': s['cAcc_m2']}
        manifest = release / 'variants' / name / 'manifest.json'
        if manifest.is_file():
            m = load_json(manifest)
            row.update(strategy=m['strategy'], budget=m['budget'], seed=m['seed'], round=m.get('round', 1),
                       labelled_images=m['labelled_images'], labelled_boxes=m['labelled_boxes'])
        else:
            row.update(strategy='baseline', budget=0, seed=None, round=0, labelled_images=0, labelled_boxes=0)
        stage = campaign / 'stages' / f'{name}.json'
        if stage.is_file():
            st = load_json(stage)
            pairs = st.get('comparison', {}).get('pair_confusions_in_rows', {})
            row.update(cAcc_valb_policy=st['cAcc_m2'], AP50_other_base=st['AP50_other']['base'],
                       AP50_other_new=st['AP50_other']['new'], pairs_base=pairs.get('base'),
                       pairs_new=pairs.get('new'), gate=st.get('gate', {}).get('decision'),
                       cAcc_trained=st.get('cAcc_trained'), cAcc_pool_untrained=st.get('cAcc_pool_untrained'))
        rows.append(row)
    return rows


def table(rows: list[dict], base: float | None) -> str:
    head = ('| Run | Chiến lược | Vòng | B | seed | cAcc Val-B (ngưỡng calib) | Δ so với r1 | cAcc (ngưỡng Val-B) | '
            'AP50 SKU khác (trước→sau) | Nhầm trong 17 SKU | Ảnh / box gán nhãn | Cổng | '
            'cAcc ảnh đã train / pool chưa train |')
    lines = [head, '|' + '---|' * 13]
    fmt = lambda v, p='.4f': '' if v is None else format(v, p)  # noqa: E731
    for r in sorted(rows, key=lambda r: (r['round'], r['budget'], r['strategy'], r['seed'] or 0)):
        delta = None if base is None else r['cAcc_calib'] - base
        ap = '' if 'AP50_other_new' not in r else f"{fmt(r['AP50_other_base'])}→{fmt(r['AP50_other_new'])}"
        pairs = '' if r.get('pairs_new') is None else f"{r['pairs_base']}→{r['pairs_new']}"
        fit = '' if r.get('cAcc_trained') is None else f"{fmt(r['cAcc_trained'])} / {fmt(r['cAcc_pool_untrained'])}"
        lines.append(f"| {r['name']} | {r['strategy']} | {r['round']} | {r['budget']} | "
                     f"{'' if r['seed'] is None else r['seed']} | "
                     f"{fmt(r['cAcc_calib'])} | {fmt(delta, '+.4f')} | {fmt(r.get('cAcc_valb_policy'))} | {ap} | "
                     f"{pairs} | {r['labelled_images']} / {r['labelled_boxes']} | {r.get('gate') or ''} | "
                     f"{fit} |")
    return '\n'.join(lines)


def curve(rows: list[dict], base: float | None, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    series = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['strategy'] not in ('baseline', 'oracle', 'all') and r['round'] == 1:
            series[r['strategy']][r['budget']].append(r['cAcc_calib'])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strategy in sorted(series):
        budgets = sorted(series[strategy])
        mean = [sum(series[strategy][b]) / len(series[strategy][b]) for b in budgets]
        line, = ax.plot(budgets, mean, marker='o', label=strategy)
        if strategy == 'random':
            ax.fill_between(budgets, [min(series[strategy][b]) for b in budgets],
                            [max(series[strategy][b]) for b in budgets], color=line.get_color(), alpha=0.2,
                            label='random (min–max seed)')
    for r in rows:
        if r['strategy'] in ('oracle', 'all'):
            ax.scatter([r['budget']], [r['cAcc_calib']], marker='*', s=120, label=f"{r['strategy']} (tham chiếu)")
    if base is not None:
        ax.axhline(base, color='grey', linestyle='--', label='model r1 (trước khi học lại)')
    ax.set_xlabel('Số ảnh quầy được gán nhãn (B)')
    ax.set_ylabel('cAcc-M2 trên Val-B')
    ax.set_title('Active learning: chọn ảnh để học lại')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data', type=Path, help='the data folder (default: ../autocheckout-data or $AC_DATA)')
    ap.add_argument('--campaign', type=Path, default=DEFAULT_CAMPAIGN,
                    help='campaign folder of update_model (default: data/training/update-model/relearn-fix-v2)')
    ap.add_argument('--baseline', default='baseline_r1', help='scored name of the model before relearning')
    args = ap.parse_args(argv)
    release = Layout.at(args.data).relearn
    rows = collect(release, args.campaign)
    base = next((r['cAcc_calib'] for r in rows if r['name'] == args.baseline), None)
    out = release / 'active/report'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'al_results.md').write_text(table(rows, base) + '\n', encoding='utf-8')
    save_json(out / 'al_results.json', rows, indent=1)
    curve(rows, base, out / 'al_curve.png')
    return rows


if __name__ == '__main__':
    print(json.dumps(main(), indent=1, ensure_ascii=False))
