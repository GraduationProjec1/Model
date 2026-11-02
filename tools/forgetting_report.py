"""B1: memory of a class-incremental run, from the accuracy matrix ``tools/eval_cl.py`` already writes
(``metrics_cl_<split>.json``: AP50 of task group j after stage k). Adds to its forgetting:

- ``ACC_k``: mean AP50 of the groups learned so far, after stage k;
- ``BWT`` (backward transfer): mean over j < T of a[T][j] - a[j][j] (negative = the model forgets);
- a heatmap of the matrix.
No model is run. Several runs can be compared (e.g. FULL-SYN r1 against naive fine-tuning, B2).

    python -m tools.forgetting_report --out docs/al/ket-qua \\
        --run r1_val=../training/full-syn-r1/runs/EXP-FULL-SYN/metrics_cl_val.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autocheckout.cl_metrics import matrix_forgetting
from autocheckout.io import load_json, save_json


def summarize(matrix: dict) -> dict:
    a = {int(k): {int(g): float(v) for g, v in row.items()} for k, row in matrix.items()}
    last = max(a)
    acc = {k: float(np.mean(list(a[k].values()))) for k in sorted(a)}
    bwt_terms = [a[last][j] - a[j][j] for j in range(1, last) if j in a and j in a[j] and j in a[last]]
    forgetting = matrix_forgetting(a)
    return {'matrix': a, 'ACC': acc, 'ACC_final': acc[last],
            'BWT': float(np.mean(bwt_terms)) if bwt_terms else None,
            'forgetting': forgetting['average'], 'forgetting_per_group': forgetting['per_group']}


def markdown(name: str, s: dict) -> str:
    stages = sorted(s['matrix'])
    lines = [f'### {name}', '', '| Sau task \\ nhóm | ' + ' | '.join(f'task {g}' for g in stages) + ' | ACC |',
             '|' + '---|' * (len(stages) + 2)]
    for k in stages:
        cells = [f"{s['matrix'][k][g] * 100:.1f}" if g in s['matrix'][k] else '' for g in stages]
        lines.append(f'| {k} | ' + ' | '.join(cells) + f" | {s['ACC'][k] * 100:.1f} |")
    lines += ['', f"ACC cuối: {s['ACC_final'] * 100:.2f} · độ quên trung bình: {s['forgetting'] * 100:+.2f} điểm · "
                  f"BWT: {s['BWT'] * 100:+.2f} điểm (AP50, %; độ quên âm / BWT dương = không quên)", '']
    return '\n'.join(lines)


def heatmap(name: str, s: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    stages = sorted(s['matrix'])
    grid = np.full((len(stages), len(stages)), np.nan)
    for i, k in enumerate(stages):
        for j, g in enumerate(stages):
            if g in s['matrix'][k]:
                grid[i, j] = s['matrix'][k][g] * 100
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(grid, cmap='viridis', vmin=0, vmax=100)
    for i in range(len(stages)):
        for j in range(len(stages)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f'{grid[i, j]:.1f}', ha='center', va='center',
                        color='white' if grid[i, j] < 60 else 'black', fontsize=9)
    ax.set_xticks(range(len(stages)), [f'nhóm {g}' for g in stages])
    ax.set_yticks(range(len(stages)), [f'sau task {k}' for k in stages])
    ax.set_title(f'AP50 (%) từng nhóm SKU: {name}')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', action='append', required=True, help='name=path/to/metrics_cl_<split>.json')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    results, parts = {}, ['# Độ quên (B1)', '']
    for item in args.run:
        name, path = item.split('=', 1)
        summary = summarize(load_json(path)['matrix'])
        results[name] = {**summary, 'source': path}
        parts.append(markdown(name, summary))
        heatmap(name, summary, args.out / f'forgetting_{name}.png')
    (args.out / 'forgetting.md').write_text('\n'.join(parts), encoding='utf-8')
    save_json(args.out / 'forgetting.json', results, indent=1)
    return results


if __name__ == '__main__':
    r = main()
    print(json.dumps({k: {m: v[m] for m in ('ACC_final', 'forgetting', 'BWT')} for k, v in r.items()}, indent=1))
