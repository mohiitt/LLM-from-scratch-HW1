#!/usr/bin/env python3
"""Export Task 2 evidence and figures without copying large model checkpoints.

Requires matplotlib (included in the professor's Part 1 dependencies).
Usage: python scripts/prepare_task2_results.py --runs /path/to/sweep_root --dest task2_architectures/results_full
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

EXPECTED = ('rnn', 'gru', 'lstm', 'lstm_512', 'gpt_textbook', 'gpt_modern')
SEEDS = (1337, 42)
EVIDENCE = ('metrics.json', 'config.yaml', 'train_log.jsonl', 'eval_log.jsonl', 'meta.json')


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def get_sweep_runs(root: Path) -> tuple[dict, list[str]]:
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    warnings = []
    for name in EXPECTED:
        for seed in SEEDS:
            src = root / f'{name}_s{seed}'
            metrics = src / 'metrics.json'
            if not metrics.is_file():
                warnings.append(f'MISSING: {name} seed {seed}')
                continue
            m = json.loads(metrics.read_text())
            if m['name'] != name or m['seed'] != seed:
                raise ValueError(f'Unexpected run identity in {metrics}: {m["name"]}/{m["seed"]}')
            if m.get('tokenizer') != 'char':
                raise ValueError(f'Non-character tokenizer in {metrics}')
            runs[name][seed] = {'src': src, 'metrics': m}
    counts = {int(entry['metrics']['tokens_seen']) for s in runs.values() for entry in s.values()}
    if len(counts) > 1:
        raise ValueError(f'UNFAIR COMPARISON: different training-token budgets: {sorted(counts)}')
    datasets = {entry['metrics']['dataset'] for s in runs.values() for entry in s.values()}
    if len(datasets) > 1:
        raise ValueError(f'UNFAIR COMPARISON: different datasets: {sorted(datasets)}')
    if counts and next(iter(counts)) not in (8_192_000, 16_384_000):
        warnings.append(f'Unexpected token budget: {next(iter(counts)):,} per run')
    return runs, warnings


def save_curve(runs: dict, destination: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    any_data = False
    for name in EXPECTED:
        grouped: dict[int, list[float]] = defaultdict(list)
        for seed, rec in runs.get(name, {}).items():
            for p in read_jsonl(rec['src'] / 'eval_log.jsonl'):
                if 'train_tokens' in p and 'bpb' in p:
                    grouped[int(p['train_tokens'])].append(float(p['bpb']))
        if not grouped:
            continue
        any_data = True
        steps = sorted(grouped)
        x = [s / 1e6 for s in steps]
        means = [statistics.mean(grouped[s]) for s in steps]
        lower = [min(grouped[s]) for s in steps]
        upper = [max(grouped[s]) for s in steps]
        line, = ax.plot(x, means, linewidth=2, marker='.', markersize=3, label=name)
        if len(runs.get(name, {})) >= 2:
            ax.fill_between(x, lower, upper, alpha=0.14, color=line.get_color())
    if any_data:
        ax.set_xlabel('Training tokens seen (millions)')
        ax.set_ylabel('Periodic validation bits per byte (lower is better)')
        ax.set_title('HW1 Task 2: Validation learning curves (two seeds)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(destination / 'task2_learning_curves.png', dpi=180)
    plt.close(fig)


def save_lstm_diagnostic(runs: dict, destination: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    found = False
    for name in ('lstm', 'lstm_512'):
        for field, filename, y_field in (('training', 'train_log.jsonl', 'loss'),
                                          ('validation', 'eval_log.jsonl', 'loss')):
            grouped = defaultdict(list)
            for seed, rec in runs.get(name, {}).items():
                pts = read_jsonl(rec['src'] / filename)
                for p in pts:
                    if y_field not in p:
                        continue
                    if 'train_tokens' in p:
                        tokens = float(p['train_tokens'])
                    elif 'step' in p:
                        tokens = float(p['step']) * float(rec['metrics']['tokens_seen']) / float(rec['metrics']['steps'])
                    else:
                        continue
                    grouped[tokens].append(float(p[y_field]))
            if grouped:
                found = True
                xs = sorted(grouped)
                ys = [statistics.mean(grouped[x]) for x in xs]
                ax.plot([x / 1e6 for x in xs], ys, label=f'{name}: {field}',
                        linewidth=1.9, linestyle='-' if field == 'validation' else '--')
    if found:
        ax.set_xlabel('Training tokens seen (millions)')
        ax.set_ylabel('Cross-entropy loss (nats/token)')
        ax.set_title('LSTM size comparison: training vs. periodic validation loss')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(destination / 'task2_lstm_overfit.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True, type=Path, help='Professor nanolm sweep output directory')
    p.add_argument('--dest', required=True, type=Path, help='HW1 Task 2 repository output directory')
    args = p.parse_args()
    runs, warnings = get_sweep_runs(args.runs)
    args.dest.mkdir(parents=True, exist_ok=True)

    for name in EXPECTED:
        for seed, rec in runs.get(name, {}).items():
            dst = args.dest / 'runs' / f'{name}_s{seed}'
            dst.mkdir(parents=True, exist_ok=True)
            for filename in EVIDENCE:
                src_file = rec['src'] / filename
                if src_file.exists():
                    shutil.copy2(src_file, dst / filename)
    for filename in ('sweep.yaml', 'summary.csv', 'summary.md', 'curves.png', 'bars.png'):
        f = args.runs / filename
        if f.exists():
            shutil.copy2(f, args.dest / filename)

    rows = []
    for name in EXPECTED:
        entries = runs.get(name, {})
        if not entries:
            continue
        values = [float(entries[s]['metrics']['val_bpb']) for s in SEEDS if s in entries]
        params = {int(e['metrics']['params']) for e in entries.values()}
        if len(params) != 1:
            raise ValueError(f'Inconsistent parameter counts for {name}: {params}')
        row = {'name': name, 'params': params.pop(),
               'seed_1337_bpb': entries.get(1337, {}).get('metrics', {}).get('val_bpb'),
               'seed_42_bpb': entries.get(42, {}).get('metrics', {}).get('val_bpb'),
               'mean_bpb': statistics.mean(values),
               'std_bpb': statistics.stdev(values) if len(values) >= 2 else None,
               'mean_train_seconds': statistics.mean(float(e['metrics']['train_time_sec']) for e in entries.values()),
               'seeds_complete': len(values)}
        rows.append(row)

    csv_path = args.dest / 'task2_results_table.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name','params','seed_1337_bpb','seed_42_bpb',
                                              'mean_bpb','std_bpb','mean_train_seconds','seeds_complete'])
        writer.writeheader()
        writer.writerows(rows)

    md = [
        '# Task 2: RNN, GRU, LSTM, and Transformer Comparison',
        '',
        'All bpb numbers below are from real saved `metrics.json` files. Lower is better.',
        '',
        '| Architecture | Parameters | Seed 1337 bpb | Seed 42 bpb | Mean +/- sample SD bpb |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        fnum = lambda x: f'{float(x):.4f}' if x is not None else 'pending'
        meanstd = f'{row["mean_bpb"]:.4f} +/- {row["std_bpb"]:.4f}' if row['std_bpb'] is not None else f'{row["mean_bpb"]:.4f} (1 seed)'
        md.append(f'| {row["name"]} | {row["params"]:,} | {fnum(row["seed_1337_bpb"])} | '
                  f'{fnum(row["seed_42_bpb"])} | {meanstd} |')
    md.extend(['', f'Run root: `{args.runs}`', '', 'Training dataset and token budget are checked across completed runs.',
               'Periodic training curves use `eval_log.jsonl`; final values use full validation-set `metrics.json`.',
               '', 'Caution: all interpretations must use these results, not numbers in the tutorial.', ''])
    if warnings:
        md.extend(['## Incomplete or unusual results', ''] + [f'- {w}' for w in warnings] + [''])
    (args.dest / 'task2_results_table.md').write_text('\n'.join(md))
    save_curve(runs, args.dest)
    save_lstm_diagnostic(runs, args.dest)
    print('\n'.join(md))
    print(f'\nSaved evidence and plots in {args.dest}')
    if warnings:
        print('NOTE: Task 2 is not yet complete; do not submit an incomplete table as final evidence.')


if __name__ == '__main__':
    main()
