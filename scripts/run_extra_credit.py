#!/usr/bin/env python3
"""HW1 extra credit: distinct 4-gram diversity versus temperature and top-p.

Run from LLM-from-scratch-HW1 after activating the SAME virtualenv used in HW1:

    python scripts/run_extra_credit.py --dry-run
    python scripts/run_extra_credit.py --run

Optional smaller sanity test:
    python scripts/run_extra_credit.py --quick --output-dir extra_credit_sampling/quick_test

The script REUSES an existing modern-Transformer checkpoint and the professor's
`nanolm.evaluate.generate` sampler, varying ONLY decoding parameters. As no
model weights are modified, validation bpb stays fixed at the saved checkpoint's
`val_bpb`. 4-grams are computed on generated characters only, excluding prompt.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_csv_floats(raw: str, name: str, minimum: float | None = None,
                     maximum: float | None = None) -> list[float]:
    try:
        out = [float(v.strip()) for v in raw.split(',')]
    except ValueError as exc:
        raise SystemExit(f'Invalid {name}: {raw}') from exc
    if not out or not all(math.isfinite(x) for x in out):
        raise SystemExit(f'Invalid {name}: expected finite numbers')
    if len(out) != len(set(out)):
        raise SystemExit(f'Invalid {name}: duplicate values')
    if minimum is not None and any(x < minimum for x in out):
        raise SystemExit(f'Invalid {name}: values must be >= {minimum}')
    if maximum is not None and any(x > maximum for x in out):
        raise SystemExit(f'Invalid {name}: values must be <= {maximum}')
    return out


def parse_csv_seeds(raw: str) -> list[int]:
    try:
        result = [int(x.strip()) for x in raw.split(',')]
    except ValueError as exc:
        raise SystemExit(f'Invalid seeds: {raw}') from exc
    if not result or min(result) < 0 or len(result) != len(set(result)):
        raise SystemExit('Seeds must be unique nonnegative integers')
    return result


def paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    llm = Path(args.llm_dir or os.environ.get('NANOLM_DIR') or
               root / 'DeepDataMiningLearning' / 'DeepDataMiningLearning' / 'llm').expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / 'extra_credit_sampling'
    if args.run_dir:
        run = Path(args.run_dir).expanduser().resolve()
    else:
        candidates = [
            llm / 'outputs/nanolm/sweeps/hw1_context_length/gpt_modern_ctx512_s1337',
            llm / 'outputs/nanolm/sweeps/hw1_char_models_full/gpt_modern_s1337',
        ]
        run = next((p for p in candidates if (p / 'model.pt').is_file()), candidates[0])
    return root, llm, run, out


def distinct_4grams(generated_text: str) -> tuple[int, int, float]:
    """Character 4-grams, counting overlapping windows in generated suffix only."""
    grams = [generated_text[i:i + 4] for i in range(len(generated_text) - 3)]
    total = len(grams)
    if not total:
        raise ValueError('Need at least four generated characters')
    distinct = len(set(grams))
    return distinct, total, distinct / total


def sample_path(out: Path, temperature: float, top_p: float, seed: int) -> Path:
    return out / 'samples' / f't{temperature:.6g}_p{top_p:.6g}_seed{seed}.json'


def save_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'Cannot write empty {path.name}')
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict], temperatures: list[float], top_ps: list[float]) -> list[dict]:
    summary = []
    for t in temperatures:
        for p in top_ps:
            group = [r for r in rows if r['temperature'] == t and r['top_p'] == p]
            if not group:
                continue
            scores = [float(r['distinct4_fraction']) for r in group]
            summary.append({
                'temperature': t,
                'top_p': p,
                'n_samples': len(group),
                'mean_distinct4_fraction': statistics.mean(scores),
                'sd_distinct4_fraction': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                'mean_distinct4_percent': 100.0 * statistics.mean(scores),
                'mean_distinct4_count': statistics.mean(float(r['distinct4_count']) for r in group),
                'mean_total4_count': statistics.mean(float(r['total4_count']) for r in group),
            })
    return summary


def plots(out: Path, summary: list[dict], temperatures: list[float], top_ps: list[float]):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lookup = {(r['temperature'], r['top_p']): r for r in summary}
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for p in top_ps:
        ys = [lookup[(t, p)]['mean_distinct4_percent'] for t in temperatures]
        ax.plot(temperatures, ys, marker='o', label=f'top-p = {p:g}')
    ax.set(xlabel='Sampling temperature', ylabel='Mean distinct character 4-grams (%)',
           title='Sample diversity versus temperature (same trained model)')
    ax.legend()
    ax.grid(alpha=.22)
    fig.savefig(out / 'diversity_vs_temperature.png', dpi=175)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for t in temperatures:
        ys = [lookup[(t, p)]['mean_distinct4_percent'] for p in top_ps]
        ax.plot(top_ps, ys, marker='o', label=f'temperature = {t:g}')
    ax.set(xlabel='Top-p (nucleus threshold)', ylabel='Mean distinct character 4-grams (%)',
           title='Sample diversity versus top-p (same trained model)')
    ax.legend()
    ax.grid(alpha=.22)
    fig.savefig(out / 'diversity_vs_top_p.png', dpi=175)
    plt.close(fig)

    values = [[lookup[(t,p)]['mean_distinct4_percent'] for p in top_ps] for t in temperatures]
    fig, ax = plt.subplots(figsize=(7, 4.7), constrained_layout=True)
    im = ax.imshow(values, aspect='auto')
    ax.set_xticks(list(range(len(top_ps))), [str(x) for x in top_ps])
    ax.set_yticks(list(range(len(temperatures))), [str(x) for x in temperatures])
    ax.set(xlabel='Top-p', ylabel='Temperature', title='Mean distinct character 4-grams (%)')
    fig.colorbar(im, ax=ax, label='Distinct 4-gram fraction (%)')
    for i, row in enumerate(values):
        for j, val in enumerate(row):
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    color='white' if val < (max(map(max, values)) + min(map(min, values))) / 2 else 'black')
    fig.savefig(out / 'diversity_heatmap.png', dpi=175)
    plt.close(fig)


def report(out: Path, run: Path, metrics: dict, rows: list[dict], summary: list[dict],
           args: argparse.Namespace, temperatures: list[float], top_ps: list[float], seeds: list[int]):
    lookup = {(r['temperature'], r['top_p']): r for r in summary}
    lines = [
        '# HW1 Extra Credit — Sampling Diversity at Fixed Validation BPB', '',
        '## Setup', '',
        f'- Checkpoint: `{run}`',
        f'- Fixed checkpoint validation bpb: **{metrics["val_bpb"]:.6f}**.',
        '- Dataset: Tiny Shakespeare (character tokenizer).',
        f'- Fixed prompt: `{args.prompt.encode("unicode_escape").decode("ascii")}`.',
        f'- New generated characters per sample: **{args.tokens}**.',
        f'- Generation seeds: **{", ".join(map(str, seeds))}**.',
        '- Top-k: **0** (disabled to avoid confounding temperature/top-p).',
        f'- Temperature values: **{", ".join(map(str, temperatures))}**.',
        f'- Top-p values: **{", ".join(map(str, top_ps))}**.',
        '', '## Definition', '',
        'Distinct 4-gram fraction = (number of unique overlapping character 4-grams) '
        '/ (total overlapping character 4-grams) in the **generated suffix only**. '
        'A generated text of N characters has N-3 overlapping character 4-grams. '
        'The prompt is excluded. For each setting, the table reports the mean and '
        'sample standard deviation over independent generation seeds.',
        '', '## Results', '',
        '| Temperature | Top-p | Samples | Mean distinct 4-grams (%) | Sample SD (percentage points) |',
        '|---:|---:|---:|---:|---:|',
    ]
    for t in temperatures:
        for p in top_ps:
            r = lookup[(t,p)]
            lines.append(f'| {t:g} | {p:g} | {r["n_samples"]} | '
                         f'{r["mean_distinct4_percent"]:.2f} | '
                         f'{100*r["sd_distinct4_fraction"]:.2f} |')
    lines += ['', '## Figures', '',
              '![Sampling diversity versus temperature](diversity_vs_temperature.png)', '',
              '![Sampling diversity versus top-p](diversity_vs_top_p.png)', '',
              '![Diversity heatmap](diversity_heatmap.png)', '',
              '## Interpretation guide', '',
              '- Compare temperature at fixed top-p, and top-p at fixed temperature.',
              '- Report the measured pattern, including any non-monotonic results; do not assume diversity must increase monotonically.',
              '- This is diversity, not fluency, correctness, or linguistic quality. Inspect `samples/` for illustrative text.',
              '- **Fixed bpb** means the same trained model is used for every sample. '
              'Validation bpb is calculated from its probability distribution on held-out text '
              'and does not change when only the sampling rule changes. This experiment does '
              'not claim that every generated sample has the same likelihood.',
              '- Every setting uses identical prompt, number of generated tokens, model, context '
              'window, and generation seed list; only temperature and top-p vary.',
              '- Limitations: short samples, a single checkpoint and fixed prompt; sample diversity '
              'alone does not capture semantic diversity or generation quality.',
              '', '## Reproduction', '',
              '`python scripts/run_extra_credit.py --run`', '',
              'Raw data: `samples.csv`, `summary.csv`, and individual JSON samples in `samples/`.', '',
    ]
    (out / 'extra_credit_results.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true', help='Validate paths and show planned settings; no model loading')
    mode.add_argument('--run', action='store_true', help='Run full diversity sweep (18 samples by default)')
    mode.add_argument('--quick', action='store_true', help='Small sanity test: nine 80-character samples')
    parser.add_argument('--llm-dir', help='Professor repository llm directory (or NANOLM_DIR)')
    parser.add_argument('--run-dir', help='Existing trained checkpoint directory')
    parser.add_argument('--output-dir', help='Output directory (default: extra_credit_sampling/)')
    parser.add_argument('--prompt', default='ROMEO:\n', help='Fixed prompt for all generation conditions')
    parser.add_argument('--tokens', type=int, default=200, help='Number of newly generated characters per sample (default 200)')
    parser.add_argument('--seeds', default='0,1', help='Comma-separated generation RNG seeds (default 0,1)')
    parser.add_argument('--temperatures', default='0.5,1.0,1.5')
    parser.add_argument('--top-ps', default='0.8,0.9,1.0')
    parser.add_argument('--ctx', type=int, help='Use at most this many context tokens (default: trained context)')
    args = parser.parse_args()
    root, llm, run, out = paths(args)
    if args.quick:
        args.tokens, args.seeds = 80, '0'
        if args.output_dir is None:
            out = out / 'quick_test'
    temps = parse_csv_floats(args.temperatures, 'temperatures', minimum=0.000001)
    top_ps = parse_csv_floats(args.top_ps, 'top-ps', minimum=0.000001, maximum=1.0)
    seeds = parse_csv_seeds(args.seeds)
    if args.tokens < 4:
        raise SystemExit('--tokens must be at least 4')
    if not args.prompt:
        raise SystemExit('Prompt must not be empty')
    if not (llm / 'nanolm' / 'generate.py').is_file():
        raise SystemExit(f'Professor nanolm source not found at {llm}; set NANOLM_DIR')
    if not (run / 'model.pt').is_file():
        raise SystemExit(f'Checkpoint not found: {run / "model.pt"}')
    metrics_file = run / 'metrics.json'
    if not metrics_file.is_file():
        raise SystemExit(f'Metrics file not found: {metrics_file}')
    metrics = json.loads(metrics_file.read_text(encoding='utf-8'))
    if not isinstance(metrics.get('val_bpb'), (float, int)):
        raise SystemExit('Checkpoint metrics.json missing numeric val_bpb')
    print('Assignment repo :', root)
    print('Professor llm    :', llm)
    print('Checkpoint       :', run)
    print(f'Fixed val bpb    : {metrics["val_bpb"]:.6f}')
    print('Output directory :', out)
    print(f'Settings         : {len(temps)} temperatures x {len(top_ps)} top-p x '
          f'{len(seeds)} seeds = {len(temps)*len(top_ps)*len(seeds)} samples')
    print('Generated chars  :', args.tokens, 'per sample')
    if args.dry_run:
        print('Dry-run passed. No text generated.')
        return 0
    if str(llm) not in sys.path:
        sys.path.insert(0, str(llm))
    try:
        from nanolm.generate import load_run
        from nanolm.evaluate import generate
    except ImportError as exc:
        raise SystemExit('Cannot import nanolm. Activate the virtual environment used for Tasks 2–4.') from exc
    cfg, model, tok, dinfo, _ = load_run(str(run))
    if tok.spec != 'char':
        raise SystemExit(f'Expected a character tokenizer, got {tok.spec}. '
                         'The metric is defined on characters for this assignment.')
    ctx = cfg.eval.eval_seq_len or cfg.train.seq_len
    if args.ctx is not None:
        if args.ctx < 4 or args.ctx > ctx:
            raise SystemExit(f'--ctx must be between 4 and trained context {ctx}')
        ctx = args.ctx
    print('Sampling device  :', dinfo.device)
    print('Sampling context :', ctx)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    all_rows: list[dict] = []
    for temp in temps:
        for p in top_ps:
            for seed in seeds:
                path = sample_path(out, temp, p, seed)
                settings = dict(run_dir=str(run), prompt=args.prompt, context=ctx,
                                new_tokens=args.tokens, temperature=temp, top_p=p,
                                top_k=0, generation_seed=seed)
                if path.exists():
                    saved = json.loads(path.read_text(encoding='utf-8'))
                    if saved.get('settings') != settings:
                        raise SystemExit(f'Cached sample settings differ: {path}. '
                                         'Use another --output-dir or remove the mismatched sample.')
                    generated = saved['generated_text']
                    cached = True
                else:
                    entire = generate(model, tok, dinfo, args.prompt, args.tokens, ctx,
                                      temperature=temp, top_k=0, seed=seed, top_p=p)
                    if not entire.startswith(args.prompt):
                        raise RuntimeError(f'Unexpected decoder output does not preserve prompt: {entire[:30]!r}')
                    generated = entire[len(args.prompt):]
                    save_json_atomic(path, dict(settings=settings, generated_text=generated))
                    cached = False
                distinct, total, fraction = distinct_4grams(generated)
                all_rows.append(dict(temperature=temp, top_p=p, seed=seed,
                                     generated_characters=len(generated),
                                     distinct4_count=distinct, total4_count=total,
                                     distinct4_fraction=fraction,
                                     distinct4_percent=fraction*100.0,
                                     sample_file=str(path.relative_to(out))))
                print(f'  temp={temp:g}, top-p={p:g}, seed={seed}: '
                      f'{fraction*100:.2f}% distinct 4-grams'
                      + (' [cached]' if cached else ''), flush=True)
    summary = summarize(all_rows, temps, top_ps)
    write_csv(out / 'samples.csv', all_rows)
    write_csv(out / 'summary.csv', summary)
    plots(out, summary, temps, top_ps)
    report(out, run, metrics, all_rows, summary, args, temps, top_ps, seeds)
    print(f'\nCompleted {len(all_rows)} samples in {time.monotonic()-started:.1f} seconds.')
    print('Saved:', out / 'summary.csv')
    print('Saved:', out / 'diversity_vs_temperature.png')
    print('Saved:', out / 'diversity_vs_top_p.png')
    print('Saved:', out / 'diversity_heatmap.png')
    print('Saved:', out / 'extra_credit_results.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
