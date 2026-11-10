#!/usr/bin/env python3
"""HW1 extra credit: extend diversity study and add blinded human review.

From the assignment repository:
    python scripts/extend_extra_credit.py --dry-run
    python scripts/extend_extra_credit.py --run
    # Complete extended/qualitative_review.csv using blinded_texts.md
    python scripts/extend_extra_credit.py --analyze

--run invokes the existing run_extra_credit.py with 10 seeds per setting,
reusing original seed 0/1 samples when valid. This preserves the original
2-seed study in extra_credit_sampling/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

TEMPERATURES = (0.5, 1.0, 1.5)
TOP_PS = (0.8, 0.9, 1.0)
SEEDS = tuple(range(10))
REVIEW_SEEDS = (2, 5, 8)  # 3 passages per sampling setting: 27 reviews.
RANDOMIZATION_SEED = 20261108


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sample_name(t: float, p: float, seed: int) -> str:
    return f't{t:.6g}_p{p:.6g}_seed{seed}.json'


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def copy_original_samples(original: Path, extended: Path) -> int:
    reused = 0
    (extended / 'samples').mkdir(parents=True, exist_ok=True)
    for t in TEMPERATURES:
        for p in TOP_PS:
            for seed in (0, 1):
                source = original / 'samples' / sample_name(t, p, seed)
                target = extended / 'samples' / source.name
                if source.is_file() and not target.exists():
                    data = read_json(source)
                    settings = data.get('settings', {})
                    if (settings.get('temperature') == t and
                        settings.get('top_p') == p and
                        settings.get('generation_seed') == seed and
                        settings.get('new_tokens') == 200 and
                        settings.get('prompt') == 'ROMEO:\n' and
                        settings.get('top_k') == 0 and
                        isinstance(data.get('generated_text'), str) and
                        len(data['generated_text']) == 200):
                        shutil.copy2(source, target)
                        reused += 1
    return reused


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_blinded_review(extended: Path, reset: bool = False) -> None:
    review_file = extended / 'qualitative_review.csv'
    key_file = extended / 'blind_key.csv'
    passage_file = extended / 'blinded_texts.md'
    if review_file.exists() and not reset:
        print(f'Preserved existing ratings: {review_file}')
        print('Run --analyze after entering your ratings; use --reset-review ONLY to discard them.')
        return

    entries = []
    for t in TEMPERATURES:
        for p in TOP_PS:
            for seed in REVIEW_SEEDS:
                filename = sample_name(t, p, seed)
                path = extended / 'samples' / filename
                if not path.is_file():
                    raise SystemExit(f'Missing sample: {path}. First run with --run.')
                d = read_json(path)
                settings = d['settings']
                if (settings['temperature'], settings['top_p'], settings['generation_seed']) != (t, p, seed):
                    raise SystemExit(f'Unexpected sample settings: {path}')
                generated = d['generated_text']
                if len(generated) != 200:
                    raise SystemExit(f'Expected 200 characters in {path}, found {len(generated)}')
                entries.append({'temperature':t, 'top_p':p, 'seed':seed,
                                'sample_file':str(path.relative_to(extended)),
                                'text':generated})

    random.Random(RANDOMIZATION_SEED).shuffle(entries)
    review_rows, key_rows = [], []
    lines = [
        '# Blinded Qualitative Review — HW1 Extra Credit', '',
        'There are 27 passages, three per temperature/top-p condition. The sampler settings',
        'are intentionally hidden. Read each passage and record ratings in',
        '`qualitative_review.csv` by the ID shown above it.', '',
        '**Readability (1–5):** 1 = largely gibberish; 3 = understandable fragments',
        'with substantial errors; 5 = readable and mostly grammatical.', '',
        '**Coherence (1–5):** 1 = ideas do not connect; 3 = partial local continuity;',
        '5 = consistently meaningful continuation.', '',
        '**Repetition severity (1–5):** 1 = no noticeable unnatural repetition;',
        '3 = some recurring words/phrases; 5 = heavy repetitive looping.', '',
        'Use whole-number ratings. Do not view `blind_key.csv` until you have finished.',
        'The text is an unedited 200-character generated suffix for the fixed prompt `ROMEO:\\n`.', '',
    ]
    for i, entry in enumerate(entries, start=1):
        rid = f'R{i:02d}'
        review_rows.append({'review_id':rid, 'readability_1to5':'', 'coherence_1to5':'',
                            'repetition_1to5':'', 'notes':''})
        key_rows.append({k:entry[k] for k in ('temperature','top_p','seed','sample_file')} | {'review_id':rid})
        lines.extend([f'## {rid}', '', '```text', entry['text'], '```', ''])

    write_csv(review_file, list(review_rows[0]), review_rows)
    write_csv(key_file, ['review_id','temperature','top_p','seed','sample_file'], key_rows)
    passage_file.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Created: {passage_file}')
    print(f'Rate 27 texts in: {review_file}')
    print(f'Keep hidden until scoring is finished: {key_file}')


def as_rating(row: dict, column: str, rid: str) -> int:
    value = str(row.get(column,'')).strip()
    if value not in ('1','2','3','4','5'):
        raise ValueError(f'{rid}: set {column} to a whole number from 1 through 5 (currently {value!r})')
    return int(value)


def analyze(extended: Path) -> None:
    review_file = extended / 'qualitative_review.csv'
    key_file = extended / 'blind_key.csv'
    diversity_file = extended / 'summary.csv'
    for file in [review_file, key_file, diversity_file]:
        if not file.exists():
            raise SystemExit(f'Missing {file}. Run --run and fill in qualitative_review.csv first.')

    with review_file.open(encoding='utf-8', newline='') as f:
        review = list(csv.DictReader(f))
    with key_file.open(encoding='utf-8', newline='') as f:
        keys = list(csv.DictReader(f))
    with diversity_file.open(encoding='utf-8', newline='') as f:
        diversity = list(csv.DictReader(f))
    if len(review) != 27 or len(keys) != 27 or len(diversity) != 9:
        raise SystemExit('Expected 27 reviews, 27 blind keys, and 9 diversity conditions.')

    key_by_id = {r['review_id']:r for r in keys}
    if len(key_by_id) != 27:
        raise SystemExit('Duplicate blind review IDs in key file.')
    joined = []
    problems = []
    for r in review:
        rid = r['review_id']
        if rid not in key_by_id:
            problems.append(f'Unknown ID: {rid}')
            continue
        try:
            ratings = {c:as_rating(r,c,rid) for c in ('readability_1to5','coherence_1to5','repetition_1to5')}
        except ValueError as e:
            problems.append(str(e))
            continue
        k = key_by_id[rid]
        joined.append({'review_id':rid, 'temperature':float(k['temperature']),
                       'top_p':float(k['top_p']), 'seed':int(k['seed']),
                       **ratings, 'notes':r.get('notes','')})
    if problems:
        print('Please finish the ratings before analysis:')
        for item in problems[:30]:
            print(' -', item)
        raise SystemExit(f'{len(problems)} missing/invalid rating(s).')

    data_by_setting = {(float(d['temperature']),float(d['top_p'])): d for d in diversity}
    summaries = []
    for t in TEMPERATURES:
        for p in TOP_PS:
            subset = [r for r in joined if (r['temperature'],r['top_p']) == (t,p)]
            if len(subset) != len(REVIEW_SEEDS):
                raise SystemExit(f'Expected {len(REVIEW_SEEDS)} rated passages for T={t} p={p}')
            div = data_by_setting[(t,p)]
            summaries.append({
                'temperature':t, 'top_p':p, 'n_generated':int(div['n_samples']),
                'mean_distinct4_percent':round(float(div['mean_distinct4_percent']),3),
                'sd_distinct4_percentage_points':round(float(div['sd_distinct4_fraction'])*100,3),
                'n_human_rated':len(subset),
                'mean_readability_1to5':round(statistics.mean(r['readability_1to5'] for r in subset),3),
                'mean_coherence_1to5':round(statistics.mean(r['coherence_1to5'] for r in subset),3),
                'mean_repetition_severity_1to5':round(statistics.mean(r['repetition_1to5'] for r in subset),3),
            })

    out_csv = extended / 'qualitative_summary.csv'
    write_csv(out_csv, list(summaries[0]), summaries)
    write_csv(extended / 'qualitative_ratings_unblinded.csv', list(joined[0]), joined)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
        for r in summaries:
            ax.scatter(r['mean_distinct4_percent'],r['mean_readability_1to5'],s=55)
            ax.annotate(f'T={r["temperature"]:g}, p={r["top_p"]:g}',
                        (r['mean_distinct4_percent'],r['mean_readability_1to5']),
                        xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.set(xlabel='Mean distinct character 4-grams (%)',
               ylabel='Mean human readability (1 to 5)',
               title='Diversity versus human-rated readability (descriptive)')
        ax.set_ylim(0.8,5.2)
        ax.grid(alpha=.2)
        fig.savefig(extended / 'diversity_vs_readability.png',dpi=175)
        plt.close(fig)
    except ImportError:
        print('Matplotlib not installed; skipped optional readability scatter plot.')

    lines = [
        '# Extra Credit Extension — Additional Seeds and Human Review', '',
        '## Method', '',
        '- Same Tiny Shakespeare character Transformer checkpoint and prompt as the original experiment.',
        '- Temperature = 0.5, 1.0, 1.5; top-p = 0.8, 0.9, 1.0.',
        '- Ten sampling seeds per setting (90 generated 200-character passages total).',
        '- 27 passages were randomly ordered and rated without displaying sampling settings:',
        '  three independent seeds per setting (2, 5, 8).',
        '- Ratings are subjective, from one reviewer, using 1–5 scales for readability,',
        '  coherence, and repetition severity (higher repetition severity is worse).',
        '- Checkpoint validation bpb remains 2.102364; sampling does not retrain the model.', '',
        '## Quantitative and qualitative results', '',
        '| Temperature | Top-p | Distinct 4-grams % (mean ± SD) | Readability /5 | Coherence /5 | Repetition severity /5 |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for r in summaries:
        lines.append('| {temperature:g} | {top_p:g} | {mean_distinct4_percent:.2f} ± {sd_distinct4_percentage_points:.2f} | {mean_readability_1to5:.2f} | {mean_coherence_1to5:.2f} | {mean_repetition_severity_1to5:.2f} |'.format(**r))
    lines += [
        '', '![Diversity versus readability](diversity_vs_readability.png)', '',
        '## Interpretation checklist', '',
        '- Compare increasing temperature *at fixed top-p*, then top-p *at fixed temperature*.',
        '- Explain whether increased distinct-4 is accompanied by a loss of readability or coherence',
        '  in these ratings; report any exceptions.',
        '- State that these are descriptive results with one human reviewer; avoid causal claims',
        '  based on noisy sample means.',
        '- Cite two short examples from `blinded_texts.md` after revealing the key:',
        '  a relatively fluent passage and a less readable passage.',
        '- Do not equate distinct-4 diversity with text quality.', '',
        '## Limitations', '',
        'Only one trained checkpoint, one fixed prompt, 200-character passages, and one human',
        'reviewer were used. Ten generation seeds improve repeatability but do not remove',
        'subjectivity or eliminate sampling uncertainty.', '',
    ]
    result_md = extended / 'extra_credit_extended_analysis.md'
    result_md.write_text('\n'.join(lines),encoding='utf-8')
    print('Completed 27 ratings. Saved:')
    print(' -',out_csv)
    print(' -',result_md)
    if (extended / 'diversity_vs_readability.png').exists():
        print(' -',extended / 'diversity_vs_readability.png')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dry-run',action='store_true')
    modes.add_argument('--run',action='store_true')
    modes.add_argument('--prepare',action='store_true', help='Create the blinded rating sheet from completed generation')
    modes.add_argument('--analyze',action='store_true', help='Summarize the completed human ratings')
    parser.add_argument('--reset-review',action='store_true',help='Recreate a blank rating sheet; deletes manual scores')
    args = parser.parse_args()
    root = repo_root()
    original = root / 'extra_credit_sampling'
    extended = original / 'extended'
    base_script = root / 'scripts' / 'run_extra_credit.py'
    print('Assignment repo:',root)
    print('Original results:',original)
    print('Extension output:',extended)
    print('Generation: 9 settings x 10 seeds = 90 samples; 200 characters each')
    print('Manual evaluation: 27 blinded samples (3 per setting)')
    if args.dry_run:
        if not base_script.is_file():
            raise SystemExit(f'Missing {base_script}')
        print('Existing original cached samples:',sum((original/'samples'/sample_name(t,p,s)).exists() for t in TEMPERATURES for p in TOP_PS for s in (0,1)))
        print('Dry run complete; nothing generated or changed.')
        return 0
    if args.run:
        if not base_script.is_file():
            raise SystemExit(f'Missing {base_script}')
        count = copy_original_samples(original,extended)
        print(f'Reused {count} original samples in the extension directory.')
        command=[sys.executable,str(base_script),'--run','--output-dir',str(extended),
                 '--tokens','200','--seeds',','.join(map(str,SEEDS)),
                 '--temperatures',','.join(map(str,TEMPERATURES)),
                 '--top-ps',','.join(map(str,TOP_PS))]
        print('Running:', ' '.join(command), flush=True)
        subprocess.run(command,check=True)
        prepare_blinded_review(extended,reset=args.reset_review)
    elif args.prepare:
        prepare_blinded_review(extended,reset=args.reset_review)
    else:
        analyze(extended)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
