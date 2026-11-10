#!/usr/bin/env python3
"""Analyze the completed assistant-rated blind review and label its provenance accurately.

Run from assignment repo: python scripts/finalize_extra_credit_review.py
Requires the original scripts/extend_extra_credit.py, the filled
extra_credit_sampling/extended/qualitative_review.csv, and its blind_key.csv
and summary.csv from the 90-sample experiment.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
extended = root / 'extra_credit_sampling' / 'extended'
review = extended / 'qualitative_review.csv'
original_analyzer = root / 'scripts' / 'extend_extra_credit.py'
for p in (review, extended / 'blind_key.csv', extended / 'summary.csv', original_analyzer):
    if not p.is_file():
        raise SystemExit(f'Missing: {p}')

with review.open(newline='',encoding='utf-8') as f:
    rated = list(csv.DictReader(f))
assert len(rated) == 27, f'Expected 27 reviews, got {len(rated)}'
for record in rated:
    for key in ('readability_1to5','coherence_1to5','repetition_1to5'):
        if record[key] not in {'1','2','3','4','5'}:
            raise SystemExit(f'Incomplete rating at {record["review_id"]}: {key}')

subprocess.run([sys.executable, str(original_analyzer), '--analyze'],cwd=root,check=True)

# The professor's follow-up script assumes human reviewers. The user requested
# assistant scoring, so make the generated evidence accurately describe it.
summary_file = extended / 'qualitative_summary.csv'
with summary_file.open(newline='',encoding='utf-8') as f:
    reader = csv.DictReader(f)
    records = list(reader)
    fields = reader.fieldnames or []
fields = ['n_assistant_rated' if x == 'n_human_rated' else x for x in fields]
for r in records:
    r['n_assistant_rated'] = r.pop('n_human_rated')
with summary_file.open('w',newline='',encoding='utf-8') as f:
    writer = csv.DictWriter(f,fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)

md_file = extended / 'extra_credit_extended_analysis.md'
md = md_file.read_text(encoding='utf-8')
md = md.replace('Additional Seeds and Human Review','Additional Seeds and Blinded Assistant Review')
md = md.replace('one human reviewer','one AI evaluator')
md = md.replace('one human\nreviewer','one AI\nevaluator')
md = md.replace('one human','one AI')
md = md.replace('human reviewer','AI evaluator')
md = md.replace('human-rated readability','assistant-rated readability')
md = md.replace('human readability','assistant-rated readability')
md = md.replace('27 passages were randomly ordered and rated without displaying sampling settings:',
                '27 passages were randomly ordered and scored by ChatGPT without displaying sampling settings:')
md = md.replace('Ratings are subjective, from one reviewer, using 1–5 scales for readability,',
                'Ratings are subjective, produced by one AI evaluator using 1–5 scales for readability,')
md = md.replace('- State that these are descriptive results with one AI evaluator; avoid causal claims',
                '- State that these are descriptive AI-assisted ratings; avoid causal claims')
md = md.replace('## Method\n','## Method\n\n**Reviewer provenance:** The 27 passages were rated by ChatGPT, not by an independent human reviewer. Scores should be described as AI-assisted qualitative ratings; their subjectivity and limited reliability must be acknowledged.\n')
md_file.write_text(md,encoding='utf-8')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
    for r in records:
        x=float(r['mean_distinct4_percent'])
        y=float(r['mean_readability_1to5'])
        ax.scatter(x,y,s=55)
        ax.annotate(f'T={float(r["temperature"]):g}, p={float(r["top_p"]):g}',
                    (x,y),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Mean distinct character 4-grams (%)',
           ylabel='Mean assistant-rated readability (1 to 5)',
           title='Diversity versus AI-assisted readability (descriptive)')
    ax.set_ylim(0.8,5.2)
    ax.grid(alpha=.2)
    plot = extended / 'diversity_vs_readability.png'
    fig.savefig(plot,dpi=175)
    plt.close(fig)
except ImportError:
    print('Matplotlib unavailable: plot retains original label; do not claim human review.')

print('Finished. AI-assessed evidence saved in:',extended)
print('  qualitative_summary.csv')
print('  qualitative_ratings_unblinded.csv')
print('  diversity_vs_readability.png')
print('  extra_credit_extended_analysis.md')
