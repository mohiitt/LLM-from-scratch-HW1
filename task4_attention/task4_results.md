# Task 4: Attention analysis (measured results)

**Checkpoint:** `/Users/mohit/LLM-from-scratch-HW1/DeepDataMiningLearning/DeepDataMiningLearning/llm/outputs/nanolm/sweeps/hw1_context_length/gpt_modern_ctx512_s1337`
**Checkpoint validation bpb:** 2.1023641395133366

## Three text types

- **dialogue:** `ROMEO:\nBut soft! what light through yonder window breaks?`; heatmap: `attention_dialogue.png`
- **repetition:** `the king and the queen and the king return.`; heatmap: `attention_repetition.png`
- **punctuation:** `To be, or not to be: that is the question.`; heatmap: `attention_punctuation.png`

## Quantified head patterns

Attention weights are recomputed from the professor's `nanolm.viz.attention_maps` on the same checkpoint and text. Layer and head indices are zero-based.

### Previous-token candidate: Layer 0, Head 1

- Mean relevant-position attention: **50.35%** across 139 eligible queries in all three texts.
- Mean uniform-causal baseline on those queries: 7.41%.
- Enrichment over uniform causal attention: 6.79x.

| Text type | Queries | Average weight | Uniform baseline |
|---|---:|---:|---:|
| dialogue | 56 | 49.72% | 6.48% |
| repetition | 42 | 47.52% | 7.98% |
| punctuation | 41 | 54.09% | 8.11% |

### Current-word-start candidate: Layer 2, Head 0

- Mean relevant-position attention: **44.13%** across 82 eligible queries in all three texts.
- Mean uniform-causal baseline on those queries: 7.50%.
- Enrichment over uniform causal attention: 5.89x.

| Text type | Queries | Average weight | Uniform baseline |
|---|---:|---:|---:|
| dialogue | 37 | 43.26% | 6.83% |
| repetition | 25 | 51.75% | 8.51% |
| punctuation | 20 | 36.24% | 7.48% |

## Definitions and interpretation

- **Previous-token pattern:** For every query at character position `t > 0`, measure the attention weight at key position `t-1`; average over queries.
- **Current-word-start pattern:** For a letter inside a word (excluding the first letter), measure the attention weight on the first letter of that same word.
- **Uniform baseline:** The mean of `1/(t+1)` over the eligible queries, reflecting how many past positions are available at each query.
- **Interpretation caution:** These are the strongest *candidates* selected after inspecting the 16 heads; the averages are descriptive, not evidence that a head exclusively implements that function. Inspect the heatmaps and compare with the baseline before describing a learned specialization.

## Output files

- `attention_dialogue.png`, `attention_repetition.png`, `attention_punctuation.png`: full layer-by-head heatmaps generated with the professor's CLI.
- `attention_scores_by_text.csv`: all heads and both patterns for each text.
- `attention_scores_summary.csv`: head-level pooled averages across three texts.
- `selected_heads.png`: zoomed attention panels for the two candidate heads.

Include the three heatmaps, selected head metrics and any limitations in the final report. No values in this file come from the tutorial's sample model.
