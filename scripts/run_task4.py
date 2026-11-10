#!/usr/bin/env python3
"""HW1 Task 4: attention maps and quantitative attention-head analysis.

Run from the assignment repository, with the same Python environment used for training:
    python scripts/run_task4.py --dry-run
    python scripts/run_task4.py --run

Uses the professor's `python -m nanolm.viz attention --run ... --text ...`
for each of three texts. Also uses `nanolm.viz.attention_maps` on the SAME
checkpoint to compute mean attention on previous-token and current-word-start
positions for every layer/head. No retraining is necessary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

DEFAULT_TEXTS = {
    "dialogue": "ROMEO:\nBut soft! what light through yonder window breaks?",
    "repetition": "the king and the queen and the king return.",
    "punctuation": "To be, or not to be: that is the question.",
}


def paths(args):
    root = Path(__file__).resolve().parents[1]
    llm_dir = Path(
        args.llm_dir or os.environ.get("NANOLM_DIR")
        or root / "DeepDataMiningLearning" / "DeepDataMiningLearning" / "llm"
    ).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "task4_attention"
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_candidates = [
            llm_dir / "outputs/nanolm/sweeps/hw1_context_length/gpt_modern_ctx512_s1337",
            llm_dir / "outputs/nanolm/sweeps/hw1_char_models_full/gpt_modern_s1337",
            llm_dir / "outputs/nanolm/sweeps/hw1_context_length/gpt_modern_ctx256_s1337",
        ]
        run_dir = next((p for p in run_candidates if (p / "model.pt").is_file()), run_candidates[0])
    return root, llm_dir, out, run_dir


def target_indices(text):
    """For every eligible query t, which key position carries the pattern?"""
    prev = [(t, t - 1) for t in range(1, len(text))]
    word = []
    start = None
    for t, ch in enumerate(text):
        if ch.isalpha():
            if t == 0 or not text[t - 1].isalpha():
                start = t
            elif start is not None and start < t:
                # Only characters INSIDE a word, not the first character itself.
                word.append((t, start))
        else:
            start = None
    return {"previous_token": prev, "word_start": word}


def analyze_maps(texts, all_maps):
    """Returns per-text rows and pooled per-head rows, weighting by eligible queries."""
    raw = []
    grouped = defaultdict(lambda: {"sum_weight": 0.0, "sum_uniform": 0.0, "count": 0})
    for label, text in texts.items():
        maps = all_maps[label]
        targets = target_indices(text)
        for pattern, pairs in targets.items():
            if not pairs:
                raise ValueError(f"No valid query positions for {pattern} in {label}")
            for layer, heads in enumerate(maps):
                for head in range(heads.shape[0]):
                    weights = [float(heads[head, t, s]) for t, s in pairs]
                    # At query t, a perfectly uniform causal head puts 1/(t+1)
                    # probability mass on EACH accessible key position.
                    baseline = [1.0 / (t + 1) for t, _ in pairs]
                    avg = sum(weights) / len(weights)
                    uniform = sum(baseline) / len(baseline)
                    row = {
                        "text_kind": label,
                        "pattern": pattern,
                        "layer": layer,
                        "head": head,
                        "query_count": len(pairs),
                        "mean_attention": avg,
                        "uniform_causal_baseline": uniform,
                        "enrichment_vs_uniform": avg / uniform if uniform > 0 else 0,
                    }
                    raw.append(row)
                    g = grouped[(pattern, layer, head)]
                    g["sum_weight"] += sum(weights)
                    g["sum_uniform"] += sum(baseline)
                    g["count"] += len(pairs)
    summary = []
    for (pattern, layer, head), g in grouped.items():
        avg = g["sum_weight"] / g["count"]
        uniform = g["sum_uniform"] / g["count"]
        summary.append({
            "pattern": pattern, "layer": layer, "head": head,
            "query_count": g["count"], "mean_attention": avg,
            "uniform_causal_baseline": uniform,
            "enrichment_vs_uniform": avg / uniform,
        })
    return raw, sorted(summary, key=lambda r: (r["pattern"], -r["mean_attention"]))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def chosen_heads(summary):
    prev = max((r for r in summary if r["pattern"] == "previous_token"),
               key=lambda r: r["mean_attention"])
    word_options = [r for r in summary if r["pattern"] == "word_start"
                    and (r["layer"], r["head"]) != (prev["layer"], prev["head"])]
    word = max(word_options or [r for r in summary if r["pattern"] == "word_start"],
               key=lambda r: r["mean_attention"])
    return prev, word


def make_report(out, run_dir, texts, raw, summary, selected):
    import math
    cfg_file = run_dir / "metrics.json"
    val_bpb = "not available"
    if cfg_file.exists():
        m = json.loads(cfg_file.read_text())
        val_bpb = str(m.get("val_bpb", "not available"))
    lines = [
        "# Task 4: Attention analysis (measured results)", "",
        f"**Checkpoint:** `{run_dir}`", f"**Checkpoint validation bpb:** {val_bpb}",
        "", "## Three text types", "",
    ]
    for kind, txt in texts.items():
        escaped_text = txt.replace("\n", "\\n")
        lines += [f"- **{kind}:** `{escaped_text}`; "
                  f"heatmap: `attention_{kind}.png`"]
    lines += ["", "## Quantified head patterns", "",
              "Attention weights are recomputed from the professor's `nanolm.viz.attention_maps` "
              "on the same checkpoint and text. Layer and head indices are zero-based.", ""]
    for r in selected:
        name = "Previous-token" if r["pattern"] == "previous_token" else "Current-word-start"
        lines += [
            f"### {name} candidate: Layer {r['layer']}, Head {r['head']}", "",
            f"- Mean relevant-position attention: **{100*r['mean_attention']:.2f}%** "
            f"across {r['query_count']} eligible queries in all three texts.",
            f"- Mean uniform-causal baseline on those queries: "
            f"{100*r['uniform_causal_baseline']:.2f}%.",
            f"- Enrichment over uniform causal attention: "
            f"{r['enrichment_vs_uniform']:.2f}x.", "",
            "| Text type | Queries | Average weight | Uniform baseline |", "|---|---:|---:|---:|",
        ]
        for kind in texts:
            item = next(x for x in raw if x["text_kind"] == kind and
                        x["pattern"] == r["pattern"] and x["layer"] == r["layer"] and
                        x["head"] == r["head"])
            lines.append(f"| {kind} | {item['query_count']} | "
                         f"{100*item['mean_attention']:.2f}% | "
                         f"{100*item['uniform_causal_baseline']:.2f}% |")
        lines += [""]
    lines += [
        "## Definitions and interpretation", "",
        "- **Previous-token pattern:** For every query at character position `t > 0`, "
        "measure the attention weight at key position `t-1`; average over queries.",
        "- **Current-word-start pattern:** For a letter inside a word (excluding the first "
        "letter), measure the attention weight on the first letter of that same word.",
        "- **Uniform baseline:** The mean of `1/(t+1)` over the eligible queries, reflecting "
        "how many past positions are available at each query.",
        "- **Interpretation caution:** These are the strongest *candidates* selected after "
        "inspecting the 16 heads; the averages are descriptive, not evidence that a "
        "head exclusively implements that function. Inspect the heatmaps and compare "
        "with the baseline before describing a learned specialization.",
        "", "## Output files", "",
        "- `attention_dialogue.png`, `attention_repetition.png`, `attention_punctuation.png`: "
        "full layer-by-head heatmaps generated with the professor's CLI.",
        "- `attention_scores_by_text.csv`: all heads and both patterns for each text.",
        "- `attention_scores_summary.csv`: head-level pooled averages across three texts.",
        "- `selected_heads.png`: zoomed attention panels for the two candidate heads.",
        "", "Include the three heatmaps, selected head metrics and any limitations "
        "in the final report. No values in this file come from the tutorial's sample model.", "",
    ]
    (out / "task4_results.md").write_text("\n".join(lines), encoding="utf-8")


def selected_plot(out, texts, all_maps, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for i, r in enumerate(chosen):
        for j, (kind, text) in enumerate(texts.items()):
            ax = axes[i][j]
            a = all_maps[kind][r["layer"]][r["head"]]
            ax.imshow(a, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
            ax.set_title(f"{r['pattern']} | {kind}\nlayer {r['layer']} head {r['head']}")
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")
            ax.grid(False)
    fig.savefig(out / "selected_heads.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="HW1 Task 4: three attention maps + quantitative head analysis")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Check paths and show planned analysis")
    mode.add_argument("--run", action="store_true", help="Generate heatmaps and analyze head weights")
    parser.add_argument("--llm-dir", help="Professor repository's llm directory (also NANOLM_DIR)")
    parser.add_argument("--run-dir", help="Specific trained Transformer run directory with model.pt")
    parser.add_argument("--output-dir", help="Output directory (default: task4_attention/)")
    args = parser.parse_args()
    root, llm_dir, out, run_dir = paths(args)
    print(f"Assignment repo : {root}", flush=True)
    print(f"Professor llm    : {llm_dir}", flush=True)
    print(f"Checkpoint run   : {run_dir}", flush=True)
    print(f"Outputs          : {out}", flush=True)
    if not (llm_dir / "nanolm" / "viz.py").is_file():
        parser.error("Cannot locate professor's nanolm/viz.py; use --llm-dir or NANOLM_DIR")
    if not (run_dir / "model.pt").is_file():
        parser.error("Checkpoint model.pt missing. Confirm Task 2/3 save_checkpoint=true and use --run-dir")
    for label, txt in DEFAULT_TEXTS.items():
        print(f"  {label:12s}: {txt!r}")
    if args.dry_run:
        print("Dry run passed. No model inference performed.")
        return

    # Import from professor's repository. Run under the same venv used for training.
    sys.path.insert(0, str(llm_dir))
    import torch
    from nanolm.viz import attention_maps, load_run

    cfg, model, tok, _dinfo, _data_meta = load_run(str(run_dir))
    if cfg.model.arch != "gpt":
        raise RuntimeError(f"Selected checkpoint is not a Transformer: {cfg.model.arch}")
    all_maps = {}
    for label, txt in DEFAULT_TEXTS.items():
        ids = tok.encode(txt)
        if len(ids) != len(txt):
            raise RuntimeError("Expected character tokenizer. Select your char-level Transformer checkpoint.")
        if not ids:
            raise RuntimeError("Text must not be empty")
        with torch.no_grad():
            all_maps[label] = attention_maps(model, torch.tensor([ids], dtype=torch.long))
        print(f"Computed weights for {label} ({len(ids)} characters)", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    # Professor's visualization always writes the same filename; preserve any
    # original figure, capture each newly rendered image, then restore it.
    professor_fig = llm_dir / "docs" / "nanolm_figs" / "p1_attention_maps.png"
    professor_fig.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nanolm_task4_") as tmp:
        saved = Path(tmp) / "original.png"
        original_existed = professor_fig.exists()
        if original_existed:
            shutil.copy2(professor_fig, saved)
        try:
            for label, txt in DEFAULT_TEXTS.items():
                professor_fig.unlink(missing_ok=True)
                cmd = [sys.executable, "-m", "nanolm.viz", "attention", "--run", str(run_dir), "--text", txt]
                print(f"Rendering professor heatmap: {label}", flush=True)
                subprocess.run(cmd, cwd=llm_dir, check=True)
                if not professor_fig.is_file():
                    raise RuntimeError(f"Professor's CLI failed to render {label}. See [viz] output above.")
                dest = out / f"attention_{label}.png"
                shutil.copy2(professor_fig, dest)
                print(f"Saved {dest}", flush=True)
        finally:
            if original_existed:
                shutil.copy2(saved, professor_fig)
            else:
                professor_fig.unlink(missing_ok=True)

    raw, summary = analyze_maps(DEFAULT_TEXTS, all_maps)
    write_csv(out / "attention_scores_by_text.csv", raw)
    write_csv(out / "attention_scores_summary.csv", summary)
    selected = chosen_heads(summary)
    make_report(out, run_dir, DEFAULT_TEXTS, raw, summary, selected)
    selected_plot(out, DEFAULT_TEXTS, all_maps, selected)
    for r in selected:
        print(f"[{r['pattern']}] layer {r['layer']}, head {r['head']}: "
              f"mean {100*r['mean_attention']:.2f}% (uniform {100*r['uniform_causal_baseline']:.2f}%), "
              f"{r['enrichment_vs_uniform']:.2f}x baseline", flush=True)
    print(f"\nTask 4 complete. Open {out / 'task4_results.md'}", flush=True)


if __name__ == "__main__":
    main()
