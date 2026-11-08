#!/usr/bin/env python3
"""HW1 Task 3: equal-token-budget LSTM vs modern Transformer context sweep.

Run from anywhere, after activating the nanolm Python virtual environment:
    python scripts/run_task3.py --dry-run
    python scripts/run_task3.py --run
    python scripts/run_task3.py --summarize

The script uses Task 2's proven model overrides and the professor's nanolm.sweep.
Experiments run in the professor's llm directory. Results and their plot are
exported into task3_context_length/ in the assignment repository.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

CONTEXTS = (64, 128, 256, 512)
MODELS = ("lstm", "gpt_modern")
TOKENS_PER_STEP = 8192
SWEEP_NAME = "hw1_context_length"


def professor_dir(repo: Path) -> Path:
    configured = os.environ.get("NANOLM_DIR")
    directory = (Path(configured).expanduser() if configured else
                 repo / "DeepDataMiningLearning" / "DeepDataMiningLearning" / "llm")
    directory = directory.resolve()
    if not (directory / "nanolm" / "sweep.py").is_file():
        raise FileNotFoundError(
            f"nanolm not found at {directory}. Set NANOLM_DIR to the professor's llm folder."
        )
    return directory


def prepare(repo: Path, llm: Path, steps: int, seed: int) -> Path:
    import yaml

    source_file = repo / "task2_architectures" / "hw1_char_models_full.yaml"
    if not source_file.is_file():
        raise FileNotFoundError(
            f"Missing Task 2 sweep: {source_file}. Task 3 reuses the model settings you tested."
        )
    previous = yaml.safe_load(source_file.read_text(encoding="utf-8"))
    task2_runs = {item["name"]: item for item in previous.get("runs", [])}
    for name in MODELS:
        if name not in task2_runs:
            raise ValueError(f"Task 2 YAML is missing model {name!r}.")

    raw_base = Path(previous["base"])
    base = raw_base if raw_base.is_absolute() else (source_file.parent / raw_base).resolve()
    if not base.is_file():
        # Some setup scripts rewrite the professor's sweep paths when copying.
        fallback = llm / "nanolm" / "configs" / "02_char_transformer.yaml"
        if not fallback.is_file():
            raise FileNotFoundError(f"Neither Task 2 base {base} nor {fallback} exists.")
        base = fallback

    task3 = repo / "task3_context_length"
    task3.mkdir(parents=True, exist_ok=True)
    sweep_file = task3 / f"{SWEEP_NAME}.yaml"

    common = dict(previous.get("common") or {})
    common.update({
        "train.max_steps": steps,
        "train.grad_accum": 1,
        "train.max_time_sec": 0.0,
        "train.eval_every": 250,
        "train.log_every": 50,
        "train.compile": False,
        "train.save_checkpoint": False,
        "eval.sample_tokens": 0,
        "eval.final_eval_tokens": 0,
        "eval.eval_seq_len": 0,
    })

    runs = []
    for name in MODELS:
        # Match Task 2 architecture, LR, weight decay, and all other run overrides.
        inherited = dict(task2_runs[name].get("overrides") or {})
        for context in CONTEXTS:
            overrides = dict(inherited)
            overrides.update({
                "train.seq_len": context,
                "train.batch_size": TOKENS_PER_STEP // context,
            })
            runs.append({"name": f"{name}_ctx{context}", "overrides": overrides})

    spec = {
        "name": SWEEP_NAME,
        "base": os.path.relpath(base, start=task3),
        "seeds": [seed],
        "common": common,
        "runs": runs,
    }
    sweep_file.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    print(f"Created {sweep_file}", flush=True)
    print(f"8 runs, {steps} steps each, {TOKENS_PER_STEP * steps:,} training tokens each, seed {seed}", flush=True)
    return sweep_file


def launch(llm: Path, sweep: Path, dry_run: bool, reclaim: bool) -> None:
    command = [sys.executable, "-m", "nanolm.sweep", "--sweep", str(sweep)]
    if dry_run:
        command.append("--dry-run")
    if reclaim:
        command.append("--reclaim")
    if not dry_run and platform.system() == "Darwin" and shutil.which("caffeinate"):
        command = ["caffeinate", "-i", *command]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=llm, check=True)


def summarize(repo: Path, llm: Path, steps: int, seed: int) -> None:
    task3 = repo / "task3_context_length"
    root = llm / "outputs" / "nanolm" / "sweeps" / SWEEP_NAME
    evidence = task3 / "evidence"
    rows = []
    missing = []
    expected_tokens = TOKENS_PER_STEP * steps
    for model in MODELS:
        for context in CONTEXTS:
            name = f"{model}_ctx{context}"
            run = root / f"{name}_s{seed}"
            metrics_path = run / "metrics.json"
            if not metrics_path.is_file():
                missing.append(str(metrics_path))
                continue
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            if data.get("tokens_seen") != expected_tokens or data.get("steps") != steps:
                raise ValueError(f"Budget mismatch for {name}: {data.get('tokens_seen')} tokens, {data.get('steps')} steps")
            row = {
                "architecture": model,
                "context_length": context,
                "batch_size": TOKENS_PER_STEP // context,
                "seed": seed,
                "steps": data["steps"],
                "tokens_seen": data["tokens_seen"],
                "params": data["params"],
                "val_loss": data["val_loss"],
                "val_bpb": data["val_bpb"],
                "val_ppl": data["val_ppl"],
                "run_dir": str(run),
            }
            rows.append(row)
            dest = evidence / name
            dest.mkdir(parents=True, exist_ok=True)
            for filename in ("metrics.json", "config.yaml", "train_log.jsonl", "eval_log.jsonl"):
                file = run / filename
                if file.exists():
                    shutil.copy2(file, dest / filename)

    if missing:
        print(f"Found {len(rows)}/8 completed runs. Missing:", file=sys.stderr)
        for entry in missing:
            print("  ", entry, file=sys.stderr)
        raise RuntimeError("Run or resume --run before creating a complete Task 3 chart")

    csv_path = task3 / "task3_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, label in (("lstm", "LSTM (256)"), ("gpt_modern", "Modern Transformer")):
        selected = [r for r in rows if r["architecture"] == model]
        ax.plot([r["context_length"] for r in selected],
                [r["val_bpb"] for r in selected], marker="o", linewidth=2, label=label)
    ax.set_xticks(CONTEXTS)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Validation bits per byte (lower is better)")
    ax.set_title("HW1 Task 3: Context length vs. validation bpb")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    chart = task3 / "task3_context_bpb.png"
    fig.savefig(chart, dpi=180)
    plt.close(fig)

    md = ["# Task 3: Context length", "", "All eight runs use the same token budget and seed.", "",
          "| Model | Context | Batch | Parameters | Validation bpb |",
          "|---|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['architecture']} | {r['context_length']} | {r['batch_size']} | {r['params']:,} | {r['val_bpb']:.6f} |")
    md += ["", "![Context length bpb plot](task3_context_bpb.png)", "",
           "Context-length saturation and interpretation should be written after reviewing these results.", ""]
    (task3 / "task3_results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved: {csv_path}\nSaved: {chart}\nSaved: {task3 / 'task3_results.md'}")
    print(f"Saved run evidence: {evidence}")
    print("\nValidation bpb by context:")
    for r in rows:
        print(f"  {r['architecture']:11}  context={r['context_length']:3}  bpb={r['val_bpb']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true", help="Generate and validate the 8-run plan; do not train")
    modes.add_argument("--run", action="store_true", help="Train/resume eight runs, then export results")
    modes.add_argument("--summarize", action="store_true", help="Generate CSV and figure from already finished runs")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps per run (default: 2000)")
    parser.add_argument("--seed", type=int, default=1337, help="Common seed (default: 1337)")
    parser.add_argument("--reclaim", action="store_true", help="Reclaim stale sweep job locks ONLY when no other sweep is running")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    repo = Path(__file__).resolve().parents[1]
    llm = professor_dir(repo)
    if args.summarize:
        summarize(repo, llm, args.steps, args.seed)
        return
    sweep = prepare(repo, llm, args.steps, args.seed)
    launch(llm, sweep, args.dry_run, args.reclaim)
    if args.run:
        summarize(repo, llm, args.steps, args.seed)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
