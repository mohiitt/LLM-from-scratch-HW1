# HW1 Task 2 - architecture comparison

The professor's `nanolm` implementation trains all six configurations on the
same Tiny Shakespeare character-level data with two seeds each.

## Mac instructions

Keep this HW1 repo next to the professor's cloned repo under `~/DL-Assignment/`.
Activate the `.venv` created earlier in the professor repo's `llm` directory.
From the root of your HW1 repository, run:

```bash
# Inspect the 12 planned runs before training
bash scripts/run_task2.sh full dry-run

# Run all 12 experiments, analyze them, and copy small evidence back here
bash scripts/run_task2.sh full
```

If the full 2000-step experiment is too slow, **instead** run the reduced
1000-step mode with its own output directory (do not mix budgets):

```bash
bash scripts/run_task2.sh quick dry-run
bash scripts/run_task2.sh quick
```

To export completed results again without retraining:

```bash
bash scripts/run_task2.sh full export-only
```

A sweep can be resumed by repeating the same command after Ctrl-C. If a process
was forcibly killed leaving stale `CLAIM` files, ensure no sweep is running and use:

```bash
bash scripts/run_task2.sh full reclaim
```

If automatic professor-repo discovery fails, run:

```bash
export NANOLM_DIR="$HOME/DL-Assignment/DeepDataMiningLearning/DeepDataMiningLearning/llm"
bash scripts/run_task2.sh full
```

## What this creates

`task2_architectures/results_full/` (or `results_quick/`) will contain:

- `task2_results_table.md` and `.csv`: parameter counts and 2-seed bpb results.
- `task2_learning_curves.png`: bpb vs. training tokens, by model.
- `task2_lstm_overfit.png`: training and validation loss for both LSTMs.
- `runs/*/metrics.json`, `config.yaml`, `train_log.jsonl`, `eval_log.jsonl`, `meta.json`.
- Professor's original `summary.md`, `summary.csv`, `curves.png`, `bars.png`, `sweep.yaml`.

The original Modern Transformer checkpoints remain in the professor repo under
`outputs/nanolm/sweeps/hw1_char_models_full/gpt_modern_s1337/model.pt`
and `gpt_modern_s42/model.pt` for the attention analysis in Task 4.
Other checkpoints are disabled to save disk space.

Share `task2_results_table.md`, both task2 figures, and optionally `summary.md`
with ChatGPT to write the evidence-based report section.

Do **not** push the model checkpoints or downloaded datasets to GitHub.
