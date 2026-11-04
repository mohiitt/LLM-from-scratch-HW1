#!/usr/bin/env bash
# Run this script from any working directory, while the nanolm venv is active.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSIGNMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODE="${1:-full}"
ACTION="${2:-run}"
if [[ "$MODE" != "full" && "$MODE" != "quick" ]]; then
  echo "Usage: bash scripts/run_task2.sh [full|quick] [run|dry-run|export-only|reclaim]" >&2
  exit 2
fi
if [[ "$ACTION" != "run" && "$ACTION" != "dry-run" && "$ACTION" != "export-only" && "$ACTION" != "reclaim" ]]; then
  echo "Unknown action: $ACTION" >&2
  exit 2
fi

# The user can override auto-detection: export NANOLM_DIR=/path/to/DeepDataMiningLearning/llm
if [[ -n "${NANOLM_DIR:-}" ]]; then
  COURSE_LLM_DIR="$NANOLM_DIR"
else
  COURSE_LLM_DIR=""
  for candidate in \
    "$ASSIGNMENT_ROOT/../DeepDataMiningLearning/DeepDataMiningLearning/llm" \
    "$ASSIGNMENT_ROOT/../DeepDataMiningLearning/llm" \
    "$ASSIGNMENT_ROOT/../llm"; do
    if [[ -f "$candidate/nanolm/sweep.py" ]]; then
      COURSE_LLM_DIR="$candidate"
      break
    fi
  done
fi
if [[ ! -f "$COURSE_LLM_DIR/nanolm/sweep.py" ]]; then
  echo "Cannot find nanolm. Set NANOLM_DIR to the professor repository's llm directory." >&2
  exit 2
fi

SWEEP_NAME="hw1_char_models_${MODE}"
FILE_NAME="${SWEEP_NAME}.yaml"
SWEEP_SRC="$ASSIGNMENT_ROOT/task2_architectures/$FILE_NAME"
SWEEP_DST="$COURSE_LLM_DIR/nanolm/configs/sweeps/$FILE_NAME"
OUT_DIR="$COURSE_LLM_DIR/outputs/nanolm/sweeps/$SWEEP_NAME"
EXPORT_DIR="$ASSIGNMENT_ROOT/task2_architectures/results_${MODE}"
cp "$SWEEP_SRC" "$SWEEP_DST"
cd "$COURSE_LLM_DIR"

if [[ "$ACTION" == "dry-run" ]]; then
  python -m nanolm.sweep --sweep "nanolm/configs/sweeps/$FILE_NAME" --dry-run
  exit 0
fi
if [[ "$ACTION" != "export-only" ]]; then
  if [[ "$ACTION" == "reclaim" ]]; then
    # Only do this if no other sweep process is still running.
    python -m nanolm.sweep --sweep "nanolm/configs/sweeps/$FILE_NAME" --reclaim
  else
    python -m nanolm.sweep --sweep "nanolm/configs/sweeps/$FILE_NAME"
  fi
fi

python -m nanolm.analyze "$OUT_DIR" --baseline gpt_textbook --metric val_bpb
python "$ASSIGNMENT_ROOT/scripts/prepare_task2_results.py" --runs "$OUT_DIR" --dest "$EXPORT_DIR"
echo "Results exported to: $EXPORT_DIR"
echo "Share results_${MODE}/task2_results_table.md and task2_learning_curves.png with ChatGPT for report drafting."
