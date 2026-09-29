#!/usr/bin/env bash
# Prepare clean data, train, evaluate clean/randomized, and write results.json.
set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$BASELINE_DIR")"
WORK="$BASELINE_DIR/work"

# The shell running this script must use the LingBot training environment.
if [[ -n "${TRAIN_ACTIVATE:-}" ]]; then
  source "$TRAIN_ACTIVATE"
elif [[ -f "$ROOT/.venvs/lingbot/bin/activate" ]]; then
  source "$ROOT/.venvs/lingbot/bin/activate"
fi
if [[ -z "${TEAM_ID:-1382990}" ]]; then
  echo "Set TEAM_ID to the actual competition team ID before starting." >&2
  exit 2
fi
if [[ -z "${CONDA_SH:-}" ]]; then
  if [[ -f "$ROOT/.venvs/robotwin/bin/activate" ]]; then
    CONDA_SH="$BASELINE_DIR/venv_conda_adapter.sh"
  elif [[ -f "$WORK/smoke/conda_compat.sh" ]]; then
    CONDA_SH="$WORK/smoke/conda_compat.sh"
  fi
fi
if [[ -z "${CONDA_SH:-}" || ! -f "$CONDA_SH" ]]; then
  echo "Set CONDA_SH to conda.sh (or a compatible environment adapter)." >&2
  exit 2
fi
CONDA_SH="$(cd "$(dirname "$CONDA_SH")" && pwd)/$(basename "$CONDA_SH")"
export CONDA_SH

if [[ "${PREPARE_DATA:-1}" == 1 ]]; then
  if [[ -z "${PROCESS_PYTHON:-}" ]]; then
    PROCESS_PYTHON="$(source "$CONDA_SH" && conda activate "${SIM_ENV:-RoboTwin}" && command -v python)"
  fi
  echo "Preparing 50 clean datasets with $PROCESS_PYTHON"
  python "$BASELINE_DIR/prepare_data.py" all --process-python "$PROCESS_PYTHON" \
    --convert-python "$(command -v python)" --workers "${PREPARE_WORKERS:-1}"
fi

echo "Training clean data to step ${MAX_STEPS:-2000}"
bash "$BASELINE_DIR/train.sh"

CHECKPOINT="$(python "$BASELINE_DIR/select_hf_checkpoint.py" \
  --output-dir "$WORK/train_output" --step "${MAX_STEPS:-2000}")"
echo "Evaluating exported checkpoint: $CHECKPOINT"
if [[ "${EVAL_COMPAT:-0}" == 1 ]]; then
  python "$BASELINE_DIR/run_eval_compat.py" "$CHECKPOINT"
else
  bash "$BASELINE_DIR/eval.sh" "$CHECKPOINT"
fi
echo "Results: $WORK/submission/01_评测结果/results.json"
