#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$BASELINE_DIR")"
MODEL_REPO="$ROOT/lingbot-vla-v2"
ROBOTWIN="$ROOT/RoboTwin"
CHECKPOINT="${1:-${CHECKPOINT:-}}"
TEAM_ID="${TEAM_ID:-}"
QWEN3VL_PATH="${QWEN3VL_PATH:-$BASELINE_DIR/models/Qwen3-VL-4B-Instruct}"
CONDA_SH="${CONDA_SH:-}"

if [[ -z "$CHECKPOINT" || "$(basename "$CHECKPOINT")" != hf_ckpt || ! -d "$CHECKPOINT" ]]; then
  echo "Usage: TEAM_ID=<id> QWEN3VL_PATH=<path> CONDA_SH=<path> bash baseline/eval.sh <.../hf_ckpt>" >&2
  exit 2
fi
if [[ -z "$TEAM_ID" || -z "$CONDA_SH" || ! -f "$CONDA_SH" || ! -f "$QWEN3VL_PATH/config.json" ]]; then
  echo "Set TEAM_ID, CONDA_SH, and QWEN3VL_PATH before evaluation." >&2
  exit 2
fi
if [[ ! -f "$(dirname "$(dirname "$(dirname "$CHECKPOINT")")")/lingbotvla_cli.yaml" ]]; then
  echo "LingBot inference requires the training config beside the checkpoint output." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null || [[ -z "$(nvidia-smi -L 2>/dev/null)" ]]; then
  echo "RoboTwin evaluation requires visible NVIDIA GPUs." >&2
  exit 2
fi

RUN_ROOT="$BASELINE_DIR/work/eval/$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$RUN_ROOT"
cd "$MODEL_REPO"
declare -a runs=()
for setting in clean randomized; do
  output_base="$RUN_ROOT/$setting"
  mkdir -p "$output_base"
  QWEN3VL_PATH="$QWEN3VL_PATH" bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
    --model_path "$CHECKPOINT" \
    --eval_workdir "$ROBOTWIN" \
    --output_base "$output_base" \
    --conda_sh "$CONDA_SH" \
    --inference_env "${INFERENCE_ENV:-lingbotvla}" \
    --sim_env "${SIM_ENV:-RoboTwin}" \
    --task_config "demo_$setting" \
    --num_tasks 50 \
    --num_gpus "${EVAL_NUM_GPUS:-1}" \
    --num_per_gpu "${EVAL_NUM_PER_GPU:-1}"
  mapfile -t found < <(find "$output_base" -mindepth 2 -maxdepth 2 -name stats.txt -type f)
  if [[ ${#found[@]} -ne 1 ]]; then
    echo "Expected one completed stats.txt under $output_base, found ${#found[@]}" >&2
    exit 1
  fi
  runs+=("$(dirname "${found[0]}")")
done

python "$BASELINE_DIR/make_results.py" \
  --team-id "$TEAM_ID" \
  --clean-run "${runs[0]}" \
  --randomized-run "${runs[1]}" \
  --output "$BASELINE_DIR/work/submission/01_评测结果/results.json"
echo "Evaluation logs: $RUN_ROOT"
