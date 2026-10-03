#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$BASELINE_DIR")"
MODEL_REPO="$ROOT/lingbot-vla-v2"
ROBOTWIN="$ROOT/RoboTwin"
CHECKPOINT="${1:-${CHECKPOINT:-}}"
TEAM_ID="${TEAM_ID:-}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
EVAL_TASKS="${EVAL_TASKS:-all}"
# Resolve and validate before compatibility workarounds or GPU startup.
EVAL_TASKS=$(python - "$EVAL_TASKS" "$EVAL_EPISODES" "$ROOT/灵波2026/初赛评测结果模板_JSON版.json" <<'PY_TASKS'
import json
import sys
all_tasks = list(json.load(open(sys.argv[3]))["results"]["clean"])
recommended = ["click_bell", "press_stapler", "turn_switch", "shake_bottle",
               "open_microwave", "place_a2b_left", "pick_dual_bottles", "stack_blocks_two"]
selection = sys.argv[1]
tasks = all_tasks if selection == "all" else recommended if selection == "recommended8" else selection.split(",")
tasks = [task.strip() for task in tasks]
if not sys.argv[2].isascii() or not sys.argv[2].isdigit() or not 1 <= int(sys.argv[2]) <= 100:
    sys.exit("EVAL_EPISODES must be an integer from 1 to 100")
if not tasks or len(set(tasks)) != len(tasks) or any(task not in all_tasks for task in tasks):
    sys.exit("EVAL_TASKS must be all, recommended8, or comma-separated unique official task names")
print("all" if selection == "all" else ",".join(tasks))
PY_TASKS
)
EVAL_EPISODES=$((10#$EVAL_EPISODES))
export EVAL_TASKS EVAL_EPISODES
QWEN3VL_PATH="${QWEN3VL_PATH:-$BASELINE_DIR/models/Qwen3-VL-4B-Instruct}"
CONDA_SH="${CONDA_SH:-}"
if [[ -n "$CONDA_SH" ]]; then
  CONDA_DIR="$(cd "$(dirname "$CONDA_SH")" 2>/dev/null && pwd)" || CONDA_DIR=""
  if [[ -n "$CONDA_DIR" ]]; then
    CONDA_SH="$CONDA_DIR/$(basename "$CONDA_SH")"
    export CONDA_SH
  fi
fi

if [[ "${EVAL_COMPAT:-0}" == 1 && "${EVAL_COMPAT_ACTIVE:-0}" != 1 ]]; then
  EVAL_COMPAT_ACTIVE=1 exec python "$BASELINE_DIR/run_eval_compat.py" "$CHECKPOINT"
fi

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
task_args=(--num_tasks 50)
if [[ "$EVAL_TASKS" != all ]]; then
  task_args=(--tasks "$EVAL_TASKS")
fi
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
    "${task_args[@]}" \
    --test_num "$EVAL_EPISODES" \
    --num_gpus "${EVAL_NUM_GPUS:-1}" \
    --num_per_gpu "${EVAL_NUM_PER_GPU:-1}" \
    --use_length "${EVAL_USE_LENGTH:-50}" \
    --use_bf16 "${EVAL_USE_BF16:-False}" \
    --use_fp32 "${EVAL_USE_FP32:-True}" \
    --use_compile "${EVAL_USE_COMPILE:-True}"
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
  --tasks "$EVAL_TASKS" \
  --episodes "$EVAL_EPISODES" \
  --output "$RUN_ROOT/results.json"
IFS=',' read -r -a selected_tasks <<< "$EVAL_TASKS"
if [[ ( "$EVAL_TASKS" == all || ${#selected_tasks[@]} -eq 50 ) && "$EVAL_EPISODES" -eq 100 ]]; then
  mkdir -p "$BASELINE_DIR/work/submission/01_评测结果"
  cp "$RUN_ROOT/results.json" "$BASELINE_DIR/work/submission/01_评测结果/results.json"
  echo "Full evaluation submission: $BASELINE_DIR/work/submission/01_评测结果/results.json"
else
  echo "Subset evaluation result: $RUN_ROOT/results.json"
fi
echo "Evaluation logs: $RUN_ROOT"
