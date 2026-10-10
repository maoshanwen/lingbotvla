#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_PATH="${1:?usage: $0 /path/to/checkpoint/hf_ckpt [method_name]}"
METHOD_NAME="${2:-three_tasks_5gpu}"
EVAL_WORKDIR="${EVAL_WORKDIR:-/home/wangsf/lingbot_competition/RoboTwin}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/wangsf/lingbot_competition/eval_results/${METHOD_NAME}}"
RESULT_JSON="${RESULT_JSON:-${OUTPUT_BASE}/${METHOD_NAME}.json}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
TASKS="click_bell,open_microwave,turn_switch"
LAUNCHER="$PROJECT_ROOT/experiment/robotwin/start_robotwin_infer_and_eval.sh"

mkdir -p "$OUTPUT_BASE/launch_state"

run_setting() {
    local setting=$1
    local gpu_offset=$2
    local num_gpus=$3
    local tasks=$4
    local port=$5
    local task_config="demo_${setting}"

    bash "$LAUNCHER" \
        --model_path "$MODEL_PATH" \
        --inference_workdir "$PROJECT_ROOT" \
        --eval_workdir "$EVAL_WORKDIR" \
        --output_base "$OUTPUT_BASE/$setting" \
        --result_json "$RESULT_JSON" \
        --method_name "$METHOD_NAME" \
        --tasks "$tasks" \
        --episodes_per_task "$EPISODES_PER_TASK" \
        --num_gpus "$num_gpus" \
        --gpu_offset "$gpu_offset" \
        --num_per_gpu 1 \
        --start_port "$port" \
        --task_config "$task_config" \
        --use_compile True \
        --use_fp32 True \
        --use_bf16 False \
        --no_video \
        --conda_sh "$CONDA_SH" \
        --inference_env lingbotvla \
        --sim_env RoboTwin
}

cd "$PROJECT_ROOT"

# Clean uses three dedicated GPUs. Randomized uses two GPUs and queues click_bell
# after the shorter task, keeping all five A100s busy without oversubscribing VRAM.
run_setting clean 0 3 "$TASKS" 9460 \
    > "$OUTPUT_BASE/launch_state/clean.log" 2>&1 &
clean_pid=$!

run_setting randomized 3 2 "open_microwave,turn_switch,click_bell" 9470 \
    > "$OUTPUT_BASE/launch_state/randomized.log" 2>&1 &
randomized_pid=$!

trap 'kill "$clean_pid" "$randomized_pid" 2>/dev/null || true' INT TERM

set +e
wait "$clean_pid"
clean_status=$?
wait "$randomized_pid"
randomized_status=$?
set -e

if [ "$clean_status" -ne 0 ] || [ "$randomized_status" -ne 0 ]; then
    echo "Evaluation failed: clean=${clean_status}, randomized=${randomized_status}" >&2
    exit 1
fi

echo "Evaluation complete: $RESULT_JSON"
