#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wangsf/lingbot_competition/team_submission_lora}"
MODEL_PATH="${1:?usage: $0 /path/to/checkpoint/hf_ckpt [method_name]}"
METHOD_NAME="${2:-lora_sample_eval}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/wangsf/lingbot_competition/eval_results/lora_sample}"
RESULT_JSON="${RESULT_JSON:-${OUTPUT_BASE}/${METHOD_NAME}.json}"

# Calibrated against the 50-task clean baseline:
# 0%, 2%, 26% -> 9.33%, versus the full clean score of 9.38%.
SAMPLE_TASKS="place_burger_fries,rotate_qrcode,open_laptop"

cd "$PROJECT_ROOT"
exec bash experiment/robotwin/start_robotwin_infer_and_eval.sh \
  --model_path "$MODEL_PATH" \
  --inference_workdir "$PROJECT_ROOT" \
  --eval_workdir /home/wangsf/lingbot_competition/RoboTwin \
  --output_base "$OUTPUT_BASE" \
  --result_json "$RESULT_JSON" \
  --method_name "$METHOD_NAME" \
  --tasks "$SAMPLE_TASKS" \
  --episodes_per_task 20 \
  --num_gpus 3 \
  --num_per_gpu 1 \
  --start_port 9460 \
  --task_config demo_clean \
  --use_compile False \
  --use_fp32 True \
  --use_bf16 False \
  --no_video \
  --conda_sh /opt/conda/etc/profile.d/conda.sh \
  --inference_env lingbotvla \
  --sim_env RoboTwin
