#!/usr/bin/env bash
set -Eeo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wangsf/lingbot_competition/team_submission_lora}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-lingbotvla}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4}"
MASTER_PORT="${MASTER_PORT:-62520}"
BASE_MODEL="${BASE_MODEL:-/home/wangsf/lingbot_competition/models/lingbot-vla-v2-6b/lingbot-vla-v2-6b}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/wangsf/lingbot_competition/models/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${PROJECT_ROOT}/assets/training_data/robotwin_clean.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/wangsf/lingbot_competition/runs/lora-action-expert-vlm-r32-clean-v1}"
MAX_STEPS="${MAX_STEPS:-110000}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
LEARNING_RATE="${LEARNING_RATE:-2.0e-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-5}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ -f "$CONDA_SH" ]] || die "Conda profile missing: $CONDA_SH"
[[ -d "$BASE_MODEL" ]] || die "Base model missing: $BASE_MODEL"
[[ -d "$TOKENIZER_PATH" ]] || die "Tokenizer missing: $TOKENIZER_PATH"
[[ -f "$TRAIN_MANIFEST" ]] || die "Clean LeRobot manifest missing: $TRAIN_MANIFEST"
grep -Eiq 'randomized|randomisation|randomization' "$TRAIN_MANIFEST" \
  && die "Randomized data is evaluation-only: $TRAIN_MANIFEST"
manifest_count=$(grep -Ec '^[^#[:space:]]+[[:space:]]+[^[:space:]]+' "$TRAIN_MANIFEST")
[[ "$manifest_count" -eq 50 ]] \
  || die "Competition training manifest must contain exactly 50 clean tasks; found ${manifest_count}"
awk 'NF && $1 !~ /^#/ && tolower($2) !~ /clean/ {print NR ":" $0; bad=1} END {exit bad}' "$TRAIN_MANIFEST" \
  || die "Every competition training path must be a clean dataset"

gpu_count=$(awk -F, '{print NF}' <<<"$CUDA_DEVICES")
(( GLOBAL_BATCH_SIZE % gpu_count == 0 )) \
  || die "GLOBAL_BATCH_SIZE must be divisible by GPU count"

mkdir -p "$OUTPUT_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export MASTER_PORT TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec bash train.sh tasks/vla/train_lingbotvla.py configs/vla/robotwin/robotwin_lora.yaml \
  --model.model_path "$BASE_MODEL" \
  --model.tokenizer_path "$TOKENIZER_PATH" \
  --data.train_path "$TRAIN_MANIFEST" \
  --train.output_dir "$OUTPUT_DIR" \
  --train.max_steps "$MAX_STEPS" \
  --train.save_steps "$SAVE_STEPS" \
  --train.lr "$LEARNING_RATE" \
  --train.global_batch_size "$GLOBAL_BATCH_SIZE"
