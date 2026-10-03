#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$BASELINE_DIR")"
MODEL_REPO="$ROOT/lingbot-vla-v2"
WORK="$BASELINE_DIR/work"
QWEN3VL_PATH="${QWEN3VL_PATH:-$BASELINE_DIR/models/Qwen3-VL-4B-Instruct}"
MOGE_PATH="${MOGE_PATH:-$BASELINE_DIR/models/moge-2-vitb-normal/model.pt}"
MANIFEST="${MANIFEST:-$WORK/clean_datasets.txt}"
NORM_STATS="$WORK/clean_norm_stats.json"

if ! command -v nvidia-smi >/dev/null || [[ -z "$(nvidia-smi -L 2>/dev/null)" ]]; then
  echo "Training requires visible NVIDIA GPUs; none are available here." >&2
  exit 2
fi
if ! command -v torchrun >/dev/null; then
  echo "torchrun is missing; activate the LingBot-VLA training environment." >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing $MANIFEST; run: python baseline/prepare_data.py all" >&2
  exit 2
fi
if [[ ! -f "$QWEN3VL_PATH/config.json" || ! -f "$MOGE_PATH" ]]; then
  echo "Missing Qwen3-VL or MoGe-2 files; see baseline/README.md." >&2
  exit 2
fi

cd "$MODEL_REPO"
python "$BASELINE_DIR/make_robot_config.py"
NCCL_COMPAT="${NCCL_COMPAT:-auto}"
if [[ "$NCCL_COMPAT" == auto ]]; then
  if [[ "${NCCL_NET_PLUGIN:-}" == gcp-fastrak ]]; then NCCL_COMPAT=1; else NCCL_COMPAT=0; fi
fi
if [[ "$NCCL_COMPAT" == 1 ]]; then
  unset NCCL_NET_PLUGIN NCCL_TUNER_PLUGIN NCCL_TUNER_CONFIG_PATH
  export NCCL_NET=Socket
  echo "Using local NCCL compatibility settings for normalization and training"
fi
export NCCL_COMPAT
norm_entry=("$BASELINE_DIR/run_torch_entry.py" scripts/compute_norm_stats.py)
train_entry=("$BASELINE_DIR/run_torch_entry.py" tasks/vla/train_lingbotvla.py)
if [[ ! -f "$NORM_STATS" ]]; then
  # Stats need actions and states only; keep their small Arrow cache on local disk.
  NORM_CACHE_DIR="${NORM_CACHE_DIR:-/tmp/lingbot-race-norm-cache}"
  mkdir -p "$NORM_CACHE_DIR"
  HF_DATASETS_CACHE="$NORM_CACHE_DIR" bash -o pipefail train.sh "${norm_entry[@]}" ./configs/vla/norm_compute/post_data.yaml \
    --data.robot_name robotwin \
    --data.train_path "$MANIFEST" \
    --data.robot_config_root "$WORK/robot_configs" \
    --data.norm_path "$NORM_STATS" \
    --data.num_workers "${NORM_WORKERS:-4}" \
    --train.micro_batch_size "${NORM_BATCH_SIZE:-8}"
fi
[[ -f "$NORM_STATS" ]] || { echo "Normalization did not produce $NORM_STATS" >&2; exit 1; }

# Training needs image bytes; its full Arrow cache is too large for the root disk.
TRAIN_CACHE_DIR="${TRAIN_CACHE_DIR:-$WORK/cache/huggingface/datasets}"
mkdir -p "$TRAIN_CACHE_DIR"

python "$BASELINE_DIR/make_train_config.py" \
  --qwen3vl "$QWEN3VL_PATH" --moge "$MOGE_PATH" \
  --max-steps "${MAX_STEPS:-50000}" --save-steps "${SAVE_STEPS:-5000}" \
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-4}"

HF_DATASETS_CACHE="$TRAIN_CACHE_DIR" bash -o pipefail train.sh "${train_entry[@]}" "$WORK/train_config.yaml"
if ! find "$WORK/train_output/checkpoints" -type d -name hf_ckpt -print -quit | grep -q .; then
  echo "Training finished without an exported hf_ckpt; inspect $MODEL_REPO/log.txt" >&2
  exit 1
fi
echo "Training checkpoints: $WORK/train_output/checkpoints"
