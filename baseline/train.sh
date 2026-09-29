#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$BASELINE_DIR")"
MODEL_REPO="$ROOT/lingbot-vla-v2"
WORK="$BASELINE_DIR/work"
QWEN3VL_PATH="${QWEN3VL_PATH:-$BASELINE_DIR/models/Qwen3-VL-4B-Instruct}"
MOGE_PATH="${MOGE_PATH:-$BASELINE_DIR/models/moge-2-vitb-normal/model.pt}"
MANIFEST="$WORK/clean_datasets.txt"
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
if [[ ! -f "$NORM_STATS" ]]; then
  bash -o pipefail train.sh scripts/compute_norm_stats.py ./configs/vla/norm_compute/post_data.yaml \
    --data.robot_name robotwin \
    --data.train_path "$MANIFEST" \
    --data.robot_config_root "$WORK/robot_configs" \
    --data.norm_path "$NORM_STATS" \
    --data.num_workers "${NORM_WORKERS:-4}" \
    --train.micro_batch_size "${NORM_BATCH_SIZE:-8}"
fi
[[ -f "$NORM_STATS" ]] || { echo "Normalization did not produce $NORM_STATS" >&2; exit 1; }

python "$BASELINE_DIR/make_train_config.py" \
  --qwen3vl "$QWEN3VL_PATH" --moge "$MOGE_PATH" \
  --max-steps "${MAX_STEPS:-2000}" --save-steps "${SAVE_STEPS:-500}" \
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-4}"

bash -o pipefail train.sh tasks/vla/train_lingbotvla.py "$WORK/train_config.yaml"
if ! find "$WORK/train_output/checkpoints" -type d -name hf_ckpt -print -quit | grep -q .; then
  echo "Training finished without an exported hf_ckpt; inspect $MODEL_REPO/log.txt" >&2
  exit 1
fi
echo "Training checkpoints: $WORK/train_output/checkpoints"
