#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wangsf/lingbot_competition/team_submission_lora}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/wangsf/lingbot_competition/data/RoboTwin2.0-clean/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/wangsf/lingbot_competition/data/lerobot_clean_v3}"
WORKERS="${WORKERS:-10}"
PYTHON="${PYTHON:-/home/wangsf/.conda/envs/lingbotvla/bin/python}"
LOG_ROOT="${LOG_ROOT:-/home/wangsf/lingbot_competition/data_conversion_logs}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$PROJECT_ROOT/assets/training_data"
mapfile -t zips < <(find "$SOURCE_ROOT" -mindepth 2 -maxdepth 2 -type f -name '*clean_50.zip' | sort)
[[ ${#zips[@]} -eq 50 ]] || { echo "Expected 50 clean ZIP files, found ${#zips[@]}" >&2; exit 1; }

export PROJECT_ROOT OUTPUT_ROOT PYTHON LOG_ROOT PYTHONNOUSERSITE=1
printf '%s\0' "${zips[@]}" | xargs -0 -n1 -P "$WORKERS" bash -c '
  zip_path="$1"
  task_name=$(basename "$(dirname "$zip_path")")
  "$PYTHON" "$PROJECT_ROOT/scripts/convert_robotwin_clean_to_lerobot.py" \
    "$zip_path" "$OUTPUT_ROOT" --val-episodes 5 \
    >"$LOG_ROOT/${task_name}.log" 2>&1
' _

train_manifest="$PROJECT_ROOT/assets/training_data/robotwin_clean.txt"
val_manifest="$PROJECT_ROOT/assets/training_data/robotwin_clean_val.txt"
: > "$train_manifest"
: > "$val_manifest"
for zip_path in "${zips[@]}"; do
  task_name=$(basename "$(dirname "$zip_path")")
  train_dir="$OUTPUT_ROOT/${task_name}-aloha-agilex_clean_50-train"
  val_dir="$OUTPUT_ROOT/${task_name}-aloha-agilex_clean_50-val"
  [[ -f "$OUTPUT_ROOT/.${task_name}.complete" && -f "$train_dir/meta/info.json" && -f "$val_dir/meta/info.json" ]] \
    || { echo "Incomplete conversion: $task_name" >&2; exit 1; }
  printf 'robotwin %s\n' "$train_dir" >> "$train_manifest"
  printf 'robotwin %s\n' "$val_dir" >> "$val_manifest"
done

echo "Conversion complete: ${#zips[@]} tasks"
echo "Train manifest: $train_manifest"
echo "Validation manifest: $val_manifest"
