#!/usr/bin/env python3
"""Make a clean-only LingBot-VLA post-training configuration."""

import argparse
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).resolve().parent
MODEL_CODE = ROOT / "lingbot-vla-v2"
BASE_WEIGHTS = Path(os.environ.get("LINGBOT_BASE_WEIGHTS", ROOT / "lingbot-vla-v2-6b"))
TEMPLATE = ROOT / "灵波2026/初赛评测结果模板_JSON版.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen3vl", type=Path, required=True)
    parser.add_argument("--moge", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=BASELINE / "work/clean_datasets.txt")
    parser.add_argument("--norm-stats", type=Path, default=BASELINE / "work/clean_norm_stats.json")
    parser.add_argument("--output", type=Path, default=BASELINE / "work/train_config.yaml")
    parser.add_argument("--train-output", type=Path, default=BASELINE / "work/train_output")
    parser.add_argument("--enable-resume", choices=("0", "1"), default="1")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--data-parallel-size", type=int, default=4,
                        help="Total data-parallel GPU count; used to derive gradient accumulation")
    args = parser.parse_args()

    for path in (BASE_WEIGHTS / "model.safetensors.index.json",
                 args.qwen3vl / "config.json", args.moge, args.norm_stats, args.manifest):
        if not path.is_file():
            parser.error(f"required file missing: {path}")
    if min(args.max_steps, args.save_steps, args.micro_batch_size, args.global_batch_size, args.data_parallel_size) < 1:
        parser.error("step and batch settings must be positive")
    if args.save_steps > args.max_steps:
        parser.error("save_steps must not exceed max_steps, or no evaluation checkpoint is exported")
    if args.global_batch_size < args.micro_batch_size:
        parser.error("global batch size must be at least micro batch size")

    batch_per_micro_step = args.micro_batch_size * args.data_parallel_size
    if args.global_batch_size % batch_per_micro_step != 0:
        parser.error("global batch size must be divisible by micro batch size * data-parallel GPU count")
    gradient_accumulation_steps = args.global_batch_size // batch_per_micro_step

    expected = list(json.loads(TEMPLATE.read_text())["results"]["clean"])
    lines = args.manifest.read_text().splitlines()
    if len(lines) != 50:
        parser.error(f"manifest needs exactly 50 clean datasets, got {len(lines)}")
    for task, line in zip(expected, lines):
        parts = line.split()
        if len(parts) != 2 or parts[0] != "robotwin" or Path(parts[1]).name != task:
            parser.error(f"unexpected manifest entry for {task}: {line}")
        if "randomized" in line.lower() or "clean" not in line.lower():
            parser.error(f"non-clean dataset in manifest: {line}")
        info_path = Path(parts[1]) / "meta/info.json"
        if not info_path.is_file():
            parser.error(f"converted dataset missing: {parts[1]}")
        if json.loads(info_path.read_text()).get("total_episodes") != 50:
            parser.error(f"converted dataset does not contain 50 episodes: {parts[1]}")

    config = yaml.safe_load((MODEL_CODE / "configs/vla/robotwin/robotwin.yaml").read_text())
    config["model"]["model_path"] = str(BASE_WEIGHTS)
    config["model"]["tokenizer_path"] = str(args.qwen3vl.resolve())
    config["data"]["train_path"] = str(args.manifest.resolve())
    config["data"]["data_name"] = "multi"
    robot_config_root = BASELINE / "work/robot_configs"
    if not (robot_config_root / "robotwin.yaml").is_file():
        parser.error(f"clean robot config missing: {robot_config_root / 'robotwin.yaml'}")
    config["data"]["robot_config_root"] = str(robot_config_root)
    config["data"]["norm_stats_file"] = str(args.norm_stats.resolve())
    config["train"]["output_dir"] = str(args.train_output.resolve())
    config["train"]["enable_resume"] = args.enable_resume == "1"
    if args.enable_resume == "0":
        # Omit the key: the official YAML parser turns explicit null into the string "null".
        config["train"].pop("load_checkpoint_path", None)
    config["train"]["max_steps"] = args.max_steps
    config["train"]["save_steps"] = args.save_steps
    config["train"]["micro_batch_size"] = args.micro_batch_size
    config["train"]["global_batch_size"] = args.global_batch_size
    config["train"]["gradient_accumulation_steps"] = gradient_accumulation_steps
    align = config["train"]["align_params"]
    align["depth"]["moge_path"] = str(args.moge.resolve())
    align["depth"]["morgbd_path"] = str(BASE_WEIGHTS / "depth/model.pt")
    align["video"]["ckpt_path"] = str(BASE_WEIGHTS / "dino_video/teacher_step_10000.pth")
    align["video"]["config_path"] = str(BASE_WEIGHTS / "dino_video/config.yaml")
    for path in (Path(align["depth"]["morgbd_path"]), Path(align["video"]["ckpt_path"]),
                 Path(align["video"]["config_path"])):
        if not path.is_file():
            parser.error(f"teacher file missing: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    print(args.output)


if __name__ == "__main__":
    main()
