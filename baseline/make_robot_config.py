#!/usr/bin/env python3
"""Copy the RoboTwin feature map with clean-only normalization statistics."""

from pathlib import Path

import yaml


root = Path(__file__).resolve().parent.parent
baseline = Path(__file__).resolve().parent
source = root / "lingbot-vla-v2/configs/robot_configs/robotwin.yaml"
target = baseline / "work/robot_configs/robotwin.yaml"
config = yaml.safe_load(source.read_text())
config["norm_stats"] = str(baseline / "work/clean_norm_stats.json")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
print(target)
