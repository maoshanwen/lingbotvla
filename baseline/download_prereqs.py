#!/usr/bin/env python3
"""Download the Qwen3-VL backbone and MoGe-2 teacher used by official LingBot training."""

import os
from pathlib import Path

models = Path(__file__).resolve().parent / "models"
os.environ.setdefault("HF_HOME", str(models.parent / "work/hf_cache"))

from huggingface_hub import snapshot_download


models.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id="Qwen/Qwen3-VL-4B-Instruct",
    local_dir=models / "Qwen3-VL-4B-Instruct",
)
snapshot_download(
    repo_id="Ruicheng/moge-2-vitb-normal",
    local_dir=models / "moge-2-vitb-normal",
)
print(f"Downloaded prerequisites to {models}")
