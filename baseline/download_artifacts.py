#!/usr/bin/env python3
"""Download pinned competition training artifacts and RoboTwin sim assets."""

import argparse
import os
from pathlib import Path
import zipfile

from huggingface_hub import hf_hub_download, snapshot_download

from prepare_data import inspect_archive, task_names


ROOT = Path(__file__).resolve().parent.parent
MODEL_REV = os.environ.get("LINGBOT_MODEL_REV", "11c703bf6a5c1f45b3b69168482da11fdbba53d7")
DATA_REV = os.environ.get("ROBOTWIN_DATA_REV", "3dc3b798668feb99ac61cc9086d84cbcc3d79186")
QWEN_REV = os.environ.get("QWEN_REV", "ebb281ec70b05090aa6165b016eac8ec08e71b17")
MOGE_REV = os.environ.get("MOGE_REV", "ca5f0e07ff01d3e5a364c1d954ed12ee1814b368")


def download_models() -> None:
    entries = (
        ("robbyant/lingbot-vla-v2-6b", MODEL_REV, ROOT / "lingbot-vla-v2-6b"),
        ("Qwen/Qwen3-VL-4B-Instruct", QWEN_REV, ROOT / "baseline/models/Qwen3-VL-4B-Instruct"),
        ("Ruicheng/moge-2-vitb-normal", MOGE_REV, ROOT / "baseline/models/moge-2-vitb-normal"),
    )
    for repo, revision, destination in entries:
        print(f"Downloading {repo}@{revision} -> {destination}", flush=True)
        snapshot_download(repo_id=repo, revision=revision, local_dir=destination)
    for path in (
        ROOT / "lingbot-vla-v2-6b/model.safetensors.index.json",
        ROOT / "lingbot-vla-v2-6b/depth/model.pt",
        ROOT / "lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth",
        ROOT / "baseline/models/Qwen3-VL-4B-Instruct/config.json",
        ROOT / "baseline/models/moge-2-vitb-normal/model.pt",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def download_data() -> None:
    destination = ROOT / "灵波2026/robotwin_official"
    for index, task in enumerate(task_names(), 1):
        filename = f"dataset/{task}/aloha-agilex_clean_50.zip"
        print(f"[{index}/50] {filename}", flush=True)
        hf_hub_download(repo_id="TianxingChen/RoboTwin2.0", repo_type="dataset",
                        revision=DATA_REV, filename=filename, local_dir=destination)
        inspect_archive(task)
    print("Verified all 50 clean archives", flush=True)


def extract_asset(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"unsafe zip entry: {member.filename}")
        source.extractall(destination)


def download_assets() -> None:
    destination = ROOT / "RoboTwin/assets"
    destination.mkdir(parents=True, exist_ok=True)
    names = ("background_texture", "embodiments", "objects")
    for name in names:
        archive = Path(hf_hub_download(
            repo_id="TianxingChen/RoboTwin2.0", repo_type="dataset",
            revision=DATA_REV, filename=f"{name}.zip", local_dir=destination,
        ))
        marker = destination / f".{name}.bootstrap_complete"
        if not (destination / name).is_dir() or not marker.is_file() or marker.read_text().strip() != DATA_REV:
            print(f"Extracting {archive}", flush=True)
            extract_asset(archive, destination)
            marker.write_text(DATA_REV + "\n")
    for name in names:
        if not (destination / name).is_dir():
            raise FileNotFoundError(destination / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "models", "data", "assets"), default="all")
    args = parser.parse_args()
    if args.only in ("all", "models"):
        download_models()
    if args.only in ("all", "data"):
        download_data()
    if args.only in ("all", "assets"):
        download_assets()


if __name__ == "__main__":
    main()
