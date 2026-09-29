#!/usr/bin/env python3
"""Prepare only the competition's 50 clean RoboTwin demonstrations."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "灵波2026/初赛评测结果模板_JSON版.json"
ARCHIVES = ROOT / "灵波2026/robotwin_official/dataset"
ROBOTWIN = ROOT / "RoboTwin"
WORK = Path(__file__).resolve().parent / "work"
SETTING = "aloha-agilex_clean_50"


def task_names():
    return list(json.loads(TEMPLATE.read_text())["results"]["clean"])


def archive_for(task):
    return ARCHIVES / task / f"{SETTING}.zip"


def expected_members():
    return {
        f"{SETTING}/{kind}/episode{i}.{suffix}"
        for kind, suffix in (("data", "hdf5"), ("instructions", "json"))
        for i in range(50)
    }


def inspect_archive(task):
    path = archive_for(task)
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = expected_members() - names
        if missing:
            raise ValueError(f"{path}: missing {len(missing)} expected members, e.g. {sorted(missing)[0]}")
        if any("randomized" in name.lower() for name in names):
            raise ValueError(f"{path}: randomized content is forbidden in training")
    return path


def unpack(task):
    path = inspect_archive(task)
    destination = ROBOTWIN / "data" / task
    with zipfile.ZipFile(path) as archive:
        for member in sorted(expected_members()):
            target = destination / member
            if target.exists() and target.stat().st_size == archive.getinfo(member).file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".partial")
            with archive.open(member) as source, temporary.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(target)
    print(f"unpacked {task}", flush=True)


def processed_dir(task):
    return ROBOTWIN / "policy/pi0/processed_data" / f"{task}-{SETTING}-50"


def is_processed(task):
    directory = processed_dir(task)
    return all((directory / f"episode_{i}" / f"episode_{i}.hdf5").is_file()
               and (directory / f"episode_{i}" / "instructions.json").is_file()
               for i in range(50))


def process(task, python):
    if not is_processed(task):
        subprocess.run([python, "scripts/process_data.py", task, SETTING, "50"],
                       cwd=ROBOTWIN / "policy/pi0", check=True)
    if not is_processed(task):
        raise RuntimeError(f"RoboTwin processing incomplete for {task}")
    print(f"processed {task}", flush=True)


def dataset_dir(task):
    return WORK / "cache/huggingface/lerobot/robotwin_clean" / task


def is_converted(task):
    info = dataset_dir(task) / "meta/info.json"
    if not info.is_file():
        return False
    try:
        return json.loads(info.read_text()).get("total_episodes") == 50
    except (ValueError, OSError):
        return False


def convert(task, python=None):
    if not is_converted(task):
        env = os.environ.copy()
        env["XDG_CACHE_HOME"] = str(WORK / "cache")
        env["HF_HOME"] = str(WORK / "cache/huggingface")
        env["HF_LEROBOT_HOME"] = str(WORK / "cache/huggingface/lerobot")
        env.setdefault("UV_CACHE_DIR", str(WORK / "uv_cache"))
        env.setdefault("UV_LINK_MODE", "copy")
        command = ([python] if python else ["uv", "run"])
        subprocess.run(command + [
            "examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py",
            "--raw_dir", str(processed_dir(task)),
            "--repo_id", f"robotwin_clean/{task}",
        ], cwd=ROBOTWIN / "policy/pi0", env=env, check=True)
    if not is_converted(task):
        raise RuntimeError(f"LeRobot conversion incomplete for {task}: {dataset_dir(task)}")
    print(f"converted {task}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["verify", "unpack", "process", "hdf5", "convert", "all"],
                        help="hdf5 runs unpack+process; all also converts to LeRobot")
    parser.add_argument("--tasks", nargs="+", help="Subset for a smoke test; full training needs all 50")
    parser.add_argument("--process-python", default=sys.executable,
                        help="Python in the RoboTwin simulation environment (needs h5py and cv2)")
    parser.add_argument("--convert-python", default=None,
                        help="Python with LeRobot and tyro installed; default is uv run in RoboTwin/policy/pi0")
    args = parser.parse_args()
    process_python = (os.path.abspath(args.process_python)
                      if os.path.sep in args.process_python else args.process_python)
    convert_python = (os.path.abspath(args.convert_python)
                      if args.convert_python and os.path.sep in args.convert_python else args.convert_python)
    expected = task_names()
    tasks = args.tasks or expected
    unknown = sorted(set(tasks) - set(expected))
    if unknown or len(tasks) != len(set(tasks)):
        parser.error(f"unknown or duplicate task names: {unknown or tasks}")
    for index, task in enumerate(tasks, 1):
        inspect_archive(task)
        print(f"[{index}/{len(tasks)}] {task}: clean archive OK", flush=True)
        if args.stage in ("unpack", "hdf5", "all"):
            unpack(task)
        if args.stage in ("process", "hdf5", "all"):
            process(task, process_python)
        if args.stage in ("convert", "all"):
            convert(task, convert_python)
    if args.stage in ("convert", "all"):
        if all(is_converted(task) for task in expected):
            WORK.mkdir(parents=True, exist_ok=True)
            manifest = WORK / "clean_datasets.txt"
            manifest.write_text("".join(f"robotwin {dataset_dir(task)}\n" for task in expected))
            print(f"50-task clean manifest: {manifest}")
        else:
            print("Subset prepared; full 50-task training manifest not written")


if __name__ == "__main__":
    main()
