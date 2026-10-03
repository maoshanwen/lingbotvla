#!/usr/bin/env python3
"""Run an official LingBot entry point with faster LeRobot parquet loading."""

import os
import runpy
import sys
from pathlib import Path

import torch.distributed as dist


def speed_up_lerobot_loading(norm_only: bool) -> None:
    """Parallelize parquet conversion; skip image bytes when computing stats."""
    import pyarrow.parquet as parquet
    from lerobot.common.datasets import lerobot_dataset

    original_load_dataset = lerobot_dataset.load_dataset
    workers = max(1, int(os.environ.get("PARQUET_WORKERS", "4")))

    def load_dataset(*args, **kwargs):
        if args and args[0] == "parquet":
            if norm_only:
                data_files = kwargs.get("data_files")
                first_file = Path(data_files[0]) if data_files else next(
                    Path(kwargs["data_dir"]).glob("chunk-*/*.parquet")
                )
                kwargs["columns"] = [
                    field.name for field in parquet.read_schema(first_file)
                    if not field.name.startswith("observation.images.")
                ]
            kwargs["num_proc"] = workers
        return original_load_dataset(*args, **kwargs)

    lerobot_dataset.load_dataset = load_dataset


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_torch_entry.py <official_script.py> [args...]")
    entry = Path(sys.argv[1]).resolve()
    if not entry.is_file():
        raise FileNotFoundError(entry)

    # This H100 worker hangs while tearing down its NCCL communicator.
    if os.environ.get("NCCL_COMPAT") == "1":
        dist.destroy_process_group = lambda *args, **kwargs: None
    speed_up_lerobot_loading(entry.name == "compute_norm_stats.py")
    sys.argv = [str(entry), *sys.argv[2:]]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
