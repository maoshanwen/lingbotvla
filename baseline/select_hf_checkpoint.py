#!/usr/bin/env python3
"""Print the complete exported HF checkpoint at the requested training step."""

import argparse
import json
from pathlib import Path


def select(output_dir: Path, step: int) -> Path:
    if step < 1:
        raise ValueError("step must be positive")
    if not (output_dir / "lingbotvla_cli.yaml").is_file():
        raise FileNotFoundError(f"missing training configuration: {output_dir / 'lingbotvla_cli.yaml'}")
    checkpoint = output_dir / "checkpoints" / f"global_step_{step}" / "hf_ckpt"
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"incomplete exported checkpoint, missing {name}: {checkpoint}")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shards = set(index.get("weight_map", {}).values())
    if not shards or any(not name.endswith(".safetensors") or Path(name).name != name for name in shards):
        raise ValueError(f"invalid weight map: {checkpoint}")
    missing = [name for name in sorted(shards) if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"missing or empty weight shards in {checkpoint}: {missing}")
    return checkpoint.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    print(select(args.output_dir, args.step))


if __name__ == "__main__":
    main()
