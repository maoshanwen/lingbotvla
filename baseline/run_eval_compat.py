#!/usr/bin/env python3
"""Run full evaluation with reversible workarounds for this H100 host."""

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ROBOTWIN = ROOT / "RoboTwin"
CLIENT = ROBOTWIN / "script/eval_policy_client_lingbotvla.py"
CLIENT_SOURCE = ROOT / "lingbot-vla-v2/experiment/robotwin/eval_policy_client_lingbotvla.py"
CONFIG_DIR = ROBOTWIN / "envs/curobo/src/curobo/content/configs/task"
CONFIGS = [
    CONFIG_DIR / name
    for name in (
        "gradient_ik_autotune.yml",
        "gradient_trajopt.yml",
        "finetune_trajopt.yml",
        "finetune_trajopt_slow.yml",
    )
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    if not CLIENT.is_file():
        shutil.copyfile(CLIENT_SOURCE, CLIENT)
        remove_client = True
    else:
        remove_client = False
    originals = {path: path.read_bytes() for path in [CLIENT, *CONFIGS]}
    try:
        for path in CONFIGS:
            content = originals[path].decode()
            if content.lower().count("use_cuda_kernel: true") != 1:
                raise ValueError(f"unexpected cuRobo config: {path}")
            path.write_text(content.replace("use_cuda_kernel: True", "use_cuda_kernel: False")
                            .replace("use_cuda_kernel: true", "use_cuda_kernel: false"))

        source = originals[CLIENT].decode()
        if not source.endswith("    main(usr_args)\n"):
            raise ValueError(f"unexpected evaluation client ending: {CLIENT}")
        CLIENT.write_text(source + "    sys.stdout.flush()\n    sys.stderr.flush()\n    os._exit(0)\n")
        print("Using reversible H100 cuRobo/shutdown workarounds; benchmark success logic is unchanged.", flush=True)
        subprocess.run(["bash", str(ROOT / "baseline/eval.sh"), str(args.checkpoint)],
                       cwd=ROOT, check=True)
    finally:
        for path, content in originals.items():
            if path == CLIENT and remove_client:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)


if __name__ == "__main__":
    main()
