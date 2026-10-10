#!/usr/bin/env python3
"""Merge one RoboTwin setting into a method-level experiment result JSON."""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


RATE_RE = re.compile(r"Success rate:\s*(?:\x1b\[[0-9;]*m)?(\d+)/(\d+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--setting", choices=("clean", "randomized"), required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    return parser.parse_args()


def read_task_results(log_dir: Path):
    tasks = {}
    for log_path in sorted(log_dir.glob("*.log")):
        text = ANSI_RE.sub("", log_path.read_text(encoding="utf-8", errors="ignore"))
        matches = RATE_RE.findall(text)
        success, attempts = (map(int, matches[-1]) if matches else (0, 0))
        tasks[log_path.stem] = {
            "attempts": attempts,
            "successes": success,
            "success_rate": round(success / attempts, 6) if attempts else None,
            "complete": attempts > 0,
            "log": str(log_path),
        }
    return tasks


def main():
    args = parse_args()
    result_path = Path(args.result_json)
    run_dir = Path(args.run_dir)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "schema": "team_experiment_v1",
            "method_name": args.method_name,
            "model_path": args.model_path,
            "settings": {},
        }

    tasks = read_task_results(run_dir / "eval_logs")
    total_attempts = sum(item["attempts"] for item in tasks.values())
    total_successes = sum(item["successes"] for item in tasks.values())
    payload["method_name"] = args.method_name
    payload["model_path"] = args.model_path
    payload["updated_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    payload["settings"][args.setting] = {
        "run_dir": str(run_dir),
        "expected_tasks": args.expected_tasks,
        "expected_episodes_per_task": args.expected_episodes,
        "completed_tasks": sum(
            item["attempts"] == args.expected_episodes for item in tasks.values()
        ),
        "total_attempts": total_attempts,
        "total_successes": total_successes,
        "overall_success_rate": (
            round(total_successes / total_attempts, 6) if total_attempts else None
        ),
        "tasks": tasks,
    }

    temp_path = result_path.with_suffix(result_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temp_path, result_path)
    print(result_path)


if __name__ == "__main__":
    main()
