#!/usr/bin/env python3
"""Convert complete RoboTwin simulation logs to the competition JSON schema."""

import argparse
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "灵波2026/初赛评测结果模板_JSON版.json"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RATE = re.compile(r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>")


def collect(run_dir, tasks, episodes=100):
    run_dir = Path(run_dir)
    stats_file = run_dir / "stats.txt"
    if not stats_file.is_file():
        raise ValueError(f"missing completed stats.txt: {stats_file}")
    stats = stats_file.read_text(errors="replace")
    if f"All tasks fully executed {episodes} episodes" not in stats:
        raise ValueError(f"run has skipped or incomplete tasks: {stats_file}")
    result = {}
    for task in tasks:
        log = run_dir / "eval_logs" / f"{task}.log"
        if not log.is_file():
            raise ValueError(f"missing task log: {log}")
        clean_text = ANSI.sub("", log.read_text(errors="replace"))
        matches = list(RATE.finditer(clean_text))
        if not matches:
            raise ValueError(f"no success count in: {log}")
        successes, attempts = map(int, matches[-1].groups())
        if attempts != episodes or not 0 <= successes <= episodes:
            raise ValueError(f"invalid or incomplete result in {log}: {successes}/{attempts}")
        result[task] = {"attempts": attempts, "successes": successes}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--randomized-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", help="comma-separated subset; defaults to all official tasks")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 100:
        parser.error("--episodes must be from 1 to 100")
    template = json.loads(TEMPLATE.read_text())
    tasks = list(template["results"]["clean"])
    if list(template["results"]["randomized"]) != tasks:
        parser.error("competition template task lists differ")
    all_tasks = tasks
    if args.tasks is not None and args.tasks != "all":
        tasks = [task.strip() for task in args.tasks.split(",")]
        if not tasks or len(set(tasks)) != len(tasks) or any(task not in all_tasks for task in tasks):
            parser.error("--tasks contains an unknown, empty, or duplicate task")
    if not args.team_id.strip() or args.team_id == "replace_with_your_team_id":
        parser.error("provide the real team ID")
    payload = {
        "schema_version": template["schema_version"],
        "team_id": args.team_id,
        "results": {
            "clean": collect(args.clean_run, tasks, args.episodes),
            "randomized": collect(args.randomized_run, tasks, args.episodes),
        },
    }
    if len(tasks) != len(all_tasks) or args.episodes != 100:
        payload["evaluation"] = {
            "scope": "subset", "tasks": tasks, "episodes_per_task": args.episodes,
            "episode_selection": "first_valid_episodes", "seed": 0,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.output} with {len(tasks)} complete tasks per setting, {args.episodes} episodes per task")


if __name__ == "__main__":
    main()
