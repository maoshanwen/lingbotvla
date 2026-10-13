#!/usr/bin/env python3
"""Read-only motion-quality audit for RoboTwin LeRobot datasets.

The audit intentionally does not delete or rewrite any episode.  It reports
structural errors, robust velocity/acceleration/jerk outliers for both state
and action, and episodes that remain unchanged for more than 95% of their
transitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ARM_DIMS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
SIGNALS = ("state", "action")
DERIVATIVES = ("velocity", "acceleration", "jerk")
REVIEW_Z = 5.0
EXTREME_Z = 10.0
EPISODE_REVIEW_Z = 3.5
REVIEW_JOINT_STEP_RAD = 0.10
STATIC_THRESHOLD = 0.95
STATIC_EPS = 1e-6
NEAR_STATIC_EPS = 1e-4
HARD_JOINT_STEP_RAD = 0.5


def parse_train_list(path: Path) -> list[tuple[str, Path]]:
    datasets: list[tuple[str, Path]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        dataset = Path(line.split()[-1])
        task = dataset.name.split("-aloha-agilex", 1)[0]
        datasets.append((task, dataset))
    return datasets


def robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-dimension median and robust standard-deviation estimate."""
    median = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - median), axis=0)
    scale = 1.4826 * mad
    q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    std_scale = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale >= 1e-12), scale, iqr_scale)
    scale = np.where(np.isfinite(scale) & (scale >= 1e-12), scale, std_scale)
    scale = np.where(np.isfinite(scale) & (scale >= 1e-12), scale, 1.0)
    return median, scale


def derivatives(values: np.ndarray, timestamps: np.ndarray) -> dict[str, np.ndarray]:
    """Compute derivatives in physical time; values are joint coordinates."""
    values = values[:, ARM_DIMS].astype(np.float64, copy=False)
    dt = np.diff(timestamps)
    if len(values) < 2 or not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        return {name: np.empty((0, len(ARM_DIMS)), np.float64) for name in DERIVATIVES}
    velocity = np.diff(values, axis=0) / dt[:, None]
    if len(velocity) < 2:
        acceleration = np.empty((0, len(ARM_DIMS)), np.float64)
    else:
        acceleration_dt = (dt[1:] + dt[:-1]) / 2
        acceleration = np.diff(velocity, axis=0) / acceleration_dt[:, None]
    if len(acceleration) < 2:
        jerk = np.empty((0, len(ARM_DIMS)), np.float64)
    else:
        velocity_mid_dt = (dt[1:] + dt[:-1]) / 2
        jerk_dt = (velocity_mid_dt[1:] + velocity_mid_dt[:-1]) / 2
        jerk = np.diff(acceleration, axis=0) / jerk_dt[:, None]
    return {"velocity": velocity, "acceleration": acceleration, "jerk": jerk}


def read_task(task: str, dataset: Path) -> list[dict]:
    parquet_files = sorted(dataset.glob("data/chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet data under {dataset}")
    table = pq.read_table(
        parquet_files,
        columns=[
            "observation.state",
            "action",
            "episode_index",
            "frame_index",
            "timestamp",
        ],
    )
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    episode_indices = np.asarray(table["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(table["frame_index"], dtype=np.int64)
    timestamps = np.asarray(table["timestamp"], dtype=np.float64)

    episodes: list[dict] = []
    for episode_index in np.unique(episode_indices):
        mask = episode_indices == episode_index
        order = np.argsort(frame_indices[mask])
        episode = {
            "task": task,
            "dataset": str(dataset),
            "episode": int(episode_index),
            "state": states[mask][order],
            "action": actions[mask][order],
            "frame_index": frame_indices[mask][order],
            "timestamp": timestamps[mask][order],
        }
        episode["derivatives"] = {
            signal: derivatives(episode[signal], episode["timestamp"])
            for signal in SIGNALS
        }
        episodes.append(episode)
    return episodes


def max_detail(
    values: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    derivative_offset: int,
) -> dict:
    if not values.size:
        return {"max_z": 0.0, "p99_z": 0.0, "count_z5": 0, "count_z10": 0,
                "frame": -1, "joint_dim": -1, "raw": 0.0,
                "max_abs_raw": 0.0, "p99_abs_raw": 0.0}
    abs_z = np.abs((values - center) / scale)
    flat = int(np.nanargmax(abs_z))
    row, arm_position = np.unravel_index(flat, abs_z.shape)
    return {
        "max_z": float(abs_z[row, arm_position]),
        "p99_z": float(np.nanpercentile(abs_z, 99)),
        "count_z5": int(np.count_nonzero(abs_z >= REVIEW_Z)),
        "count_z10": int(np.count_nonzero(abs_z >= EXTREME_Z)),
        "frame": int(row + derivative_offset),
        "joint_dim": int(ARM_DIMS[arm_position]),
        "raw": float(values[row, arm_position]),
        "max_abs_raw": float(np.nanmax(np.abs(values))),
        "p99_abs_raw": float(np.nanpercentile(np.abs(values), 99)),
    }


def robust_z_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        q25, q75 = np.nanpercentile(values, [25, 75])
        scale = (q75 - q25) / 1.349
    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.nanstd(values)
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(values)
    return (values - median) / scale


def static_ratios(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return 1.0, 1.0
    transition = np.max(np.abs(np.diff(values.astype(np.float64), axis=0)), axis=1)
    return float(np.mean(transition <= STATIC_EPS)), float(np.mean(transition <= NEAR_STATIC_EPS))


def analyse_task(task: str, dataset: Path) -> list[dict]:
    episodes = read_task(task, dataset)
    baselines: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for signal in SIGNALS:
        for name in DERIVATIVES:
            parts = [ep["derivatives"][signal][name] for ep in episodes]
            nonempty = [part for part in parts if part.size]
            pooled = np.concatenate(nonempty, axis=0)
            baselines[(signal, name)] = robust_location_scale(pooled)

    rows: list[dict] = []
    derivative_offsets = {"velocity": 1, "acceleration": 2, "jerk": 3}
    for ep in episodes:
        frames = ep["frame_index"]
        timestamps = ep["timestamp"]
        frame_delta = np.diff(frames)
        timestamp_delta = np.diff(timestamps)
        median_dt = float(np.median(timestamp_delta)) if len(timestamp_delta) else math.nan
        finite_state = bool(np.isfinite(ep["state"]).all())
        finite_action = bool(np.isfinite(ep["action"]).all())
        frame_contiguous = bool(np.all(frame_delta == 1))
        timestamp_monotonic = bool(np.all(timestamp_delta > 0))
        timestamp_regular = bool(
            not len(timestamp_delta)
            or (
                timestamp_monotonic
                and np.max(np.abs(timestamp_delta - median_dt))
                <= max(1e-4, abs(median_dt) * 0.05)
            )
        )
        row: dict = {
            "task": task,
            "dataset": str(dataset),
            "episode": ep["episode"],
            "length": int(len(frames)),
            "finite_state": finite_state,
            "finite_action": finite_action,
            "frame_contiguous": frame_contiguous,
            "timestamp_monotonic": timestamp_monotonic,
            "timestamp_regular": timestamp_regular,
            "median_dt": median_dt,
        }

        worst = {"z": 0.0, "metric": "", "frame": -1, "joint_dim": -1, "raw": 0.0}
        for signal in SIGNALS:
            static_exact, static_near = static_ratios(ep[signal])
            row[f"{signal}_static_ratio_1e6"] = static_exact
            row[f"{signal}_static_ratio_1e4"] = static_near
            arm_step = np.diff(ep[signal][:, ARM_DIMS], axis=0)
            row[f"{signal}_max_joint_step"] = (
                float(np.max(np.abs(arm_step))) if arm_step.size else 0.0
            )
            for name in DERIVATIVES:
                center, scale = baselines[(signal, name)]
                detail = max_detail(
                    ep["derivatives"][signal][name],
                    center,
                    scale,
                    derivative_offsets[name],
                )
                prefix = f"{signal}_{name}"
                for key, value in detail.items():
                    row[f"{prefix}_{key}"] = value
                if detail["max_z"] > worst["z"]:
                    worst = {
                        "z": detail["max_z"],
                        "metric": prefix,
                        "frame": detail["frame"],
                        "joint_dim": detail["joint_dim"],
                        "raw": detail["raw"],
                    }

        structural_reasons = []
        if not finite_state:
            structural_reasons.append("non_finite_state")
        if not finite_action:
            structural_reasons.append("non_finite_action")
        if not frame_contiguous:
            structural_reasons.append("non_contiguous_frame_index")
        if not timestamp_monotonic:
            structural_reasons.append("non_monotonic_timestamp")
        if not timestamp_regular:
            structural_reasons.append("irregular_timestamp")

        static_remove = bool(
            row["state_static_ratio_1e6"] > STATIC_THRESHOLD
            and row["action_static_ratio_1e6"] > STATIC_THRESHOLD
        )
        row.update(
            {
                "point_worst_metric": worst["metric"],
                "point_worst_abs_robust_z": worst["z"],
                "point_worst_frame": worst["frame"],
                "point_worst_joint_dim": worst["joint_dim"],
                "point_worst_raw_derivative": worst["raw"],
                "structural_flag": bool(structural_reasons),
                "structural_reasons": ";".join(structural_reasons),
                "static_remove_flag": static_remove,
            }
        )
        rows.append(row)

    # A point-wise derivative Z-score is useful for locating a discontinuity,
    # but it is not by itself a cleaning criterion: normal contact and stopping
    # phases often have tiny task-level MAD and therefore enormous point Z.
    # Compare each episode's derivative peak and P99 against the other 44
    # demonstrations of the same task, then gate by a real joint displacement.
    episode_metrics = []
    for signal in SIGNALS:
        for name in DERIVATIVES:
            for statistic in ("max_abs_raw", "p99_abs_raw"):
                episode_metrics.append(f"{signal}_{name}_{statistic}")
    for metric in episode_metrics:
        values = np.array([row[metric] for row in rows], dtype=np.float64)
        zscores = robust_z_1d(np.log1p(values))
        for row, zscore in zip(rows, zscores):
            row[f"episode_z_{metric}"] = float(zscore)

    for row in rows:
        z_fields = [f"episode_z_{metric}" for metric in episode_metrics]
        worst_field = max(z_fields, key=lambda field: row[field])
        trajectory_z = max(0.0, float(row[worst_field]))
        max_step = max(row["state_max_joint_step"], row["action_max_joint_step"])
        smoothness_review = bool(
            (trajectory_z >= EPISODE_REVIEW_Z and max_step >= REVIEW_JOINT_STEP_RAD)
            or max_step >= 0.25
        )
        smoothness_priority_review = bool(
            trajectory_z >= EPISODE_REVIEW_Z and max_step >= 0.25
        )
        smoothness_remove = bool(max_step >= HARD_JOINT_STEP_RAD)
        remove_reasons = []
        if row["structural_flag"]:
            remove_reasons.extend(row["structural_reasons"].split(";"))
        if row["static_remove_flag"]:
            remove_reasons.append("state_and_action_static_gt_95pct")
        if smoothness_remove:
            remove_reasons.append("joint_step_ge_0.5_rad_per_frame")
        row.update(
            {
                "worst_metric": worst_field.removeprefix("episode_z_"),
                "worst_abs_robust_z": trajectory_z,
                "worst_frame": row["point_worst_frame"],
                "worst_joint_dim": row["point_worst_joint_dim"],
                "worst_raw_derivative": row["point_worst_raw_derivative"],
                "smoothness_review_flag": smoothness_review,
                "smoothness_priority_review_flag": smoothness_priority_review,
                "smoothness_remove_flag": smoothness_remove,
                "remove_flag": bool(remove_reasons),
                "remove_reasons": ";".join(remove_reasons),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_dict(row: dict) -> dict:
    keys = (
        "task", "dataset", "episode", "length", "worst_metric",
        "worst_abs_robust_z", "worst_frame", "worst_joint_dim",
        "worst_raw_derivative", "state_max_joint_step",
        "action_max_joint_step", "state_static_ratio_1e6",
        "action_static_ratio_1e6", "structural_reasons", "remove_reasons",
    )
    return {key: row[key] for key in keys}


def write_candidate_text(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(
                f"{row['task']} episode={row['episode']} "
                f"worst={row['worst_metric']} z={row['worst_abs_robust_z']:.3f} "
                f"frame={row['worst_frame']} joint={row['worst_joint_dim']} "
                f"state_step={row['state_max_joint_step']:.6f} "
                f"action_step={row['action_max_joint_step']:.6f} "
                f"state_static={row['state_static_ratio_1e6']:.4f} "
                f"action_static={row['action_static_ratio_1e6']:.4f} "
                f"reasons={row['remove_reasons'] or 'manual_review'}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    datasets = parse_train_list(args.train_list)
    rows: list[dict] = []
    for index, (task, dataset) in enumerate(datasets, 1):
        task_rows = analyse_task(task, dataset)
        rows.extend(task_rows)
        print(f"[{index:02d}/{len(datasets)}] {task}: {len(task_rows)} episodes", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "episode_motion_metrics.csv", rows)

    smoothness = sorted(
        (row for row in rows if row["smoothness_review_flag"]),
        key=lambda row: row["worst_abs_robust_z"], reverse=True,
    )
    static = sorted(
        (row for row in rows if row["static_remove_flag"]),
        key=lambda row: min(row["state_static_ratio_1e6"], row["action_static_ratio_1e6"]),
        reverse=True,
    )
    remove = sorted(
        (row for row in rows if row["remove_flag"]),
        key=lambda row: (not row["structural_flag"], -row["worst_abs_robust_z"]),
    )
    priority_review = sorted(
        (row for row in rows if row["smoothness_priority_review_flag"]),
        key=lambda row: row["state_max_joint_step"] if row["state_max_joint_step"] >= row["action_max_joint_step"]
        else row["action_max_joint_step"],
        reverse=True,
    )
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: {
        "episodes": 0, "smoothness_review": 0, "static_remove": 0, "remove": 0
    })
    for row in rows:
        item = by_task[row["task"]]
        item["episodes"] += 1
        item["smoothness_review"] += int(row["smoothness_review_flag"])
        item["static_remove"] += int(row["static_remove_flag"])
        item["remove"] += int(row["remove_flag"])

    summary = {
        "tasks": len(datasets),
        "episodes": len(rows),
        "finite_state_episodes": sum(row["finite_state"] for row in rows),
        "finite_action_episodes": sum(row["finite_action"] for row in rows),
        "frame_contiguous_episodes": sum(row["frame_contiguous"] for row in rows),
        "timestamp_monotonic_episodes": sum(row["timestamp_monotonic"] for row in rows),
        "timestamp_regular_episodes": sum(row["timestamp_regular"] for row in rows),
        "smoothness_review_count": len(smoothness),
        "smoothness_priority_review_count": len(priority_review),
        "smoothness_remove_count": sum(row["smoothness_remove_flag"] for row in rows),
        "static_remove_count": len(static),
        "structural_remove_count": sum(row["structural_flag"] for row in rows),
        "total_remove_count": len(remove),
        "thresholds": {
            "point_diagnostic_abs_robust_z": [REVIEW_Z, EXTREME_Z],
            "episode_peak_p99_review_abs_robust_z": EPISODE_REVIEW_Z,
            "smoothness_review_joint_step_rad_per_frame": REVIEW_JOINT_STEP_RAD,
            "smoothness_remove_joint_step_rad_per_frame": HARD_JOINT_STEP_RAD,
            "static_unchanged_epsilon": STATIC_EPS,
            "static_remove_ratio": STATIC_THRESHOLD,
            "static_requires": "both state and action exceed threshold",
            "timestamp_relative_tolerance": 0.05,
            "gripper_dims_excluded_from_smoothness": [6, 13],
        },
        "task_counts": dict(sorted(by_task.items())),
        "remove_candidates": [candidate_dict(row) for row in remove],
        "priority_review_candidates": [candidate_dict(row) for row in priority_review],
        "static_candidates": [candidate_dict(row) for row in static],
        "top_smoothness_candidates": [candidate_dict(row) for row in smoothness[:100]],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    write_candidate_text(args.output_dir / "remove_candidates.txt", remove)
    write_candidate_text(args.output_dir / "static_remove_candidates.txt", static)
    write_candidate_text(args.output_dir / "smoothness_review_candidates.txt", smoothness)
    write_candidate_text(args.output_dir / "priority_review_candidates.txt", priority_review)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"task_counts", "remove_candidates", "priority_review_candidates", "static_candidates",
                                     "top_smoothness_candidates"}}, indent=2))
    print("Top smoothness candidates:")
    for row in smoothness[:30]:
        print(
            f"  {row['task']} ep{row['episode']:02d} {row['worst_metric']} "
            f"z={row['worst_abs_robust_z']:.2f} frame={row['worst_frame']} "
            f"joint={row['worst_joint_dim']} state_step={row['state_max_joint_step']:.4f} "
            f"action_step={row['action_max_joint_step']:.4f}"
        )


if __name__ == "__main__":
    main()
