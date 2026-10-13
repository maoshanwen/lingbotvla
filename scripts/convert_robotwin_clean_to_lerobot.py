#!/usr/bin/env python3
"""Convert one official RoboTwin clean ZIP into LeRobot v3 train/val datasets."""

import argparse
import io
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


CAMERAS = {
    "observation.images.cam_high": "head_camera",
    "observation.images.cam_left_wrist": "left_camera",
    "observation.images.cam_right_wrist": "right_camera",
}


def decode_rgb(value: np.ndarray) -> np.ndarray:
    """Decode RoboTwin image bits to an RGB ndarray.

    RoboTwin clean_50 archives currently use the legacy encoding: an RGB
    ndarray was passed directly to cv2.imencode, so cv2.imdecode returns the
    original RGB channel order numerically even though OpenCV calls that array
    BGR.  Newer XPolicyLab payloads are standard JPEGs and carry the XPL-RGB1
    marker; those need the normal BGR-to-RGB conversion.

    Do not blindly call cvtColor here.  Doing so swaps red and blue for every
    legacy clean_50 frame (yellow becomes cyan).
    """
    encoded = bytes(value)
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV failed to decode a camera frame")
    if b"XPL-RGB1" in encoded:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def features(height: int, width: int) -> dict:
    joint_names = [
        "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
        "left_wrist_angle", "left_wrist_rotate", "left_gripper",
        "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
        "right_wrist_angle", "right_wrist_rotate", "right_gripper",
    ]
    result = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": joint_names},
        "action": {"dtype": "float32", "shape": (14,), "names": joint_names},
    }
    for key in CAMERAS:
        result[key] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return result


def instruction_for_episode(zf: zipfile.ZipFile, prefix: str, episode: int) -> str:
    path = f"{prefix}/instructions/episode{episode}.json"
    payload = json.loads(zf.read(path))
    candidates = payload.get("seen", []) + payload.get("unseen", [])
    if not candidates:
        raise ValueError(f"No instruction candidates in {path}")
    return str(candidates[episode % len(candidates)])


def build_dataset(root: Path, repo_name: str, image_shape: tuple[int, int]) -> LeRobotDataset:
    if root.exists():
        shutil.rmtree(root)
    return LeRobotDataset.create(
        repo_id=f"local/{repo_name}",
        root=root,
        fps=15,
        robot_type="aloha",
        features=features(*image_shape),
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=3,
        batch_encoding_size=1,
    )


def convert(zip_path: Path, output_root: Path, val_episodes: int, limit_episodes: int | None) -> None:
    task_name = zip_path.parent.name
    train_root = output_root / f"{task_name}-aloha-agilex_clean_50-train"
    val_root = output_root / f"{task_name}-aloha-agilex_clean_50-val"
    complete = output_root / f".{task_name}.complete"
    if complete.exists() and train_root.exists() and val_root.exists():
        print(f"SKIP {task_name}: already complete", flush=True)
        return

    with zipfile.ZipFile(zip_path) as zf:
        episode_paths = sorted(
            (name for name in zf.namelist() if name.endswith(".hdf5") and "/data/episode" in name),
            key=lambda name: int(Path(name).stem.removeprefix("episode")),
        )
        if len(episode_paths) != 50:
            raise ValueError(f"{task_name}: expected 50 episodes, found {len(episode_paths)}")
        if limit_episodes is not None:
            episode_paths = episode_paths[:limit_episodes]
        if not 0 < val_episodes < len(episode_paths):
            raise ValueError("val_episodes must be positive and smaller than the episode count")
        prefix = episode_paths[0].split("/data/", 1)[0]

        with h5py.File(io.BytesIO(zf.read(episode_paths[0])), "r") as h5:
            sample = decode_rgb(h5["observation/head_camera/rgb"][0])
            image_shape = (int(sample.shape[0]), int(sample.shape[1]))

        train_ds = build_dataset(train_root, train_root.name, image_shape)
        val_ds = build_dataset(val_root, val_root.name, image_shape)
        split_at = len(episode_paths) - val_episodes

        for episode_idx, episode_path in enumerate(episode_paths):
            dataset = train_ds if episode_idx < split_at else val_ds
            instruction = instruction_for_episode(zf, prefix, episode_idx)
            with h5py.File(io.BytesIO(zf.read(episode_path)), "r") as h5:
                joints = np.asarray(h5["joint_action/vector"], dtype=np.float32)
                if len(joints) < 2:
                    raise ValueError(f"{task_name} episode {episode_idx}: fewer than two frames")
                camera_data = {
                    out_key: h5[f"observation/{raw_name}/rgb"]
                    for out_key, raw_name in CAMERAS.items()
                }
                for frame_idx in range(len(joints) - 1):
                    frame = {
                        "observation.state": joints[frame_idx],
                        "action": joints[frame_idx + 1],
                        "task": instruction,
                    }
                    for out_key, values in camera_data.items():
                        frame[out_key] = decode_rgb(values[frame_idx])
                    dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=True)
            print(f"{task_name} episode {episode_idx + 1}/{len(episode_paths)}", flush=True)

        train_ds.finalize()
        val_ds.finalize()

    complete.write_text("ok\n")
    print(f"DONE {task_name} train={len(episode_paths) - val_episodes} val={val_episodes}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--val-episodes", type=int, default=5)
    parser.add_argument("--limit-episodes", type=int)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    convert(args.zip_path, args.output_root, args.val_episodes, args.limit_episodes)


if __name__ == "__main__":
    main()
