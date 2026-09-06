#!/usr/bin/env python
"""Turn nuScenes keyframes into Qwen-Drive planning scenes.

nuScenes keyframes are 2 Hz, which is exactly the 0.5 s camera cadence the model expects,
and ``ego_pose`` is recorded per sample_data (LIDAR_TOP runs at 20 Hz), so a real 10 Hz ego
history and a real 5 s future can both be interpolated from it. Output is the scene-file
format documented in docs/data.md, so scripts/demo.py and scripts/run_planning.py read it
without changes.

    python tools/nuscenes_to_scenes.py --root data/nuscenes --output data/nuscenes_scenes.jsonl

nuScenes has no route command, so nav_command is inferred from the recorded future; see
--turn-threshold.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CAMS = {"<FRONT VIEW>": "CAM_FRONT", "<FRONT LEFT VIEW>": "CAM_FRONT_LEFT",
        "<FRONT RIGHT VIEW>": "CAM_FRONT_RIGHT"}
VIEW_ORDER = list(CAMS)
NUM_HISTORY, NUM_FUTURE, HZ = 16, 50, 10.0
FRAME_OFFSETS = (-1.5, -1.0, -0.5, 0.0)
# The pixel budgets the processor applies: history frames near 320p, current near 720p.
CURRENT_PIXELS, HISTORY_PIXELS = 921600, 174080


def load(root: Path, name: str) -> list[dict]:
    return json.loads((root / "v1.0-mini" / f"{name}.json").read_text())


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_yaw(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def unwrap_to(reference: np.ndarray) -> np.ndarray:
    return np.unwrap(reference)


def resized_size(width: int, height: int, budget: int) -> tuple[int, int]:
    scale = (budget / (width * height)) ** 0.5
    return max(1, round(width * scale)), max(1, round(height * scale))


class Mini:
    """The bits of the nuScenes tables this conversion needs."""

    def __init__(self, root: Path):
        self.root = root
        self.scenes = load(root, "scene")
        self.samples = {s["token"]: s for s in load(root, "sample")}
        self.ego_pose = {e["token"]: e for e in load(root, "ego_pose")}
        self.sample_data = load(root, "sample_data")

        self.keyframe_images: dict[tuple[str, str], dict] = {}
        by_scene_lidar: dict[str, list[dict]] = defaultdict(list)
        for sd in self.sample_data:
            channel = sd["filename"].split("/")[1] if "/" in sd["filename"] else ""
            if sd["is_key_frame"] and channel in CAMS.values():
                self.keyframe_images[(sd["sample_token"], channel)] = sd
            if channel == "LIDAR_TOP":
                by_scene_lidar[self.samples[sd["sample_token"]]["scene_token"]].append(sd)

        # A dense pose timeline per scene, from the 20 Hz lidar stream.
        self.timeline: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for scene_token, entries in by_scene_lidar.items():
            entries.sort(key=lambda e: e["timestamp"])
            poses = [self.ego_pose[e["ego_pose_token"]] for e in entries]
            times = np.array([p["timestamp"] for p in poses], dtype=np.float64) * 1e-6
            xy = np.array([p["translation"][:2] for p in poses], dtype=np.float64)
            yaw = np.unwrap(np.array([quat_to_yaw(p["rotation"]) for p in poses]))
            self.timeline[scene_token] = (times, xy, yaw)

    def sample_chain(self, scene: dict) -> list[dict]:
        chain, token = [], scene["first_sample_token"]
        while token:
            sample = self.samples[token]
            chain.append(sample)
            token = sample["next"]
        return chain

    def poses_at(self, scene_token: str, targets: np.ndarray):
        """Interpolate (x, y, yaw) in the global frame at the given absolute times."""
        times, xy, yaw = self.timeline[scene_token]
        if targets.min() < times[0] or targets.max() > times[-1]:
            return None
        return (
            np.stack([np.interp(targets, times, xy[:, 0]), np.interp(targets, times, xy[:, 1])], 1),
            np.interp(targets, times, yaw),
        )


def to_ego_frame(xy: np.ndarray, yaw: np.ndarray, origin_xy, origin_yaw) -> np.ndarray:
    """Global poses expressed in the ego frame of the current timestamp."""
    rotation = np.array([[np.cos(origin_yaw), np.sin(origin_yaw)],
                         [-np.sin(origin_yaw), np.cos(origin_yaw)]])
    local = (xy - origin_xy) @ rotation.T
    heading = np.arctan2(np.sin(yaw - origin_yaw), np.cos(yaw - origin_yaw))
    return np.concatenate([local, heading[:, None]], axis=1)


def build_record(mini: Mini, sample: dict, chain_index: int, chain: list[dict],
                 turn_threshold: float) -> dict | None:
    scene_token = sample["scene_token"]
    if chain_index < 3:
        return None                                   # needs 1.5 s of camera history
    history_samples = chain[chain_index - 3 : chain_index + 1]

    current_time = sample["timestamp"] * 1e-6
    # One extra point either side so the finite differences are not one-sided.
    grid = current_time + np.arange(-(NUM_HISTORY), NUM_FUTURE + 2) / HZ
    interpolated = mini.poses_at(scene_token, grid)
    if interpolated is None:
        return None                                   # not enough recorded time around it
    xy, yaw = interpolated
    origin_index = NUM_HISTORY                        # grid[NUM_HISTORY] == current_time
    poses = to_ego_frame(xy, yaw, xy[origin_index], yaw[origin_index])

    velocity = np.gradient(poses[:, :2], 1.0 / HZ, axis=0)
    acceleration = np.gradient(velocity, 1.0 / HZ, axis=0)

    history = poses[origin_index - NUM_HISTORY + 1 : origin_index + 1]
    future = poses[origin_index + 1 : origin_index + 1 + NUM_FUTURE]
    if len(future) < NUM_FUTURE:
        return None
    hist_vel = velocity[origin_index - NUM_HISTORY + 1 : origin_index + 1]
    hist_acc = acceleration[origin_index - NUM_HISTORY + 1 : origin_index + 1]

    # nuScenes carries no route command; read one off the recorded future.
    turn = float(future[-1, 2])
    nav = 1 if turn > turn_threshold else 2 if turn < -turn_threshold else 0
    driving_command = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]][nav]

    content = []
    for view in VIEW_ORDER:
        content.append({"text": view})
        for index, (offset, hist_sample) in enumerate(zip(FRAME_OFFSETS, history_samples)):
            sd = mini.keyframe_images[(hist_sample["token"], CAMS[view])]
            budget = CURRENT_PIXELS if index == len(FRAME_OFFSETS) - 1 else HISTORY_PIXELS
            rw, rh = resized_size(sd["width"], sd["height"], budget)
            content.append({"text": f"frame: {index}"})
            content.append({
                "image": sd["filename"], "width": sd["width"], "height": sd["height"],
                "timestamp_idx": index, "timestamp_offset_sec": offset,
                "resized_width": rw, "resized_height": rh,
            })

    trajectory = {
        "hist_traj_10hz": np.round(history, 4).tolist(),
        "hist_vel_10hz": np.round(hist_vel, 4).tolist(),
        "hist_acc_10hz": np.round(hist_acc, 4).tolist(),
        "future_traj_10hz": np.round(future, 4).tolist(),
        "future_valid_mask_10hz": [1.0] * NUM_FUTURE,
        "ego_status": {
            "ego_velocity": np.round(hist_vel[-1], 4).tolist(),
            "ego_acceleration": np.round(hist_acc[-1], 4).tolist(),
            "driving_command": driving_command,
        },
        "nav_command": nav,
    }
    return {
        "type": "chatml",
        "messages": [{"role": "user", "content": content}],
        "trajectory": trajectory,
        "meta_info": {"token": sample["token"], "scene_token": scene_token},
    }


def instruction_for(record: dict, root: Path) -> str:
    """Let DrivingScene synthesize the prompt, so the wording matches the released files."""
    from qwen_drive import CameraFrame, DrivingScene

    content = record["messages"][0]["content"]
    images = [item for item in content if "image" in item]
    per_view = len(images) // len(VIEW_ORDER)
    views = {
        view: [CameraFrame(root / images[i * per_view + k]["image"]) for k in range(per_view)]
        for i, view in enumerate(VIEW_ORDER)
    }
    trajectory = record["trajectory"]
    scene = DrivingScene(
        views=views,
        history=np.asarray(trajectory["hist_traj_10hz"]),
        history_velocity=np.asarray(trajectory["hist_vel_10hz"]),
        history_acceleration=np.asarray(trajectory["hist_acc_10hz"]),
        ego_velocity=trajectory["ego_status"]["ego_velocity"],
        ego_acceleration=trajectory["ego_status"]["ego_acceleration"],
        driving_command=trajectory["ego_status"]["driving_command"],
        nav_command=trajectory["nav_command"],
    )
    return scene.instruction()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--output", type=Path, default=Path("data/nuscenes_scenes.jsonl"))
    parser.add_argument("--turn-threshold", type=float, default=0.35,
                        help="radians of heading change over 5 s that counts as a turn")
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth usable sample")
    args = parser.parse_args()

    mini = Mini(args.root)
    records, counts = [], defaultdict(int)
    for scene in mini.scenes:
        chain = mini.sample_chain(scene)
        for index, sample in enumerate(chain):
            record = build_record(mini, sample, index, chain, args.turn_threshold)
            if record is None:
                continue
            record["messages"][0]["content"].append({"text": instruction_for(record, args.root)})
            records.append(record)
            counts[record["trajectory"]["nav_command"]] += 1

    records = records[:: args.stride]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    names = {0: "straight", 1: "left", 2: "right"}
    spread = ", ".join(f"{names[k]} {v}" for k, v in sorted(counts.items()))
    print(f"wrote {len(records)} scenes to {args.output}  ({spread}, before stride)")


if __name__ == "__main__":
    main()
