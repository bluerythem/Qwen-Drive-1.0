#!/usr/bin/env python
"""Pack nuScenes keyframes into the perception frame layout, for the same samples that
tools/nuscenes_to_scenes.py turned into planning scenes.

That lets both heads run on one keyframe: the trajectory and the predicted BEV map come
from the same moment, so the plan can be drawn on the map.

    python tools/nuscenes_perception_frames.py --scenes data/nuscenes_scenes.jsonl \
        --root data/nuscenes --output data/nuscenes_perception

Images are symlinked, not copied. `gt.npz` is written empty: nuScenes ships no occupancy or
map rasters (those need Occ3D and the map expansion), and inference never reads it - only
the visualizer's ground-truth panels would, and those are not used here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
VIEW_LABELS = ["<FRONT VIEW>", "<FRONT RIGHT VIEW>", "<BACK RIGHT VIEW>",
               "<BACK VIEW>", "<BACK LEFT VIEW>", "<FRONT LEFT VIEW>"]
PROMPT = "Analyze the scene."
OCC_SHAPE, MAP_SHAPE = (200, 200, 16), (200, 400)
OCC_EMPTY = 9  # the 'empty' class index


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def transform(rotation, translation) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = quat_to_matrix(rotation)
    matrix[:3, 3] = translation
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--scenes", type=Path, default=Path("data/nuscenes_scenes.jsonl"),
                        help="only pack the samples this scene file uses")
    parser.add_argument("--output", type=Path, default=Path("data/nuscenes_perception"))
    args = parser.parse_args()

    meta = args.root / "v1.0-mini"
    sample_data = json.loads((meta / "sample_data.json").read_text())
    calibrated = {c["token"]: c for c in json.loads((meta / "calibrated_sensor.json").read_text())}

    wanted = {json.loads(line)["meta_info"]["token"] for line in open(args.scenes)}

    by_sample: dict[str, dict[str, dict]] = {}
    for sd in sample_data:
        if not sd["is_key_frame"] or sd["sample_token"] not in wanted:
            continue
        channel = sd["filename"].split("/")[1] if "/" in sd["filename"] else ""
        if channel in CAM_ORDER or channel == "LIDAR_TOP":
            by_sample.setdefault(sd["sample_token"], {})[channel] = sd

    args.output.mkdir(parents=True, exist_ok=True)
    written = 0
    for token, channels in by_sample.items():
        if not all(c in channels for c in CAM_ORDER + ["LIDAR_TOP"]):
            continue
        lidar_calib = calibrated[channels["LIDAR_TOP"]["calibrated_sensor_token"]]
        lidar2ego = transform(lidar_calib["rotation"], lidar_calib["translation"])
        ego2lidar = np.linalg.inv(lidar2ego)

        intrinsics, rotations, translations = [], [], []
        for cam in CAM_ORDER:
            calib = calibrated[channels[cam]["calibrated_sensor_token"]]
            cam2lidar = ego2lidar @ transform(calib["rotation"], calib["translation"])
            intrinsics.append(np.asarray(calib["camera_intrinsic"], dtype=np.float64))
            rotations.append(cam2lidar[:3, :3])
            translations.append(cam2lidar[:3, 3])

        frame_dir = args.output / token
        (frame_dir / "images").mkdir(parents=True, exist_ok=True)
        for cam in CAM_ORDER:
            link = frame_dir / "images" / f"{cam}.jpg"
            target = (args.root / channels[cam]["filename"]).resolve()
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)

        (frame_dir / "frame.json").write_text(json.dumps({
            "dataset_type": "nuscenes",
            "cam_order": CAM_ORDER,
            "content": [item for view, cam in zip(VIEW_LABELS, CAM_ORDER)
                        for item in ({"text": view}, {"image": cam})] + [{"text": PROMPT}],
        }, indent=1))
        np.savez(frame_dir / "calib.npz",
                 cam_intrinsic=np.stack(intrinsics),
                 sensor2lidar_rotation=np.stack(rotations),
                 sensor2lidar_translation=np.stack(translations),
                 lidar2ego=lidar2ego)
        # Placeholders: nuScenes has no occupancy or map rasters of its own.
        np.savez_compressed(frame_dir / "gt.npz",
                 occ=np.full(OCC_SHAPE, OCC_EMPTY, dtype=np.int8),
                 map=np.zeros(MAP_SHAPE, dtype=np.int8),
                 boxes=np.zeros((0, 9), dtype=np.float32),
                 labels=np.zeros((0,), dtype=np.int64))
        written += 1

    print(f"wrote {written} perception frames to {args.output}")


if __name__ == "__main__":
    main()
