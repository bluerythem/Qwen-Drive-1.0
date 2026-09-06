#!/usr/bin/env python
"""Scan a nuScenes split for lane changes, from metadata alone.

No camera images are needed: a lane change is decided from the ego poses and the map's
`lane_divider` geometry, both of which live in `v1.0-trainval_meta.tgz` (0.5 GB) and the map
expansion. The image blobs are ~350 GB and cannot be fetched per file, so this runs first and
tells you which scenes would be worth the download.

    python tools/nuscenes_scan_lane_changes.py --root data/nuscenes-trainval \\
        --version v1.0-trainval --split val --output data/lane_changes_val.json

Same test as tools/nuscenes_lane_change.py: the recorded 5 s future is intersected with the
lane dividers, gated on the vehicle going roughly straight so a turn is not counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).parent))
from nuscenes_map_gt import NuScenesMapGT  # noqa: E402
from nuscenes_lane_change import MAX_HEADING_CHANGE, MIN_LATERAL_SHIFT  # noqa: E402

HISTORY_KEYFRAMES = 3        # 1.5 s at 2 Hz, what a planning scene needs before it
FUTURE_SECONDS = 5.0
SEARCH_MARGIN = 20.0


def ego_frame(poses: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Global (x, y, yaw) rows expressed in the ego frame of ``origin``."""
    yaw = origin[2]
    rotation = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    local = (poses[:, :2] - origin[:2]) @ rotation.T
    heading = np.arctan2(np.sin(poses[:, 2] - yaw), np.cos(poses[:, 2] - yaw))
    return np.column_stack([local, heading])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes-trainval"))
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--maps-from", type=Path, default=Path("data/nuscenes"),
                        help="directory holding maps/expansion (shared across versions)")
    parser.add_argument("--output", type=Path, default=Path("data/lane_changes_val.json"))
    args = parser.parse_args()

    from nuscenes.utils import splits

    wanted = set(getattr(splits, args.split))
    helper = NuScenesMapGT(args.root, args.version)
    helper.root = args.maps_from          # the expansion is version-independent
    if not helper.available():
        raise SystemExit(f"needs {args.maps_from}/maps/expansion/")

    meta = args.root / args.version
    scenes = json.loads((meta / "scene.json").read_text())
    samples = {s["token"]: s for s in json.loads((meta / "sample.json").read_text())}

    found, per_scene = [], defaultdict(int)
    scanned = 0
    for scene in scenes:
        if scene["name"] not in wanted:
            continue
        chain, token = [], scene["first_sample_token"]
        while token:
            chain.append(token)
            token = samples[token]["next"]
        usable = [t for t in chain if t in helper.pose]
        if len(usable) != len(chain):
            continue
        poses = np.array([helper.pose[t] for t in chain], dtype=np.float64)
        poses[:, 2] = np.radians(poses[:, 2])
        times = np.array([samples[t]["timestamp"] for t in chain]) * 1e-6
        nmap = helper.map(helper.location[chain[0]])

        for index in range(HISTORY_KEYFRAMES, len(chain)):
            ahead = np.flatnonzero(times - times[index] <= FUTURE_SECONDS + 1e-6)
            ahead = ahead[ahead > index]
            if len(ahead) < 2 or times[ahead[-1]] - times[index] < FUTURE_SECONDS - 0.6:
                continue                                   # not enough recorded future
            scanned += 1
            local = ego_frame(poses[ahead], poses[index])
            shift, heading = float(local[-1, 1]), abs(float(local[-1, 2]))
            if heading > MAX_HEADING_CHANGE or abs(shift) < MIN_LATERAL_SHIFT:
                continue

            path_xy = poses[ahead][:, :2]
            path = LineString(np.vstack([poses[index, :2], path_xy]))
            box = (path_xy[:, 0].min() - SEARCH_MARGIN, path_xy[:, 1].min() - SEARCH_MARGIN,
                   path_xy[:, 0].max() + SEARCH_MARGIN, path_xy[:, 1].max() + SEARCH_MARGIN)
            hits = 0
            for record in nmap.get_records_in_patch(box, ["lane_divider"], "intersect").get(
                    "lane_divider", []):
                line = nmap.extract_line(nmap.get("lane_divider", record)["line_token"])
                if not line.is_empty and path.intersects(line):
                    hits += 1
            if hits:
                per_scene[scene["name"]] += 1
                found.append({
                    "scene": scene["name"], "location": helper.location[chain[0]],
                    "sample_token": chain[index], "keyframe": index,
                    "lateral_shift_m": round(shift, 2), "heading_change_rad": round(heading, 3),
                    "divider_crossings": hits,
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(found, indent=1))
    print(f"scanned {scanned} usable keyframes across {len(wanted)} {args.split} scenes")
    print(f"{len(found)} lane-change keyframes in {len(per_scene)} scenes -> {args.output}")
    for name, count in sorted(per_scene.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {name}  {count} keyframes")


if __name__ == "__main__":
    main()
