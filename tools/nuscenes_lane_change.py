#!/usr/bin/env python
"""Flag the nuScenes planning scenes where the ego changes lane, in place.

Detection is geometric, not graph-based: the ego's recorded 5 s path is intersected with
the map's `lane_divider` polylines, and a scene counts as a lane change when the path
crosses one while the vehicle is going roughly straight.

The straightness guard is what separates a lane change from a turn - a vehicle turning at a
junction sweeps across dividers without changing lane in any meaningful sense - and the
lateral guard drops crossings that are an artefact of a path clipping a divider's endpoint
rather than genuinely moving over.

An earlier version of this asked whether the end lane was reachable from the start lane
through the map's lane graph. That does not work: lane records are short and their
connectivity is sparse enough that a plain straight drive frequently ends in an
"unreachable" lane, which flagged 25 scenes with no lateral motion at all.

    python tools/nuscenes_lane_change.py --scenes data/nuscenes_scenes.jsonl

Writes `meta_info.lane_change` (bool) and `meta_info.lane_shift_m` into each record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nuscenes_map_gt import NuScenesMapGT  # noqa: E402

MAX_HEADING_CHANGE = 0.35   # rad over the 5 s; above this it is a turn, not a lane change
MIN_LATERAL_SHIFT = 1.5     # m of sideways motion relative to the starting heading
SEARCH_MARGIN = 20.0        # m of map to pull in around the path


def path_in_global(helper, token: str, future: np.ndarray) -> np.ndarray:
    """The recorded future, expressed in the map's global frame."""
    x, y, yaw_deg = helper.pose[token]
    yaw = np.radians(yaw_deg)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return np.array([x, y]) + future[:, :2] @ rotation.T


def divider_crossings(nmap, path_xy: np.ndarray) -> int:
    """How many distinct lane dividers the path crosses."""
    from shapely.geometry import LineString

    path = LineString(path_xy)
    box = (path_xy[:, 0].min() - SEARCH_MARGIN, path_xy[:, 1].min() - SEARCH_MARGIN,
           path_xy[:, 0].max() + SEARCH_MARGIN, path_xy[:, 1].max() + SEARCH_MARGIN)
    records = nmap.get_records_in_patch(box, ["lane_divider"], mode="intersect")
    crossed = 0
    for token in records.get("lane_divider", []):
        line_token = nmap.get("lane_divider", token)["line_token"]
        divider = nmap.extract_line(line_token)
        # `crosses` wants an interior-interior intersection, but dividers are short 2-point
        # segments and a path often meets one at its endpoint, so `intersects` is correct here.
        if not divider.is_empty and path.intersects(divider):
            crossed += 1
    return crossed


def classify(helper: NuScenesMapGT, token: str, future: np.ndarray) -> tuple[bool, float, int]:
    """(is_lane_change, lateral shift in m, dividers crossed)."""
    # future is already in the ego frame of the current timestep, so y is the sideways
    # displacement relative to where the car was pointing when the scene starts.
    shift = float(future[-1, 1])
    heading = float(abs(future[-1, 2]))
    if heading > MAX_HEADING_CHANGE or abs(shift) < MIN_LATERAL_SHIFT:
        return False, shift, 0

    nmap = helper.map(helper.location[token])
    crossings = divider_crossings(nmap, path_in_global(helper, token, future))
    return crossings > 0, shift, crossings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--scenes", type=Path, default=Path("data/nuscenes_scenes.jsonl"))
    args = parser.parse_args()

    helper = NuScenesMapGT(args.root)
    if not helper.available():
        raise SystemExit("needs data/nuscenes/maps/expansion/ from the map-expansion pack")

    records = [json.loads(line) for line in open(args.scenes)]
    changed = 0
    for record in records:
        token = record["meta_info"]["token"]
        future = np.asarray(record["trajectory"]["future_traj_10hz"], dtype=np.float64)
        is_change, shift, crossings = classify(helper, token, future)
        record["meta_info"]["lane_change"] = bool(is_change)
        record["meta_info"]["lane_shift_m"] = round(shift, 2)
        record["meta_info"]["divider_crossings"] = crossings
        changed += is_change

    with open(args.scenes, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"{changed} of {len(records)} scenes flagged as a lane change")


if __name__ == "__main__":
    main()
