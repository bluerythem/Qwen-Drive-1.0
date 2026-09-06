#!/usr/bin/env python
"""Flag the nuScenes scenes where the ego could plausibly change lane, in place.

A lane change needs somewhere to go: a lane beside the ego's own, running the same way.
This probes sideways from the ego pose, snaps each probe to a lane, and keeps the lanes whose
heading agrees with the ego's - an oncoming lane across a centre line is not somewhere you
can move to.

It also reports the marking between the lanes. nuScenes types its divider segments, so
`DOUBLE_DASHED_WHITE` (crossable) can be told from `DOUBLE_SOLID_WHITE` (not), which is the
difference between a lane change being possible and merely geometrically imaginable.

    python tools/nuscenes_multilane.py --scenes data/nuscenes_scenes.jsonl

Writes `meta_info.lanes_left`, `lanes_right`, `multi_lane` and `divider_types` per record.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nuscenes_map_gt import NuScenesMapGT  # noqa: E402

PROBE_OFFSETS = np.arange(2.5, 11.0, 0.5)   # m sideways; a lane is ~3.5 m wide
SNAP_RADIUS = 2.0                            # m, for a probe to count as inside a lane
SAME_DIRECTION = np.radians(45.0)            # heading agreement with the ego's own lane
CROSSABLE = {"DOUBLE_DASHED_WHITE", "SINGLE_DASHED_WHITE", "NIL"}


def lane_heading(nmap, lane_token: str, point: np.ndarray) -> float | None:
    """Heading of a lane's centreline at the point on it nearest ``point``."""
    poses = nmap.discretize_lanes([lane_token], 1.0).get(lane_token) or []
    if not poses:
        return None
    poses = np.asarray(poses)
    return float(poses[np.argmin(np.linalg.norm(poses[:, :2] - point, axis=1)), 2])


def real_lanes(nmap) -> set[str]:
    """Lane tokens proper, excluding lane_connectors.

    A lane_connector is a path through a junction. Several of them fan out side by side, so
    counting them as neighbours makes every junction look like a multi-lane road: they were
    61% of all hits before this filter.
    """
    return {record["token"] for record in nmap.lane}


def neighbours(nmap, x: float, y: float, yaw: float, lanes: set[str]) -> tuple[set, set]:
    """Distinct same-direction lanes to the ego's left and right."""
    own = nmap.get_closest_lane(x, y, radius=SNAP_RADIUS)
    if not own:
        return set(), set()
    own_heading = lane_heading(nmap, own, np.array([x, y]))
    if own_heading is None:
        return set(), set()

    left_unit = np.array([-np.sin(yaw), np.cos(yaw)])
    found: dict[str, set] = {"left": set(), "right": set()}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        for offset in PROBE_OFFSETS:
            probe = np.array([x, y]) + sign * offset * left_unit
            lane = nmap.get_closest_lane(probe[0], probe[1], radius=SNAP_RADIUS)
            if not lane or lane == own or lane not in lanes:
                continue
            heading = lane_heading(nmap, lane, probe)
            if heading is None:
                continue
            delta = abs(np.arctan2(np.sin(heading - own_heading), np.cos(heading - own_heading)))
            if delta <= SAME_DIRECTION:
                found[side].add(lane)
    return found["left"], found["right"]


def divider_types(nmap, x: float, y: float) -> list[str]:
    """Marking types on either side of the ego's lane, if the map records them.

    ``get_closest_lane`` can return a lane_connector - a lane through a junction - and those
    carry no divider segments, so there is nothing to report for them.
    """
    own = nmap.get_closest_lane(x, y, radius=SNAP_RADIUS)
    if not own or own not in real_lanes(nmap):
        return []
    record = nmap.get("lane", own)
    segments = (record.get("left_lane_divider_segments", [])
                + record.get("right_lane_divider_segments", []))
    return sorted({s["segment_type"] for s in segments if s.get("segment_type")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scenes", type=Path, default=Path("data/nuscenes_scenes.jsonl"))
    args = parser.parse_args()

    helper = NuScenesMapGT(args.root, args.version)
    if not helper.available():
        raise SystemExit("needs maps/expansion/ from the map-expansion pack")

    records = [json.loads(line) for line in open(args.scenes)]
    tally: Counter = Counter()
    lane_index: dict = {}
    for record in records:
        token = record["meta_info"]["token"]
        x, y, yaw_deg = helper.pose[token]
        nmap = helper.map(helper.location[token])
        lanes = lane_index.setdefault(id(nmap), real_lanes(nmap))
        own = nmap.get_closest_lane(x, y, radius=SNAP_RADIUS)
        in_junction = bool(own) and own not in lanes
        left, right = neighbours(nmap, x, y, np.radians(yaw_deg), lanes)
        types = divider_types(nmap, x, y)

        meta = record["meta_info"]
        meta["lanes_left"] = len(left)
        meta["lanes_right"] = len(right)
        meta["in_junction"] = in_junction
        # Inside a junction there is no lane to change into; the concept does not apply.
        meta["multi_lane"] = bool(left or right) and not in_junction
        meta["divider_types"] = types
        meta["crossable_divider"] = bool(types) and any(t in CROSSABLE for t in types)
        tally["multi"] += meta["multi_lane"]
        tally["junction"] += in_junction
        tally["left"] += bool(left) and not in_junction
        tally["right"] += bool(right) and not in_junction
        tally["crossable"] += meta["multi_lane"] and meta["crossable_divider"]

    with open(args.scenes, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    total = len(records)
    print(f"{tally['multi']}/{total} scenes have a same-direction lane beside the ego "
          f"({tally['left']} with one on the left, {tally['right']} on the right)")
    print(f"{tally['crossable']} of those have a crossable marking recorded")
    print(f"{tally['junction']}/{total} scenes have the ego inside a junction (excluded above)")


if __name__ == "__main__":
    main()
