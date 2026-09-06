#!/usr/bin/env python
"""Hand-built navigation targets and ego trajectories, for demonstrating the interface.

Nothing here runs the model. The point is to show what a lane-change command *would* look
like going in and coming out: a navigation target drawn along the centre of the lane the
command points at, and a plausible future the ego might drive in response.

Everything is in the ego frame of the current timestep, x forward and y left, matching the
model's own output so the mock can be drawn by the same plotting code.
"""

from __future__ import annotations

import numpy as np

HORIZON_S = 5.0
HZ = 10.0
NUM_POINTS = 50
LANE_CHANGE_WINDOW = (0.6, 3.6)     # s, when the lateral move happens
PULL_OVER_WINDOW = (0.5, 4.0)
NOODLE_START = 10.0                 # m; the navigation target begins ahead of the ego, not at it
# There is no lane at the kerb, so the pull-over gets an imagined one: a lane whose centre
# sits half its width in from the drivable edge. The mocked future aims at that same centre,
# which puts the car against the edge rather than merely leaning towards it.
IMAGINARY_LANE_HALF_WIDTH = 0.9

COMMANDS = ("GO STRAIGHT", "CHANGE LANE LEFT", "CHANGE LANE RIGHT")
COMMAND_COLOURS = {"GO STRAIGHT": "#1f77b4", "CHANGE LANE LEFT": "#2ca02c",
                   "CHANGE LANE RIGHT": "#ff7f0e"}


def smoothstep(t: np.ndarray, start: float, end: float) -> np.ndarray:
    """A 0..1 blend with zero slope at both ends, so the lateral move has no kink."""
    u = np.clip((t - start) / (end - start), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _profile(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """A polyline resampled as y(x) on ``grid``, tolerating unsorted input."""
    order = np.argsort(x)
    return np.interp(grid, x[order], y[order])


def lane_profiles(geoms: dict, grid: np.ndarray, own_tolerance: float = 1.8) -> dict:
    """Lateral offset of the ego's own lane, the lane to its right, and the left kerb.

    Centrelines arrive as many short pieces, so they are grouped by their offset near the
    ego rather than by lane token: what matters for drawing is where they sit.
    """
    pieces = []
    for line in geoms.get("lane_centreline", []):
        near = line[np.abs(line[:, 0]) < 8.0]
        if len(near) and line[:, 0].max() > 5.0:
            pieces.append((float(np.mean(near[:, 1])), line))
    if not pieces:
        return {}

    own_offset = min(pieces, key=lambda p: abs(p[0]))[0]
    lanes: dict[str, np.ndarray] = {}
    for name, target in (("own", own_offset), ("right", own_offset - 3.4)):
        chosen = [line for offset, line in pieces if abs(offset - target) < own_tolerance]
        if not chosen:
            continue
        stacked = np.vstack(chosen)
        ahead = stacked[stacked[:, 0] >= -5.0]
        lanes[name] = _profile(ahead[:, 0], ahead[:, 1], grid)

    edge = left_edge(geoms, grid, own_offset)
    if edge is not None and "own" in lanes:
        lanes["left_edge"] = edge
        # The imagined pull-over lane runs parallel to the ego's own lane, offset by the
        # kerb distance measured near the ego. Tracking the far-field boundary directly
        # does not work here: it drifts from 2.1 m to 0.6 m over 70 m while the lane itself
        # stays straight, so the noodle would bend across the ego's lane.
        near = grid <= 25.0
        gap = float(np.median(edge[near] - lanes["own"][near]))
        lanes["pull_over"] = lanes["own"] + gap - IMAGINARY_LANE_HALF_WIDTH
    return lanes


def left_edge(geoms: dict, grid: np.ndarray, own_offset: float,
              window: float = 4.0) -> np.ndarray | None:
    """The drivable boundary just to the ego's left, as y(x).

    Taken as the *nearest* boundary point above the ego at each x rather than by
    interpolating the ring: a drivable polygon wraps around, so its points are neither
    sorted in x nor single-valued in y, and interpolating them directly is meaningless.
    """
    xs, ys = [], []
    for ring in geoms.get("drivable_area", []):
        for loop in [ring["exterior"], *ring["holes"]]:
            xs.extend(loop[:, 0])
            ys.extend(loop[:, 1])
    if not xs:
        return None
    xs, ys = np.asarray(xs), np.asarray(ys)

    edge = np.full(len(grid), np.nan)
    for index, gx in enumerate(grid):
        near = (np.abs(xs - gx) < window) & (ys > own_offset + 0.5)
        if near.any():
            edge[index] = ys[near].min()
    valid = ~np.isnan(edge)
    if valid.sum() < 2:
        return None
    return np.interp(grid, grid[valid], edge[valid])


def targets(geoms: dict, speed: float) -> dict[str, np.ndarray]:
    """The navigation target polyline for each command, as (x, y) in the ego frame.

    Each starts ``NOODLE_START`` metres ahead: a route hint is about where to be shortly,
    not a rail bolted to the front bumper, and leaving the gap keeps the ego and the start
    of its plan readable.
    """
    # Just past where the plan ends: a target that runs far beyond it stretches the
    # bird's-eye panel and squashes everything worth looking at.
    reach = max(40.0, speed * HORIZON_S * 1.05)
    grid = np.linspace(0.0, reach, 160)
    lanes = lane_profiles(geoms, grid)
    if "own" not in lanes:
        return {}

    ahead = grid >= NOODLE_START
    out = {"GO STRAIGHT": np.column_stack([grid, lanes["own"]])[ahead]}
    if "right" in lanes:
        out["CHANGE LANE RIGHT"] = np.column_stack([grid, lanes["right"]])[ahead]
    if "pull_over" in lanes:
        out["CHANGE LANE LEFT"] = np.column_stack([grid, lanes["pull_over"]])[ahead]
    return out


def trajectory(command: str, geoms: dict, speed: float) -> np.ndarray:
    """A mocked 5 s future, ``[50, 3]`` of (x, y, heading), for one command.

    Straight and the lane change hold speed; the pull-over decelerates to a stop, which is
    what makes it a pull-over rather than a second lane change.
    """
    time = np.arange(1, NUM_POINTS + 1) / HZ
    reach = max(40.0, speed * HORIZON_S * 1.3)
    grid = np.linspace(0.0, reach, 160)
    lanes = lane_profiles(geoms, grid)
    if "own" not in lanes:
        return np.zeros((NUM_POINTS, 3))

    if command == "CHANGE LANE LEFT":                      # pull over and stop
        travelled = speed * (time - time**2 / (2 * HORIZON_S))
    else:
        travelled = speed * time

    own = np.interp(travelled, grid, lanes["own"])
    if command == "CHANGE LANE RIGHT" and "right" in lanes:
        target = np.interp(travelled, grid, lanes["right"])
        blend = smoothstep(time, *LANE_CHANGE_WINDOW)
    elif command == "CHANGE LANE LEFT" and "pull_over" in lanes:
        target = np.interp(travelled, grid, lanes["pull_over"])
        blend = smoothstep(time, *PULL_OVER_WINDOW)
    else:
        target, blend = own, np.zeros_like(time)

    lateral = own + (target - own) * blend
    heading = np.arctan2(np.gradient(lateral), np.gradient(travelled))
    return np.column_stack([travelled, lateral, heading])
