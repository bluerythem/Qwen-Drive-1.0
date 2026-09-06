#!/usr/bin/env python
"""Trion: a mocked two-system driving stack, end to end, with a real map resolver.

    user command ──► Trion-Reason (symbols) ──► Resolver (HD map, deterministic)
                                                       ──► Trion-Action (trajectory)

Only the resolver is real. Trion-Reason is a rule-based stand-in for the slow reasoning
model: it turns a command into a lane-relative goal with a window and a speed preference,
never geometry. Trion-Action is a stand-in for the fast planner: it picks *when* inside the
window and draws the manoeuvre. Nothing here runs a neural network; the point is to exercise
and show the interface between the two systems.

Frames: everything the systems exchange is in the map frame conceptually; for display it is
expressed in the ego frame of the current keyframe (x forward, y left).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# nuScenes' map carries no speed limits, so an urban limit is assumed.
SPEED_LIMIT = 50.0 / 3.6            # m/s
CRUISE_STEP = 1.5                   # m/s per preference notch
MAX_ACCEL = 1.5                     # m/s^2, Trion-Action's comfort bound
LANE_HALF_WIDTH = 1.7
HORIZON_S, HZ = 5.0, 10.0
CROSSABLE = ("DASHED", "NIL")       # divider marking substrings that permit a change

COLOURS = {"KEEP": "#1f77b4", "LEFT": "#2ca02c", "RIGHT": "#ff7f0e", "PULL_OVER": "#9467bd"}


# --------------------------------------------------------------------------- Trion-Reason
@dataclass
class ReasonMessage:
    command: str
    lateral: str                    # KEEP | LEFT | RIGHT | PULL_OVER
    window_m: tuple[float, float]   # complete the manoeuvre between these distances
    deadline_m: float               # after this it is no longer wanted
    cruise: int                     # notches relative to the speed limit
    planned_stop: str | None
    which_light: str
    why: str
    source: str = "NONE"            # ROUTE | VOICE | NONE - who asked for the lateral goal
    pending: str | None = None      # e.g. "+1 RIGHT before the junction": the next goal
    reroute: bool = False           # the route's manoeuvre cannot be made; ask nav to reroute
    route_turn_m: float | None = None

    @property
    def cruise_label(self) -> str:
        return f"LIMIT{self.cruise:+d}"


def reason(command: str, preference: int, ctx: dict, route: dict | None = None) -> ReasonMessage:
    """The slow system, mocked: route step + voice command + context in, one symbolic goal out.

    Two intents meet here and this is where they are arbitrated. The route is the default;
    a voice request is a scoped override - it wins until done, then the route resumes. The
    route's own feasibility is judged symbolically: N lane changes need roughly N x 2.5 s of
    travel plus a margin, and if the junction is closer than that the honest answer is to
    continue and ask the nav app to reroute, not to force the gap.

    It deliberately does *not* consult lane geometry for the voice request - a reasoning
    model may not know there is no lane to the left. Catching that is the resolver's job.
    """
    text = command.lower()
    speed = float(ctx.get("speed", 10.0))
    why: list[str] = []

    # -- the route's demand ---------------------------------------------------------------
    r_lateral, r_window, r_deadline, r_stop, pending, reroute, turn_m = "KEEP", (0.0, 0.0), 0.0, None, None, False, None
    per_change = 2.5 * speed + 10.0
    if route and route.get("matched"):
        turn_m = route.get("at_m")
        if route["manoeuvre"] == "ARRIVE":
            r_lateral, r_stop = "PULL_OVER", "PULL_OVER_POINT"
            r_window, r_deadline = (10.0, 50.0), 60.0
            why.append(f"Route: arriving, destination on the {route.get('side', 'LEFT').lower()}.")
        elif route.get("changes_needed"):
            n, direction = route["changes_needed"], route["direction"]
            needed = n * per_change + 15.0
            if turn_m >= needed:
                r_lateral = direction
                r_window = (10.0, 10.0 + per_change)
                r_deadline = turn_m - 15.0 - (n - 1) * per_change
                if n > 1:
                    pending = f"+{n - 1} {direction} before the junction"
                why.append(f"Route: {route['manoeuvre'].lower()} in {turn_m:.0f} m needs {n} change(s) "
                           f"{direction.lower()}; {turn_m:.0f} m is enough for that at {speed:.0f} m/s.")
            else:
                reroute = True
                why.append(f"Route: {route['manoeuvre'].lower()} in {turn_m:.0f} m needs {n} change(s) "
                           f"{direction.lower()}, which takes ~{needed:.0f} m at {speed:.0f} m/s. "
                           f"Not safely possible: continue and ask the nav app to reroute.")
        elif route["manoeuvre"] != "CONTINUE":
            why.append(f"Route: {route['manoeuvre'].lower()} in {turn_m:.0f} m; already in a lane that "
                       f"makes it, so hold the lane.")
        else:
            why.append("Route: continue on this road.")
    elif route and not route.get("matched"):
        why.append(f"Route: {route.get('reason', 'could not be matched to the map')}; follow the road.")

    # -- the voice request ------------------------------------------------------------------
    v_lateral, v_stop = "KEEP", None
    if "pull" in text or "park" in text or ("stop" in text and "don't" not in text):
        v_lateral, v_stop = "PULL_OVER", "PULL_OVER_POINT"
    elif "left" in text:
        v_lateral = "LEFT"
    elif "right" in text:
        v_lateral = "RIGHT"

    # -- arbitration ------------------------------------------------------------------------
    if v_lateral != "KEEP":
        lateral, stop, source = v_lateral, v_stop, "VOICE"
        window = (10.0, 40.0) if v_lateral == "PULL_OVER" else (12.0, 45.0)
        deadline = 60.0 if v_lateral == "PULL_OVER" else 120.0
        why.append({"PULL_OVER": "Voice: pull over - leave the carriageway on the near side and stop.",
                    "LEFT": "Voice: take the lane to the left.",
                    "RIGHT": "Voice: take the lane to the right."}[v_lateral])
        if r_lateral != "KEEP" and r_lateral != v_lateral:
            why.append("The voice request overrides the route for now; the route resumes after, "
                       "and the nav app will reroute if the turn is missed.")
            pending = f"route: {r_lateral} for the turn" if not reroute else pending
    else:
        lateral, stop, window, deadline = r_lateral, r_stop, r_window, r_deadline
        source = "ROUTE" if r_lateral != "KEEP" or reroute else "NONE"
        if not why:
            why.append("No lane request: continue in the current lane.")

    cruise = int(np.clip(preference + ("faster" in text) - ("slower" in text), -2, 1))
    why.append({-2: "Cruise well under the limit.", -1: "Cruise a notch under the limit.",
                0: "Cruise at the limit.", 1: "Cruise a notch over the limit."}[cruise])
    if lateral == "KEEP":
        window, deadline = (0.0, 0.0), 0.0
    return ReasonMessage(command, lateral, window, deadline, cruise, stop, "STRAIGHT", " ".join(why),
                         source=source, pending=pending, reroute=reroute, route_turn_m=turn_m)


# ------------------------------------------------------------------------------- Resolver
@dataclass
class Resolved:
    lateral: str                        # what will actually be driven
    requested: str                      # what Trion-Reason asked for
    fallback: bool
    current_xy: np.ndarray              # (N, 2) ego frame, the lane the ego is in
    target_xy: np.ndarray               # (N, 2) ego frame, the lane to end in
    window_m: tuple[float, float]
    deadline_m: float
    speed_cap: float
    stop_x: float | None
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def colour(self) -> str:
        return COLOURS[self.lateral]


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


class Resolver:
    """Deterministic: HD map + localization in, corridors and checks out. No learning."""

    def __init__(self, helper, lanes_by_map: dict | None = None, horizon: float = 110.0):
        self.helper = helper
        self.horizon = horizon
        self._lanes = lanes_by_map or {}

    # -- map helpers ------------------------------------------------------------------
    def _lane_tokens(self, nmap) -> set[str]:
        key = id(nmap)
        if key not in self._lanes:
            self._lanes[key] = {r["token"] for r in nmap.lane}
        return self._lanes[key]

    def _to_ego(self, xy: np.ndarray, pose) -> np.ndarray:
        x, y, yaw = pose
        rot = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        return (xy - (x, y)) @ rot.T

    def _lane_poses(self, nmap, token) -> np.ndarray:
        return np.asarray(nmap.discretize_lanes([token], 1.0).get(token) or np.zeros((0, 3)))

    def _corridor(self, nmap, start: str, pose) -> np.ndarray:
        """Walk outgoing lanes from ``start``, straightest branch first, to the horizon."""
        pieces, token, travelled = [], start, 0.0
        while token and travelled < self.horizon + 30.0:
            poses = self._lane_poses(nmap, token)
            if len(poses) < 2:
                break
            pieces.append(poses[:, :2])
            travelled += float(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1).sum())
            outgoing = nmap.get_outgoing_lane_ids(token)
            if not outgoing:
                break
            end_yaw = poses[-1, 2]
            token = min(outgoing, key=lambda t: abs(float(_wrap(
                (self._lane_poses(nmap, t)[0, 2] if len(self._lane_poses(nmap, t)) else end_yaw)
                - end_yaw))))
        if not pieces:
            return np.zeros((0, 2))
        local = self._to_ego(np.vstack(pieces), pose)
        local = local[local[:, 0] > -10.0]
        return local[np.argsort(local[:, 0])]

    @staticmethod
    def _y_at(xy: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.interp(x, xy[:, 0], xy[:, 1])

    def _side_profile(self, nmap, corridor, pose, side: str, lanes: set[str], own: str,
                      step: float = 4.0, reach: float = 90.0) -> list[tuple[float, str, str | None]]:
        """What lies beside the corridor at each distance: ("lane", token), "junction" or "none".

        A lane change is only possible where the target side is a proper lane. Junction
        connectors count as a junction - no changing lanes inside one - and their extent is
        what the resolver uses to move the window.
        """
        x, y, yaw = pose
        rot_back = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        sign = 1.0 if side == "LEFT" else -1.0
        profile = []
        for s in np.arange(step, reach + step, step):
            base = np.array([s, float(self._y_at(corridor, np.array([s]))[0])])
            found, token = "none", None
            for offset in np.arange(2.5, 7.5, 0.5):
                gx, gy = rot_back @ (base + (0.0, sign * offset)) + (x, y)
                cand = nmap.get_closest_lane(gx, gy, radius=2.0)
                if not cand or cand == own:
                    continue
                if cand not in lanes:
                    found = "junction"
                    break
                poses = self._lane_poses(nmap, cand)
                if len(poses) == 0:
                    continue
                near = poses[np.argmin(np.linalg.norm(poses[:, :2] - (gx, gy), axis=1))]
                if abs(float(_wrap(near[2] - yaw))) < np.radians(45):
                    found, token = "lane", cand
                break
            profile.append((float(s), found, token))
        return profile

    @staticmethod
    def _lane_runs(profile) -> list[tuple[float, float, str]]:
        """Contiguous stretches where the side is a real lane: (start, end, first token)."""
        runs, current = [], None
        for s, kind, token in profile:
            if kind == "lane":
                if current is None:
                    current = [s, s, token]
                current[1] = s
            elif current is not None:
                runs.append(tuple(current))
                current = None
        if current is not None:
            runs.append(tuple(current))
        return runs

    # -- the resolver proper -------------------------------------------------------------
    def resolve(self, token: str, msg: ReasonMessage, geoms: dict | None = None,
                speed: float | None = None) -> Resolved:
        self.helper_speed = speed
        x, y, yaw_deg = self.helper.pose[token]
        pose = (x, y, np.radians(yaw_deg))
        nmap = self.helper.map(self.helper.location[token])
        lanes = self._lane_tokens(nmap)
        checks: list[tuple[str, bool, str]] = []

        own = nmap.get_closest_lane(x, y, radius=2.0)
        in_lane = bool(own) and own in lanes
        checks.append(("localized in a lane", in_lane,
                       f"lane {own[:8]}" if in_lane else ("inside a junction" if own else "no lane within 2 m")))
        if not in_lane:
            zero = np.array([[0.0, 0.0], [self.horizon, 0.0]])
            return Resolved("KEEP", msg.lateral, msg.lateral != "KEEP", zero, zero, (0, 0), 0,
                            SPEED_LIMIT + CRUISE_STEP * msg.cruise, None, checks)

        current = self._corridor(nmap, own, pose)
        cap = SPEED_LIMIT + CRUISE_STEP * msg.cruise
        target, lateral, fallback, stop_x = current, msg.lateral, False, None
        window, deadline = msg.window_m, msg.deadline_m

        if msg.lateral in ("LEFT", "RIGHT"):
            profile = self._side_profile(nmap, current, pose, msg.lateral, lanes, own)
            runs = self._lane_runs(profile)
            junctions = [s for s, kind, _ in profile if kind == "junction"]
            # Long enough to complete the change at this speed, ~2.5 s of travel.
            needed = max(25.0, 2.5 * (speed or 10.0))
            usable = [r for r in runs if r[1] - max(r[0], window[0]) >= needed * 0.6
                      and r[0] <= msg.deadline_m]
            neighbour = usable[0][2] if usable else None
            side = msg.lateral.lower()
            if neighbour is None:
                note = "none found" if not runs else (
                    f"a lane exists at {runs[0][0]:.0f}-{runs[0][1]:.0f} m but not for long enough")
                checks.append((f"same-direction lane on the {side}", False, note))
            else:
                run_start, run_end, _ = usable[0]
                new_start = max(window[0], run_start)
                new_end = min(run_end, new_start + needed)
                if msg.deadline_m:
                    new_end = min(new_end, msg.deadline_m)   # the route's turn is after this
                shifted = abs(new_start - window[0]) > 1.0 or abs(new_end - window[1]) > 1.0
                checks.append((f"same-direction lane on the {side}", True,
                               f"lane {neighbour[:8]}, continuous {run_start:.0f}-{run_end:.0f} m"))
                if shifted:
                    reason_txt = (f"junction beside the road at {min(junctions):.0f}-{max(junctions):.0f} m"
                                  if junctions else "lane not available that early")
                    checks.append(("window fitted to the map", True,
                                   f"{window[0]:.0f}-{window[1]:.0f} m -> {new_start:.0f}-{new_end:.0f} m "
                                   f"({reason_txt})"))
                window = (new_start, new_end)
            side_key = "left" if msg.lateral == "LEFT" else "right"
            marks = {s["segment_type"] for s in nmap.get("lane", own)[f"{side_key}_lane_divider_segments"]}
            crossable = (not marks) or all(any(c in m for c in CROSSABLE) for m in marks)
            checks.append(("divider is crossable", crossable,
                           ", ".join(sorted(marks)) if marks else "no marking recorded (kerb or unmarked)"))
            distinct = False
            if neighbour is not None:
                target = self._corridor(nmap, neighbour, pose)
                reach = float(target[:, 0].max()) if len(target) else 0.0
                continuous = reach >= window[1] + 5.0
                checks.append(("target lane continues past the window", continuous,
                               f"mapped to {reach:.0f} m ahead"))
                # Maps carry overlapping lane records at splits and merges. A "neighbour"
                # whose centreline sits on top of the current one is not somewhere to go.
                xs = np.linspace(window[0], window[1], 8)
                gap = float(np.mean(np.abs(self._y_at(target, xs) - self._y_at(current, xs))))
                distinct = gap >= 2.0
                checks.append((f"target lane is a distinct lane", distinct,
                               f"{gap:.1f} m from the current centreline"))
                deadline = min(deadline, reach)
            else:
                continuous = False
            if not (neighbour is not None and crossable and continuous and distinct):
                lateral, fallback, target = "KEEP", True, current
                window, deadline = (0.0, 0.0), 0.0

        elif msg.lateral == "PULL_OVER":
            edge = _left_edge(geoms or {}, current)
            checks.append(("kerb-side boundary found", edge is not None,
                           f"{edge:.1f} m left of the lane centre" if edge is not None else "no drivable boundary on the near side"))
            if edge is not None:
                shift = edge - 0.9
                target = np.column_stack([current[:, 0], current[:, 1] + shift])
                # Stop where a 2.5 m/s^2 brake from the current speed lands, or at the end
                # of the window, whichever comes first.
                stop_x = min(window[1], self.helper_speed**2 / (2.0 * 2.5)) if self.helper_speed else window[1]
                window = (min(window[0], 0.3 * stop_x), stop_x)   # be at the kerb by the stop
            else:
                lateral, fallback, window, deadline = "KEEP", True, (0.0, 0.0), 0.0

        if fallback:
            checks.append(("fallback", True, "request rejected; continuing in the current lane"))
        return Resolved(lateral, msg.lateral, fallback, current, target, window, deadline, cap, stop_x, checks)


def _left_edge(geoms: dict, corridor: np.ndarray, reach: float = 25.0) -> float | None:
    """Kerb distance measured near the ego: nearest drivable boundary above the lane."""
    xs, ys = [], []
    for ring in geoms.get("drivable_area", []):
        for loop in [ring["exterior"], *ring["holes"]]:
            xs.extend(loop[:, 0])
            ys.extend(loop[:, 1])
    if not xs:
        return None
    xs, ys = np.asarray(xs), np.asarray(ys)
    gaps = []
    for s in np.linspace(0.0, reach, 6):
        own = float(np.interp(s, corridor[:, 0], corridor[:, 1]))
        near = (np.abs(xs - s) < 4.0) & (ys > own + 0.5)
        if near.any():
            gaps.append(float(ys[near].min() - own))
    return float(np.median(gaps)) if gaps else None


# ---------------------------------------------------------------------------- Trion-Action
def smoothstep(t, a, b):
    u = np.clip((t - a) / max(b - a, 1e-6), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def act(res: Resolved, speed: float) -> tuple[np.ndarray, dict]:
    """The fast system, mocked: a comfort-bounded speed profile, and the lateral move placed
    inside the window. Returns ``[50, 3]`` (x, y, heading) and a summary."""
    t = np.arange(1, int(HORIZON_S * HZ) + 1) / HZ
    dt = 1.0 / HZ
    if res.stop_x is not None:
        decel = speed**2 / (2.0 * max(res.stop_x, 1.0))
        v = np.maximum(speed - decel * t, 0.0)
    else:
        v = speed + np.clip(res.speed_cap - speed, -MAX_ACCEL * t, MAX_ACCEL * t)
    x = np.cumsum(v) * dt
    if res.stop_x is not None:
        x = np.minimum(x, res.stop_x)

    y_cur = np.interp(x, res.current_xy[:, 0], res.current_xy[:, 1])
    y_tgt = np.interp(x, res.target_xy[:, 0], res.target_xy[:, 1])
    s_min, s_max = res.window_m
    go = s_min + 0.35 * (s_max - s_min)          # the "gap accepted" moment, mocked
    blend = smoothstep(x, go, s_max) if res.lateral != "KEEP" else np.zeros_like(x)
    yv = y_cur + (y_tgt - y_cur) * blend
    heading = np.arctan2(np.gradient(yv), np.gradient(x))

    # Measured against the current lane's centreline at the same distance, so following a
    # curving lane reads as 0 and a lane change reads as one lane width.
    summary = {"reach_m": float(x[-1]), "lateral_move_m": float(yv[-1] - y_cur[-1]),
               "final_speed": float(v[-1]), "commit_at_m": float(go) if res.lateral != "KEEP" else None}
    return np.column_stack([x, yv, heading]), summary


# ------------------------------------------------------------------------ Route matcher
# A nav app hands over a road-level route: "turn left in 180 m". The stack needs lane-level
# goals. This turns one into the other from the HD map: which lanes lead into the requested
# turn at the next junction, how many changes that is from the ego's lane, and how far away.

TURN_DEG = 25.0                      # heading change through a connector that counts as a turn
MANOEUVRES = ("CONTINUE", "TURN LEFT", "TURN RIGHT", "ARRIVE")


def lane_group(resolver: Resolver, nmap, pose, own: str, lanes: set[str]) -> list[str]:
    """Same-direction lanes across the road at the ego, ordered left to right."""
    x, y, yaw = pose
    left_unit = np.array([-np.sin(yaw), np.cos(yaw)])
    found: dict[str, float] = {own: 0.0}
    for offset in np.arange(-11.0, 11.5, 0.5):
        gx, gy = np.array([x, y]) + offset * left_unit
        cand = nmap.get_closest_lane(gx, gy, radius=2.0)
        if not cand or cand in found or cand not in lanes:
            continue
        poses = resolver._lane_poses(nmap, cand)
        if len(poses) == 0:
            continue
        near = poses[np.argmin(np.linalg.norm(poses[:, :2] - (gx, gy), axis=1))]
        if abs(float(_wrap(near[2] - yaw))) < np.radians(45):
            found[cand] = float(offset)
    return [token for token, _ in sorted(found.items(), key=lambda kv: -kv[1])]  # left first


def junction_options(resolver: Resolver, nmap, pose, token: str, lanes: set[str],
                     horizon: float = 160.0) -> dict | None:
    """Walk a lane forward to the first junction with a real turn; report what it offers."""
    current, travelled = token, 0.0
    for _ in range(8):
        poses = resolver._lane_poses(nmap, current)
        if len(poses) < 2:
            return None
        end_x = float(resolver._to_ego(poses[-1:, :2], pose)[0, 0])
        outgoing = nmap.get_outgoing_lane_ids(current)
        turns = {}
        for out in outgoing:
            q = resolver._lane_poses(nmap, out)
            if len(q) < 2:
                continue
            change = float(np.degrees(_wrap(q[-1, 2] - q[0, 2])))
            kind = "TURN LEFT" if change > TURN_DEG else "TURN RIGHT" if change < -TURN_DEG else "CONTINUE"
            turns.setdefault(kind, out)
        if any(k != "CONTINUE" for k in turns) and end_x <= horizon:
            return {"at_m": end_x, "options": turns}
        if end_x > horizon or not outgoing:
            return {"at_m": end_x, "options": turns} if turns else None
        # straightest continuation
        current = min(outgoing, key=lambda o: abs(float(_wrap(
            (resolver._lane_poses(nmap, o)[0, 2] if len(resolver._lane_poses(nmap, o)) else 0.0)
            - poses[-1, 2]))))
    return None


def lane_route(resolver: Resolver, token: str, manoeuvre: str, arrive_side: str = "LEFT") -> dict:
    """The LaneRoute step for one requested manoeuvre at the next junction."""
    helper = resolver.helper
    x, y, yaw_deg = helper.pose[token]
    pose = (x, y, np.radians(yaw_deg))
    nmap = helper.map(helper.location[token])
    lanes = resolver._lane_tokens(nmap)
    own = nmap.get_closest_lane(x, y, radius=2.0)
    if not own or own not in lanes:
        return {"matched": False, "reason": "ego not in a lane (junction)", "manoeuvre": manoeuvre}

    group = lane_group(resolver, nmap, pose, own, lanes)
    own_index = group.index(own)
    per_lane = {t: junction_options(resolver, nmap, pose, t, lanes) for t in group}
    step = {"manoeuvre": manoeuvre, "matched": True, "lanes": len(group), "current_lane": own_index,
            "per_lane": {t[:8]: (o["options"] if o else {}) for t, o in per_lane.items()}}

    if manoeuvre == "ARRIVE":
        step.update(at_m=60.0, side=arrive_side, valid=[0 if arrive_side == "LEFT" else len(group) - 1])
    elif manoeuvre == "CONTINUE":
        step.update(at_m=None, valid=list(range(len(group))))
    else:
        junctions = [(o["at_m"], t) for t, o in per_lane.items() if o and manoeuvre in o["options"]]
        if not junctions:
            nearest = min((o["at_m"] for o in per_lane.values() if o), default=None)
            step.update(matched=False, at_m=nearest,
                        reason=f"no {manoeuvre.lower()} from this road within {160:.0f} m")
            return step
        at_m = min(a for a, _ in junctions)
        valid = [group.index(t) for a, t in junctions if abs(a - at_m) < 15.0]
        step.update(at_m=at_m, valid=sorted(valid))
    valid = step.get("valid", [])
    if valid:
        nearest_valid = min(valid, key=lambda i: abs(i - own_index))
        step["changes_needed"] = abs(nearest_valid - own_index)
        step["direction"] = ("LEFT" if nearest_valid < own_index else
                             "RIGHT" if nearest_valid > own_index else None)
    return step
