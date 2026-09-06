"""Ground-truth BEV map rasters for nuScenes keyframes, in the perception head's layout.

The nuScenes map expansion (``maps/expansion/<location>.json``) holds the vector map. This
module rasterizes a 60 m x 30 m ego-centred patch of it into the same ``(200, 400)`` grid
of six classes that ``QwenDrivePerception`` predicts, so the two can be drawn, compared,
or written into a frame's ``gt.npz`` interchangeably.

Class mapping (the model's names on the left, nuScenes layers on the right):

    driveable_surface  <- drivable_area
    walkway            <- walkway
    crosswalk          <- ped_crossing
    road_line          <- lane_divider + road_divider   (painted lines)
    road_edge          <- boundary of drivable_area     (nuScenes has no edge layer)

Painting order is fill, then walkway and crosswalk, then edges, then lines on top, so the
thin classes are not swallowed by the fills.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# Model layout, from qwen_drive_perception.configuration_perception.
MAP_XBOUND = (-30.0, 30.0, 0.15)     # longitudinal, 400 cells
MAP_YBOUND = (-15.0, 15.0, 0.15)     # lateral, 200 cells
CANVAS = (200, 400)                  # (rows = Y, cols = X)
BACKGROUND, DRIVEABLE, ROAD_LINE, ROAD_EDGE, CROSSWALK, WALKWAY = range(6)

LAYERS = ["drivable_area", "walkway", "ped_crossing", "lane_divider", "road_divider"]


def quat_to_yaw_deg(q) -> float:
    w, x, y, z = q
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def _rings(geom) -> list[dict]:
    """Shapely polygon geometry as {exterior, holes} arrays; other types give nothing."""
    kind = geom.geom_type
    if kind == "Polygon":
        return [{
            "exterior": np.asarray(geom.exterior.coords),
            "holes": [np.asarray(r.coords) for r in geom.interiors],
        }]
    if kind in ("MultiPolygon", "GeometryCollection"):
        return [ring for part in geom.geoms for ring in _rings(part)]
    return []


def _polylines(geom) -> list[np.ndarray]:
    kind = geom.geom_type
    if kind == "LineString":
        return [np.asarray(geom.coords)] if len(geom.coords) > 1 else []
    if kind in ("MultiLineString", "GeometryCollection"):
        return [line for part in geom.geoms for line in _polylines(part)]
    return []


def _runs(mask: np.ndarray):
    """Start/stop index pairs of each contiguous True run."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


class NuScenesMapGT:
    """Lazy wrapper: one ``NuScenesMap`` per city, built on first use."""

    def __init__(self, root: str | Path, version: str = "v1.0-mini"):
        self.root = Path(root)
        meta = self.root / version
        samples = {s["token"]: s for s in json.loads((meta / "sample.json").read_text())}
        scenes = {s["token"]: s for s in json.loads((meta / "scene.json").read_text())}
        logs = {l["token"]: l for l in json.loads((meta / "log.json").read_text())}
        ego_pose = {e["token"]: e for e in json.loads((meta / "ego_pose.json").read_text())}

        # Pose of each keyframe, taken from its LIDAR_TOP sample_data (the sample's own time).
        self.pose: dict[str, tuple[float, float, float]] = {}
        for sd in json.loads((meta / "sample_data.json").read_text()):
            if sd["is_key_frame"] and "LIDAR_TOP" in sd["filename"]:
                p = ego_pose[sd["ego_pose_token"]]
                self.pose[sd["sample_token"]] = (
                    p["translation"][0], p["translation"][1], quat_to_yaw_deg(p["rotation"])
                )
        self.location = {
            token: logs[scenes[s["scene_token"]]["log_token"]]["location"]
            for token, s in samples.items()
        }
        self._maps: dict = {}

    def map(self, location: str):
        if location not in self._maps:
            from nuscenes.map_expansion.map_api import NuScenesMap

            self._maps[location] = NuScenesMap(dataroot=str(self.root), map_name=location)
        return self._maps[location]

    def available(self) -> bool:
        return (self.root / "maps" / "expansion").is_dir()

    def layers(self, sample_token: str) -> dict[str, np.ndarray]:
        """Binary masks per nuScenes layer, ``(200, 400)`` each, ego frame."""
        x, y, yaw = self.pose[sample_token]
        nmap = self.map(self.location[sample_token])
        patch = (x, y, MAP_YBOUND[1] - MAP_YBOUND[0], MAP_XBOUND[1] - MAP_XBOUND[0])
        masks = nmap.get_map_mask(patch, yaw, LAYERS, canvas_size=CANVAS)
        return {name: mask.astype(bool) for name, mask in zip(LAYERS, masks)}

    def vector(self, sample_token: str, centreline_spacing: float = 1.0) -> dict:
        """Ego-frame vector geometry for the patch, plus discretized lane centrelines.

        Polygon layers come back as ``{exterior, holes}``; line layers and centrelines as
        arrays of ``(x_forward, y_left)`` points.
        """
        x, y, yaw = self.pose[sample_token]
        nmap = self.map(self.location[sample_token])
        patch = (x, y, MAP_YBOUND[1] - MAP_YBOUND[0], MAP_XBOUND[1] - MAP_XBOUND[0])

        out: dict[str, list] = {}
        for name, geoms in nmap.get_map_geom(patch, yaw, POLYGON_LAYERS + LINE_LAYERS):
            collect = _rings if name in POLYGON_LAYERS else _polylines
            out.setdefault(name, []).extend(item for g in geoms for item in collect(g))
        out["lane_centreline"] = self._centrelines(nmap, x, y, yaw, centreline_spacing)
        return out

    def _centrelines(self, nmap, x, y, yaw_deg, spacing) -> list[np.ndarray]:
        """Lane centrelines from arcline_path_3, clipped to the patch.

        These are global poses, unlike get_map_geom's output, so they are rotated here.
        """
        reach = float(np.hypot(MAP_XBOUND[1], MAP_YBOUND[1])) + 5.0
        records = nmap.get_records_in_patch(
            (x - reach, y - reach, x + reach, y + reach), ["lane", "lane_connector"], mode="intersect"
        )
        tokens = [t for layer in ("lane", "lane_connector") for t in records.get(layer, [])]
        if not tokens:
            return []
        yaw = np.radians(yaw_deg)
        rotation = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])

        lines = []
        for poses in nmap.discretize_lanes(tokens, spacing).values():
            if len(poses) < 2:
                continue
            local = (np.asarray(poses)[:, :2] - (x, y)) @ rotation.T
            inside = (np.abs(local[:, 0]) <= -MAP_XBOUND[0]) & (np.abs(local[:, 1]) <= -MAP_YBOUND[0])
            # A lane can leave and re-enter the patch, so keep each run separately.
            for start, stop in _runs(inside):
                if stop - start > 1:
                    lines.append(local[start:stop])
        return lines

    def raster(self, sample_token: str) -> np.ndarray:
        """The six-class grid, ``int8 (200, 400)``, indexed like the model's prediction."""
        m = self.layers(sample_token)
        grid = np.full(CANVAS, BACKGROUND, dtype=np.int8)
        grid[m["drivable_area"]] = DRIVEABLE
        grid[m["walkway"]] = WALKWAY
        grid[m["ped_crossing"]] = CROSSWALK

        drivable = m["drivable_area"].astype(np.uint8)
        eroded = cv2.erode(drivable, np.ones((3, 3), np.uint8), iterations=1)
        edge = (drivable & ~eroded).astype(bool)
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
        grid[edge] = ROAD_EDGE                     # ~2 px, like the devkit's lines

        grid[m["lane_divider"] | m["road_divider"]] = ROAD_LINE
        return grid


# Vector form. get_map_geom clips each record to the patch and returns it already rotated
# into the ego frame (local X forward, Y left), so the geometry needs no further transform.
POLYGON_LAYERS = ["drivable_area", "road_segment", "ped_crossing", "walkway", "carpark_area"]
LINE_LAYERS = ["road_divider", "lane_divider"]
