#!/usr/bin/env python
"""Project ego-frame points into the nuScenes cameras.

Trajectories and navigation targets live in the ego frame at ground level (z = 0: the ego
origin sits on the road, which is why the front camera's calibrated height is ~1.5 m). To
draw them over a camera image they go through the sensor's extrinsics and then its intrinsics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VIEW_TO_CAM = {"<FRONT VIEW>": "CAM_FRONT", "<FRONT LEFT VIEW>": "CAM_FRONT_LEFT",
               "<FRONT RIGHT VIEW>": "CAM_FRONT_RIGHT"}


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class CameraRig:
    """Intrinsics and ego->camera extrinsics for each keyframe's forward cameras."""

    def __init__(self, root: str | Path, version: str = "v1.0-mini"):
        meta = Path(root) / version
        calibrated = {c["token"]: c
                      for c in json.loads((meta / "calibrated_sensor.json").read_text())}
        self.by_sample: dict[tuple[str, str], dict] = {}
        for sd in json.loads((meta / "sample_data.json").read_text()):
            if not sd["is_key_frame"]:
                continue
            parts = sd["filename"].split("/")
            if len(parts) != 3 or parts[1] not in VIEW_TO_CAM.values():
                continue
            calib = calibrated[sd["calibrated_sensor_token"]]
            sensor2ego = np.eye(4)
            sensor2ego[:3, :3] = quat_to_matrix(calib["rotation"])
            sensor2ego[:3, 3] = calib["translation"]
            self.by_sample[(sd["sample_token"], parts[1])] = {
                "K": np.asarray(calib["camera_intrinsic"], dtype=np.float64),
                "ego2cam": np.linalg.inv(sensor2ego),
                "size": (sd["width"], sd["height"]),
            }

    def has(self, sample_token: str) -> bool:
        return any((sample_token, cam) in self.by_sample for cam in VIEW_TO_CAM.values())

    def project(self, sample_token: str, view: str, points_xy: np.ndarray,
                height: float = 0.0) -> np.ndarray | None:
        """Ego-frame (x, y) at ``height`` as pixel coordinates, or None if unavailable.

        Points behind the camera are dropped before the divide, and the polyline is cut at
        the first gap so a track that leaves and re-enters view is not joined by a chord.
        """
        entry = self.by_sample.get((sample_token, VIEW_TO_CAM.get(view, "")))
        if entry is None or len(points_xy) == 0:
            return None
        points = np.column_stack([points_xy[:, 0], points_xy[:, 1],
                                  np.full(len(points_xy), height), np.ones(len(points_xy))])
        camera = points @ entry["ego2cam"].T
        in_front = camera[:, 2] > 0.5
        if not in_front.any():
            return None
        uv = (entry["K"] @ camera[:, :3].T).T
        uv = uv[:, :2] / uv[:, 2:3]
        uv[~in_front] = np.nan

        width, height_px = entry["size"]
        inside = ((uv[:, 0] > -width) & (uv[:, 0] < 2 * width)
                  & (uv[:, 1] > -height_px) & (uv[:, 1] < 2 * height_px))
        uv[~inside] = np.nan
        return uv

    def project_band(self, sample_token: str, view: str, points_xy: np.ndarray,
                     width: float = 1.6, height: float = 0.0) -> np.ndarray | None:
        """A ribbon of constant *metric* width around a path, as a fillable pixel polygon.

        Drawing the centreline with a fixed pixel width would make the far end as wide as
        the near end; projecting both edges instead lets perspective narrow it, so the
        target reads as something painted on the road.
        """
        if len(points_xy) < 2:
            return None
        offset = np.column_stack([np.zeros(len(points_xy)), np.full(len(points_xy), width / 2)])
        left = self.project(sample_token, view, points_xy + offset, height)
        right = self.project(sample_token, view, points_xy - offset, height)
        if left is None or right is None:
            return None
        good = ~(np.isnan(left).any(axis=1) | np.isnan(right).any(axis=1))
        if good.sum() < 2:
            return None
        return np.vstack([left[good], right[good][::-1]])
