#!/usr/bin/env python
"""Interactive Qwen-Drive-1.0 demo: plan under a navigation command you choose, and
ask the VLM questions about the scene or about your own images.

    source env.sh && python app.py            # then open http://127.0.0.1:7860

The model loads once at startup and stays resident; every control below is just a
different call into it.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in a headless/background session

import gradio as gr
import numpy as np
import torch

from qwen_drive import InferenceMode, QwenDriveForPlanning
from qwen_drive.benchmarks import read_scene_file
from qwen_drive.images import ImageArchive
from qwen_drive.scene import CAMERA_VIEWS, NAV_COMMANDS
from qwen_drive.visualize import plot_scene_summary

ROOT = Path(__file__).parent
import sys

sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "outputs" / "interactive"

# The four bundled scenes, in file order, described in the README.
SCENE_BLURBS = [
    "night intersection, light turns green",
    "left turn",
    "right turn",
    "slow down past a parked truck",
]

# driving_command is the one-hot [left, straight, right, unknown] the expert reads as
# part of ego_status; nav_command is the scalar that gets written into the prompt.
DRIVING_COMMAND = {0: [0.0, 1.0, 0.0, 0.0], 1: [1.0, 0.0, 0.0, 0.0], 2: [0.0, 0.0, 1.0, 0.0]}
# One colour per navigation command when all three are drawn together.
NAV_COLOURS = {0: "#1f77b4", 1: "#2ca02c", 2: "#ff7f0e"}
# A fourth choice alongside the three real commands, not a command itself.
COMPARE_ALL = "ALL THREE"

PLANNERS = {"planner-rl": "Qwen-Drive-1.0-4B/planner-rl", "planner-sft": "Qwen-Drive-1.0-4B/planner-sft"}
PERCEPTION_DIR = ROOT / "data" / "demo" / "perception"
PERCEPTION_FRAMES = sorted(d.name for d in PERCEPTION_DIR.iterdir() if d.is_dir())
# Perception frames packed from nuScenes by tools/nuscenes_perception_frames.py, keyed by
# the same sample token the planning scenes use, so both heads can run on one keyframe.
NS_PERCEPTION_DIR = ROOT / "data" / "nuscenes_perception"
# The map raster: 60 m x 30 m at 0.15 m, ego frame, X forward and Y left (docs/perception.md).
MAP_EXTENT = (15.0, -15.0, -30.0, 30.0)   # imshow (left, right, bottom, top) in plot coords

lock = threading.Lock()
state = {"planner": None}
perception = {}          # lazily built head + processor
gt_map = {}              # lazily built NuScenesMapGT
perception_cache = {}    # frame token -> inference result


def with_nav_command(scene, nav: int):
    """A copy of the scene planned under a different navigation command.

    Benchmark scenes carry their prompt verbatim in ``instruction_text``, so the command
    has to be rewritten there as well as in the two numeric fields.
    """
    text = scene.instruction_text
    if text is not None:
        text = re.sub(
            r"(Active navigation command: )\[[^\]]*\]",
            lambda m: f"{m.group(1)}[{NAV_COMMANDS[nav]}]",
            text,
        )
    return dataclasses.replace(
        scene, nav_command=nav, driving_command=DRIVING_COMMAND[nav], instruction_text=text
    )


def load_scenes(args):
    """The four bundled WOD-E2E scenes, plus any nuScenes scenes built by tools/."""
    samples = list(
        read_scene_file(
            args.scenes,
            image_archive=ImageArchive.open(args.image_archive),
            num_history_points=16,
        )
    )
    labels = [
        f"scene {i} - {SCENE_BLURBS[i] if i < len(SCENE_BLURBS) else s.token}"
        for i, s in enumerate(samples)
    ]

    extra = ROOT / args.nuscenes_scenes
    if extra.exists():
        turns = {0: "straight", 1: "left", 2: "right"}
        found = list(read_scene_file(extra, image_root=ROOT / args.nuscenes_root,
                                     num_history_points=16))
        for index, sample in enumerate(found):
            turn = turns[sample.scene.nav_command]
            meta = sample.scene.metadata
            marks = []
            if meta.get("lane_change"):                      # tools/nuscenes_lane_change.py
                marks.append("LANE CHANGE")
            # multi_lane already excludes junctions: a lane beside a junction path is not
            # somewhere the ego can move to, so the raw counts must not be advertised there.
            if meta.get("multi_lane"):                       # tools/nuscenes_multilane.py
                left, right = meta.get("lanes_left", 0), meta.get("lanes_right", 0)
                room = "".join(f"{n}{side}" for n, side in ((left, "L"), (right, "R")) if n)
                marks.append(f"{left + right + 1} lanes, room {room}")
            elif meta.get("in_junction"):
                marks.append("junction")
            mark = (", " + ", ".join(marks)) if marks else ""
            labels.append(
                f"nuscenes {index:03d} - {turn}, {sample.initial_speed:.1f} m/s{mark}"
            )
        samples.extend(found)
        print(f"loaded {len(found)} nuScenes scenes from {extra.name}")
    return samples, labels


def select_planner(name: str):
    if state["planner"] != name:
        model.load_planner(ROOT / PLANNERS[name])
        state["planner"] = name


def plan(scene_label, nav_label, mode_label, planner_name, num_samples, seed, map_choice,
         progress=gr.Progress()):
    index = LABELS.index(scene_label)
    sample = SAMPLES[index]
    mode = (
        InferenceMode.REASONING_PLANNING
        if mode_label.startswith("reasoning")
        else InferenceMode.DIRECT_PLANNING
    )
    compare_all = nav_label == COMPARE_ALL
    commands = [0, 1, 2] if compare_all else [NAV_COMMANDS.index(nav_label)]

    results = {}
    map_grid, map_geoms, map_note = None, None, ""
    with lock:
        if map_choice == "predicted":
            frame_dir = NS_PERCEPTION_DIR / sample.token
            if frame_dir.is_dir():
                progress(0.05, desc="running the BEV head")
                map_grid = perception_result(frame_dir)["map"]
                map_note = "\n\nUnder the trajectory: the **predicted** BEV map from the perception head."
            else:
                map_note = (
                    "\n\n⚠️ No predicted map: this scene has no packed perception frame. "
                    "Only the nuScenes scenes have one, because the BEV head needs the full "
                    "camera ring and calibration, which the WOD-E2E demo scenes do not carry."
                )
        elif map_choice.startswith("ground truth"):
            as_vector = "vector" in map_choice
            progress(0.05, desc="reading the nuScenes map")
            if as_vector:
                map_geoms = vector_map(sample.token)
                found = map_geoms is not None
            else:
                map_grid = ground_truth_map(sample.token)
                found = map_grid is not None
            if found:
                form = "vector geometry" if as_vector else "raster"
                map_note = (
                    f"\n\nUnder the trajectory: the **ground-truth** map from the nuScenes map "
                    f"expansion, as {form}. nuScenes has no road-edge layer, so that is the "
                    f"drivable-area boundary."
                )
            else:
                map_note = (
                    "\n\n⚠️ No ground-truth map: needs a nuScenes scene and "
                    "`data/nuscenes/maps/expansion/` from the map-expansion pack."
                )
        progress(0.05, desc=f"loading {planner_name}")
        select_planner(planner_name)
        for step, nav in enumerate(commands):
            progress(0.1 + 0.85 * step / len(commands), desc=f"planning: {NAV_COMMANDS[nav]}")
            scene = with_nav_command(sample.scene, nav)
            results[nav] = (
                scene,
                model.run(mode, scene=scene, num_samples=int(num_samples), seed=int(seed)),
            )

    OUT.mkdir(parents=True, exist_ok=True)
    recorded = sample.scene.nav_command
    gt = sample.future_trajectory

    if compare_all:
        path = OUT / f"plan_{index}_all.png"
        overlays = [
            (NAV_COMMANDS[nav], results[nav][1].trajectories, NAV_COLOURS[nav]) for nav in commands
        ]
        scene = results[recorded if recorded in results else commands[0]][0]
        save_plan_figure(
            path, map_grid=map_grid, map_geoms=map_geoms, scene=scene,
            trajectories=results[commands[0]][1].trajectories,
            history=scene.history, ground_truth=gt, reasoning=None,
            title=f"{sample.token} ({mode.value})\nall navigation commands",
            overlays=overlays,
        )
        reasoning = "\n".join(
            f"{NAV_COMMANDS[nav]}: {results[nav][1].reasoning or '(direct planning: no trace)'}"
            for nav in commands
        )
        lines = [
            f"**{int(num_samples)}** trajectories per command, "
            f"**{results[commands[0]][1].trajectories.shape[1]}** points each (5 s @ 10 Hz)"
        ]
        if gt is not None:
            scores = []
            for nav in commands:
                traj = results[nav][1].trajectory
                error = np.linalg.norm(traj[:, :2] - gt[: len(traj), :2], axis=-1)
                mark = " ← recorded" if nav == recorded else ""
                scores.append(
                    f"| {NAV_COMMANDS[nav]} | {error.mean():.3f} | {error[-1]:.3f} | "
                    f"{np.round(traj[-1, 1], 2)} |{mark}"
                )
            lines.append(
                "| command | ADE (m) | FDE (m) | endpoint y (m) | |\n|---|---|---|---|---|\n"
                + "\n".join(scores)
            )
            lines.append(
                f"Only **{NAV_COMMANDS[recorded]}** is scored fairly: the recorded future was "
                f"driven under it, so the other two rows are the cost of departing from it."
            )
        return str(path), reasoning, "\n\n".join(lines) + map_note

    nav = commands[0]
    scene, result = results[nav]
    path = OUT / f"plan_{index}_{nav}.png"
    save_plan_figure(
        path, map_grid=map_grid, map_geoms=map_geoms, scene=scene,
        trajectories=result.trajectories,
        history=scene.history, ground_truth=gt, reasoning=result.reasoning,
        title=f"{sample.token} ({mode.value})",
    )
    end = np.round(result.trajectory[-1], 2).tolist()
    lines = [
        f"**endpoint of sample 0** (x fwd, y left, heading): `{end}` &nbsp;·&nbsp; "
        f"**{result.trajectories.shape[0]}** trajectories x **{result.trajectories.shape[1]}** "
        f"points (5 s @ 10 Hz)",
    ]
    if gt is not None:
        error = np.linalg.norm(result.trajectory[:, :2] - gt[: len(result.trajectory), :2], axis=-1)
        note = ""
        if nav != recorded:
            note = (
                f" &nbsp;·&nbsp; ⚠️ the recorded future was driven under "
                f"**{NAV_COMMANDS[recorded]}**, so these are not a fair score "
                f"for **{NAV_COMMANDS[nav]}**"
            )
        lines.append(f"**ADE** {error.mean():.3f} m &nbsp;·&nbsp; **FDE** {error[-1]:.3f} m{note}")
    return (str(path), result.reasoning or "(direct planning: no reasoning trace)",
            "\n\n".join(lines) + map_note)


def ask_scene(scene_label, question, which_images):
    if not question.strip():
        return "Type a question first."
    sample = SAMPLES[LABELS.index(scene_label)]
    scene = sample.scene
    if which_images.startswith("front view"):
        frames = list(scene.views[CAMERA_VIEWS[0]])
    elif which_images.startswith("current"):
        frames = [scene.views[view][-1] for view in CAMERA_VIEWS]
    else:
        frames = scene.frames_in_order()
    with lock:
        result = model.run(InferenceMode.VQA, images=frames, question=question)
    return result.text


def ask_images(images, question):
    if not images:
        return "Upload at least one image."
    if not question.strip():
        return "Type a question first."
    from PIL import Image

    loaded = [Image.open(item[0] if isinstance(item, (list, tuple)) else item).convert("RGB") for item in images]
    with lock:
        result = model.generate_text(images=loaded, question=question)
    return result.text


def camera_ring(scene_label):
    """The scene's 12 frames, front row first, as a gallery."""
    scene = SAMPLES[LABELS.index(scene_label)].scene
    items = []
    for view in CAMERA_VIEWS:
        label = view.strip("<>").replace(" VIEW", "").title()
        for step, frame in enumerate(scene.views[view]):
            items.append((frame.load(), f"{label}  t-{(len(scene.views[view]) - 1 - step) * 0.5:.1f}s"))
    return items


# ---------------------------------------------------------------------------
# Perception frames, and planning on top of one.
#
# A perception frame is a single timestep of the full camera ring plus calibration.
# It carries no ego state at all, so planning on one means inventing the ego history:
# see the note rendered in the tab.
# ---------------------------------------------------------------------------


def get_perception():
    """Attach the BEV head to the VLM that is already resident. Built on first use."""
    if not perception:
        from transformers import AutoTokenizer

        from qwen_drive_perception import QwenDrivePerception
        from qwen_drive_perception.dataset import PerceptionProcessor

        head = QwenDrivePerception.from_pretrained(
            ROOT / "Qwen-Drive-1.0-4B" / "perception", dtype=torch.bfloat16
        )
        head.to(model.device).eval()
        processor = PerceptionProcessor(AutoTokenizer.from_pretrained(ROOT / "Qwen-Drive-1.0-4B"))
        head.attach(model.vlm, processor)
        perception.update(head=head, processor=processor)
    return perception["head"], perception["processor"]


def view_to_camera(frame) -> dict[str, str]:
    """The frame labels its own cameras with the same view names planning uses."""
    mapping, current = {}, None
    for item in frame.content:
        if "text" in item:
            current = item["text"]
        elif "image" in item and current is not None:
            mapping[current] = item["image"]
    return mapping


def perception_result(frame_dir: Path) -> dict:
    """Run the BEV head on one packed frame, once. Caller holds the lock."""
    from qwen_drive_perception.dataset import PerceptionFrame

    key = str(frame_dir)
    if key not in perception_cache:
        head, processor = get_perception()
        inputs, img_metas = processor(PerceptionFrame(frame_dir), device=model.device)
        perception_cache[key] = head.infer(inputs, img_metas)
    return perception_cache[key]


def map_raster_rgb(grid: np.ndarray) -> np.ndarray:
    """The (Y, X) map raster as an image with ego-forward up and ego-left left."""
    from qwen_drive_perception.configuration_perception import MAP_PALETTE

    palette = np.array(MAP_PALETTE, dtype=np.uint8)
    labels = np.clip(np.asarray(grid, dtype=np.int64), 0, len(palette) - 1)
    return palette[labels].transpose(1, 0, 2)[::-1, ::-1]


def _map_helper():
    """The NuScenesMapGT wrapper, built on first use, or None without the expansion pack."""
    if "helper" not in gt_map:
        from nuscenes_map_gt import NuScenesMapGT

        helper = NuScenesMapGT(ROOT / "data" / "nuscenes")
        gt_map["helper"] = helper if helper.available() else None
    return gt_map["helper"]


def ground_truth_map(token: str):
    """The nuScenes map-expansion raster for a keyframe, or None if unavailable."""
    helper = _map_helper()
    if helper is None or token not in helper.pose:
        return None
    key = f"gt:{token}"
    if key not in perception_cache:
        perception_cache[key] = helper.raster(token)
    return perception_cache[key]


def vector_map(token: str):
    """Ego-frame vector geometry for a keyframe, or None if unavailable."""
    helper = _map_helper()
    if helper is None or token not in helper.pose:
        return None
    key = f"vec:{token}"
    if key not in perception_cache:
        perception_cache[key] = helper.vector(token)
    return perception_cache[key]


def draw_vector_map(axis) -> list:
    """Return the (drawer, colour, label) plan for the vector map, painted back to front."""
    from qwen_drive_perception.configuration_perception import MAP_PALETTE

    rgb = [tuple(v / 255 for v in c) for c in MAP_PALETTE]
    #                layer            colour        kind     label
    return [("drivable_area", rgb[1], "fill", "driveable surface"),
            ("walkway", rgb[5], "fill", "walkway"),
            ("ped_crossing", rgb[4], "fill", "crosswalk"),
            ("lane_centreline", (0.45, 0.45, 0.45), "dotted", "lane centreline"),
            ("lane_divider", rgb[2], "dashed", "lane divider"),
            ("road_divider", rgb[3], "solid", "road divider")]


def _inside_patch(loop, margin: float = 0.4):
    """Split a clipped ring into the runs that are genuinely inside the patch.

    MAP_EXTENT is (left, right, bottom, top) in plot coordinates, so the patch spans
    +/-15 m laterally and +/-30 m longitudinally; points sitting on that box are the cut,
    not real geometry.
    """
    # The patch is whatever the geometry was clipped to; read it off the ring itself so a
    # larger map request does not lose its road edges beyond the default 30 m.
    half_longitudinal = max(abs(MAP_EXTENT[2]), float(np.abs(loop[:, 0]).max())) - margin
    half_lateral = max(abs(MAP_EXTENT[0]), float(np.abs(loop[:, 1]).max())) - margin
    inside = ((np.abs(loop[:, 0]) < half_longitudinal)
              & (np.abs(loop[:, 1]) < half_lateral))
    padded = np.concatenate([[False], inside, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [loop[start:stop] for start, stop in zip(edges[::2], edges[1::2])
            if stop - start > 1]


def paint_vector_map(axis, geoms) -> tuple[list, list]:
    """Draw the vector map under the trajectories. Returns extra legend handles/labels."""
    import matplotlib.lines as mlines
    import matplotlib.patches as patches
    from matplotlib.path import Path as MplPath
    from qwen_drive_perception.configuration_perception import MAP_PALETTE

    handles, labels = [], []
    for layer, colour, kind, label in draw_vector_map(axis):
        items = geoms.get(layer) or []
        if not items:
            continue
        if kind == "fill":
            for ring in items:
                # Build one path per polygon so interior rings punch real holes.
                vertices, codes = [], []
                for loop in [ring["exterior"], *ring["holes"]]:
                    vertices.extend(loop[:, ::-1])          # (x fwd, y left) -> (plot x, y)
                    codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(loop) - 2)
                                 + [MplPath.CLOSEPOLY])
                axis.add_patch(patches.PathPatch(
                    MplPath(vertices, codes), facecolor=colour, edgecolor="none", zorder=0))
            handles.append(patches.Patch(color=colour))
        else:
            style = {"solid": "-", "dashed": (0, (6, 4)), "dotted": (0, (1, 3))}[kind]
            width = 0.8 if kind == "dotted" else 1.4
            for line in items:
                axis.plot(line[:, 1], line[:, 0], color=colour, linestyle=style,
                          linewidth=width, zorder=1, alpha=0.9)
            handles.append(mlines.Line2D([], [], color=colour, linestyle=style, linewidth=width))
        labels.append(label)

    # nuScenes has no road-edge layer, so the drivable boundary stands in for one. The
    # polygon was clipped to the patch, though, and the cut runs along the patch border -
    # drawing that would put a road edge straight across the road ahead of the ego.
    edge = tuple(v / 255 for v in MAP_PALETTE[3])
    for ring in geoms.get("drivable_area") or []:
        for loop in [ring["exterior"], *ring["holes"]]:
            for piece in _inside_patch(loop):
                axis.plot(piece[:, 1], piece[:, 0], color=edge, linewidth=1.6, zorder=1)
    if geoms.get("drivable_area"):
        handles.append(mlines.Line2D([], [], color=edge, linewidth=1.6))
        labels.append("road edge (drivable boundary)")
    return handles, labels


def _floor_longitudinal_span(figure, minimum: float = 20.0) -> None:
    """Keep the trajectory panel readable when the ego is stopped.

    A stationary scene has every point at x ~ 0, so the automatic limits collapse to a
    sliver and the equal aspect ratio squashes the panel into a line.
    """
    axis = figure.axes[-1]
    low, high = axis.get_ylim()
    if high - low < minimum:
        centre = 0.5 * (low + high)
        axis.set_ylim(centre - minimum / 2, centre + minimum / 2)


def _draw_on_montage(figure, scene, overlays) -> None:
    """Draw pixel-space polylines over the current frame of each camera view.

    plot_scene_summary builds the montage row by row and adds the trajectory panel last, so
    a view's current frame is at ``row * num_frames + (num_frames - 1)``.
    """
    if scene is None:
        return
    columns = scene.num_camera_frames
    for row, view in enumerate(CAMERA_VIEWS):
        index = row * columns + (columns - 1)
        if view not in overlays or index >= len(figure.axes) - 1:
            continue
        axis = figure.axes[index]
        # Plotting autoscales, and a projected path runs well outside the frame - the point
        # at the ego's own bumper lands ~600 px below it. Pin the axes back to the image so
        # the picture keeps its size and everything outside is simply clipped away.
        xlim, ylim = axis.get_xlim(), axis.get_ylim()
        for item in overlays[view]:
            if item["kind"] == "band":
                axis.fill(item["uv"][:, 0], item["uv"][:, 1], color=item["colour"],
                          alpha=item["alpha"], linewidth=0, zorder=4)
            else:
                axis.plot(item["uv"][:, 0], item["uv"][:, 1], color=item["colour"],
                          linewidth=item["width"], alpha=item["alpha"],
                          solid_capstyle="round", zorder=5)
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_autoscale_on(False)


def save_plan_figure(path, map_grid=None, map_geoms=None, noodles=None,
                     camera_overlays=None, **kwargs) -> None:
    """plot_scene_summary, optionally with a BEV map drawn underneath.

    ``map_grid`` paints a raster, ``map_geoms`` draws vector geometry, and ``noodles`` are
    thick translucent polylines drawn under the trajectories, for navigation targets.
    ``camera_overlays`` maps a view name to polylines already in pixel coordinates, drawn
    over that view's current frame in the montage.
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    figure = plot_scene_summary(output=None, **kwargs)
    if map_geoms is not None:
        axis = figure.axes[-1]
        xlim, ylim = axis.get_xlim(), axis.get_ylim()
        extra_handles, extra_labels = paint_vector_map(axis, map_geoms)
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        handles, labels = axis.get_legend_handles_labels()
        seen = set(labels)
        for handle, label in zip(extra_handles, extra_labels):
            if label not in seen:
                handles.append(handle)
                labels.append(label)
        axis.legend(handles, labels, loc="lower left", fontsize=7, framealpha=0.92)
    if map_grid is not None:
        axis = figure.axes[-1]                       # the trajectory panel is added last
        xlim, ylim = axis.get_xlim(), axis.get_ylim()
        axis.imshow(map_raster_rgb(map_grid), extent=MAP_EXTENT, origin="upper",
                    zorder=0, interpolation="nearest")
        axis.set_xlim(xlim)                          # the raster is wider than the plot
        axis.set_ylim(ylim)
        axis.set_aspect("equal")

        from qwen_drive_perception.configuration_perception import MAP_CLASS_NAMES, MAP_PALETTE

        handles, labels = axis.get_legend_handles_labels()
        present = [c for c in np.unique(np.asarray(map_grid)) if c != 0]
        for index in present:
            colour = tuple(v / 255 for v in MAP_PALETTE[int(index)])
            handles.append(patches.Patch(color=colour))
            labels.append(MAP_CLASS_NAMES[int(index)].replace("_", " "))
        # Ahead of the ego is the part worth seeing, so the bigger legend goes behind it.
        axis.legend(handles, labels, loc="lower left", fontsize=7, framealpha=0.92)

    if noodles:
        import matplotlib.lines as mlines

        axis = figure.axes[-1]
        handles, labels = axis.get_legend_handles_labels()
        for label, polyline, colour in noodles:
            axis.plot(polyline[:, 1], polyline[:, 0], color=colour, linewidth=13,
                      alpha=0.30, solid_capstyle="round", solid_joinstyle="round",
                      zorder=0.6)
            handles.append(mlines.Line2D([], [], color=colour, linewidth=7, alpha=0.45))
            labels.append(label)
        axis.legend(handles, labels, loc="lower left", fontsize=7, framealpha=0.92)

    if camera_overlays:
        _draw_on_montage(figure, kwargs.get("scene"), camera_overlays)
    _floor_longitudinal_span(figure)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_perception_frame(token, score_thr, progress=gr.Progress()):
    """Perception only. Planning is not offered here: a perception frame carries no ego
    state, and planning without a recorded ego history is not worth looking at."""
    from qwen_drive_perception.dataset import PerceptionFrame
    from qwen_drive_perception.visualize import render_frame

    frame = PerceptionFrame(PERCEPTION_DIR / token)
    with lock:
        progress(0.2, desc="running the BEV head")
        result = perception_result(PERCEPTION_DIR / token)

    summary = render_frame(frame, result, score_threshold=float(score_thr))[:, :, ::-1]
    kept = int((result["scores"] >= float(score_thr)).sum())
    info = (
        f"**{frame.dataset_type}** frame, {len(frame.cam_order)} cameras &nbsp;·&nbsp; "
        f"**{kept}** boxes above {float(score_thr):.2f} &nbsp;·&nbsp; occupancy "
        f"`{result['occ'].shape}` &nbsp;·&nbsp; map `{result['map'].shape}`"
    )
    return summary, info


# ---------------------------------------------------------------------------
# Trion mock: Trion-Reason (symbols) -> Resolver (HD map) -> Trion-Action (trajectory).
# No neural network runs. Only the resolver is real; the two systems are rule-based
# stand-ins so the interface between them can be exercised and shown.
# ---------------------------------------------------------------------------

TRION_SCENES = ("nuscenes 147", "nuscenes 186", "nuscenes 199", "nuscenes 203")
TRION_COMMANDS = ("keep going", "change to the right lane", "change to the left lane",
                  "pull over", "go faster", "go slower")


def camera_rig():
    """Calibration for the nuScenes forward cameras, built on first use."""
    if "rig" not in gt_map:
        try:
            from nuscenes_cameras import CameraRig

            gt_map["rig"] = CameraRig(ROOT / "data" / "nuscenes")
        except (FileNotFoundError, KeyError):
            gt_map["rig"] = None
    return gt_map["rig"]


def trion_scene_labels() -> list[str]:
    return [label for label in LABELS if any(label.startswith(s + " ") for s in TRION_SCENES)]


def trion_run(scene_label, command_choice, custom_command, preference, progress=gr.Progress()):
    from trion import Resolver, act, reason

    command = (custom_command or "").strip() or command_choice
    index = LABELS.index(scene_label)
    sample = SAMPLES[index]
    token = sample.scene.metadata.get("token", "")
    speed = sample.initial_speed
    helper = _map_helper()
    if helper is None or token not in helper.pose:
        return None, "needs the nuScenes map expansion in data/nuscenes/maps/expansion/"

    progress(0.2, desc="Trion-Reason")
    pref = {"slower": -1, "normal": 0, "faster": 1}[preference]
    msg = reason(command, pref, {"speed": speed})

    progress(0.4, desc="Resolver")
    if "resolver" not in gt_map:
        gt_map["resolver"] = Resolver(helper)
    geometry = helper.vector(token, half_length=70.0, half_width=25.0)
    resolved = gt_map["resolver"].resolve(token, msg, geometry, speed=speed)

    progress(0.7, desc="Trion-Action")
    trajectory, summary = act(resolved, speed)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"trion_{index}_{resolved.requested.lower()}.png"
    trion_figure(out_path, sample, command, msg, resolved, trajectory, summary,
                 helper.vector(token, half_length=95.0, half_width=14.0), camera_rig(), token)

    ok = "".join("✓" if ok else "✗" for _, ok, _ in resolved.checks)
    verdict = ("**rejected → fallback KEEP**" if resolved.fallback else "**accepted**")
    lines = [
        f"**Trion-Reason** → `{msg.lateral}`, cruise `{msg.cruise_label}`"
        + (f", window {msg.window_m[0]:.0f}–{msg.window_m[1]:.0f} m" if msg.lateral != "KEEP" else "")
        + f"  \n_{msg.why}_",
        f"**Resolver** → {verdict} &nbsp;`{ok}`  \n"
        + "  \n".join(f"{'✓' if ok else '✗'} {name} — {note}" for name, ok, note in resolved.checks),
        f"**Trion-Action** → drives `{resolved.lateral}`: reaches {summary['reach_m']:.0f} m in 5 s, "
        f"lateral {summary['lateral_move_m']:+.1f} m, ends at {summary['final_speed']:.1f} m/s "
        f"(cap {resolved.speed_cap:.1f})"
        + (f", commits to the change at {summary['commit_at_m']:.0f} m" if summary["commit_at_m"] else ""),
    ]
    return str(out_path), "\n\n".join(lines)


def _card(axis, title, body, y, colour="#2f3437", width=44):
    import textwrap

    axis.text(0.0, y, title, transform=axis.transAxes, fontsize=9.5, fontweight="bold",
              color=colour, va="top", family="monospace")
    lines = []
    for paragraph in body:
        lines.extend(textwrap.wrap(paragraph, width) or [""])
    axis.text(0.0, y - 0.035, "\n".join(lines), transform=axis.transAxes, fontsize=7.6,
              va="top", family="monospace", color="#2f3437", linespacing=1.25)
    return y - 0.035 - 0.0215 * (len(lines) + 1.2)


def trion_figure(path, sample, command, msg, res, traj, summary, map_geoms, rig, token) -> None:
    """Story left to right: the three stages as cards, the front camera, the map."""
    import matplotlib.lines as mlines
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from qwen_drive.visualize import _draw_trajectories

    scene = sample.scene
    figure = plt.figure(figsize=(21.0, 7.4))
    outer = figure.add_gridspec(1, 3, width_ratios=[0.92, 1.85, 1.25], wspace=0.05)

    # -- column 1: the pipeline as cards ----------------------------------------------
    card = figure.add_subplot(outer[0])
    card.set_axis_off()
    y = 0.99
    y = _card(card, "INPUT", [f'Scene: {sample.token[:8]}  ego {sample.initial_speed:.1f} m/s',
                              f'Voice: "{command}"'], y)
    y = _card(card, "TRION-REASON  (slow, symbolic)", [
        f"lateral   {msg.lateral}" + (f"   window {msg.window_m[0]:.0f}-{msg.window_m[1]:.0f} m,"
                                     f" deadline {msg.deadline_m:.0f} m" if msg.lateral != "KEEP" else ""),
        f"cruise    {msg.cruise_label}   ({SPEED_KMH(res.speed_cap)})",
        f"stop      {msg.planned_stop or '-'}",
        f"light     {msg.which_light}",
        "", *textwrap_lines(msg.why)], y, colour="#7b3fa0")
    checks = [f"{'✓' if ok else '✗'} {name}: {note}" for name, ok, note in res.checks]
    verdict = "REJECTED -> fallback KEEP" if res.fallback else f"accepted: drive {res.lateral}"
    y = _card(card, "RESOLVER  (HD map, deterministic)", [*checks, "", verdict], y, colour="#1f5f8b")
    action = [f"reach      {summary['reach_m']:.0f} m in 5 s",
              f"lateral    {summary['lateral_move_m']:+.1f} m",
              f"speed      {sample.initial_speed:.1f} -> {summary['final_speed']:.1f} m/s (cap {res.speed_cap:.1f})"]
    if summary["commit_at_m"]:
        action.append(f"commits at {summary['commit_at_m']:.0f} m  (gap accepted, mocked)")
    if res.stop_x is not None:
        action.append(f"stops at   {res.stop_x:.0f} m")
    _card(card, "TRION-ACTION  (fast, geometric)", action, y, colour="#b5651d")

    # -- column 2: front camera ------------------------------------------------------
    hero = figure.add_subplot(outer[1])
    hero.set_xticks([])
    hero.set_yticks([])
    hero.imshow(scene.views[CAMERA_VIEWS[0]][-1].load())
    hero.set_title(f"Trion-Action on the front camera  -  {res.lateral}"
                   + ("  (request rejected)" if res.fallback else ""), fontsize=10, color="#2f3437")
    xlim, ylim = hero.get_xlim(), hero.get_ylim()
    if rig is not None and rig.has(token):
        s_min = res.window_m[0] if res.lateral != "KEEP" else 10.0
        ahead = res.target_xy[res.target_xy[:, 0] >= s_min]
        band = rig.project_band(token, CAMERA_VIEWS[0], ahead, width=1.1)
        if band is not None:
            hero.fill(band[:, 0], band[:, 1], color=res.colour, alpha=0.32, linewidth=0, zorder=4)
        uv = rig.project(token, CAMERA_VIEWS[0], traj[:, :2])
        if uv is not None:
            hero.plot(uv[:, 0], uv[:, 1], color="#111111", linewidth=3.6, alpha=0.95,
                      solid_capstyle="round", zorder=5)
            hero.plot(uv[:, 0], uv[:, 1], color=res.colour, linewidth=2.0, alpha=1.0,
                      solid_capstyle="round", zorder=6)
    hero.set_xlim(xlim)
    hero.set_ylim(ylim)

    # -- column 3: bird's-eye --------------------------------------------------------
    axis = figure.add_subplot(outer[2])
    _draw_trajectories(axis, traj[None], scene.history, None, 20.0,
                       [(f"Trion-Action ({res.lateral})", traj[None], res.colour)])
    handles, labels = axis.get_legend_handles_labels()
    paint_vector_map(axis, map_geoms)                 # context only; its legend is noise here
    # Show the manoeuvre, not the resolver's 200 m horizon: the window, the stop and the
    # plan set the range, and the lateral axis is stretched so lanes are readable.
    top = max(res.window_m[1], summary["reach_m"], res.stop_x or 0.0, 45.0) + 15.0
    cur = res.current_xy[(res.current_xy[:, 0] >= 0) & (res.current_xy[:, 0] <= top)]
    axis.plot(cur[:, 1], cur[:, 0], color="0.45", linewidth=10, alpha=0.18,
              solid_capstyle="round", zorder=0.5)
    handles.append(mlines.Line2D([], [], color="0.45", linewidth=6, alpha=0.3))
    labels.append("current lane corridor")
    if res.lateral != "KEEP":
        tgt = res.target_xy[(res.target_xy[:, 0] >= res.window_m[0]) & (res.target_xy[:, 0] <= top)]
        axis.plot(tgt[:, 1], tgt[:, 0], color=res.colour, linewidth=12, alpha=0.28,
                  solid_capstyle="round", zorder=0.6)
        handles.append(mlines.Line2D([], [], color=res.colour, linewidth=7, alpha=0.45))
        labels.append(f"target: resolver ({res.requested})")
        markers = [(res.window_m[0], "window start")]
        if res.stop_x is None or abs(res.stop_x - res.window_m[1]) > 1.0:
            markers.append((res.window_m[1], "window end"))
        for s, name in markers:
            axis.axhline(s, color=res.colour, linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.8)
            axis.text(11.6, s, f" {name} {s:.0f} m", fontsize=7, va="bottom", ha="left",
                      color=res.colour)
        if res.deadline_m and res.deadline_m < top:
            axis.axhline(res.deadline_m, color="#c0392b", linestyle=":", linewidth=1.0)
            axis.text(11.6, res.deadline_m, f" deadline {res.deadline_m:.0f} m",
                      fontsize=7, va="bottom", ha="left", color="#c0392b")
    if res.stop_x is not None:
        axis.axhline(res.stop_x, color=res.colour, linestyle="-", linewidth=1.2)
        axis.text(11.6, res.stop_x, f" stop {res.stop_x:.0f} m", fontsize=7,
                  va="bottom", ha="left", color=res.colour)
    axis.set_aspect("auto")
    axis.set_xlim(12.0, -12.0)                        # left positive, drawn on the left
    axis.set_ylim(-12.0, top)
    axis.set_title("Resolver corridors + Trion-Action plan  (lateral stretched)", fontsize=10)
    axis.legend(handles, labels, loc="upper left", fontsize=7, framealpha=0.92)

    figure.subplots_adjust(left=0.015, right=0.99, top=0.93, bottom=0.06)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def SPEED_KMH(mps: float) -> str:
    return f"{mps * 3.6:.0f} km/h"


def textwrap_lines(text: str, width: int = 44) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width)


def build_ui():
    with gr.Blocks(title="Qwen-Drive-1.0") as ui:
        gr.Markdown(
            "# Qwen-Drive-1.0 &nbsp;·&nbsp; interactive demo\n"
            "One VLM, three inference modes. Pick a navigation command and plan, or just "
            "ask the model about the scene."
        )
        with gr.Tab("Planning"):
            gr.Markdown(
                "`scene N` are the four bundled WOD-E2E scenes. `nuscenes NNN` are built "
                "from nuScenes mini keyframes by `tools/nuscenes_to_scenes.py`, with real "
                "10 Hz ego history and a real 5 s future, so ADE/FDE are meaningful."
            )
            with gr.Row():
                scene_dd = gr.Dropdown(LABELS, value=LABELS[0], label="Scene", scale=3)
                nav_rd = gr.Radio(
                    list(NAV_COMMANDS) + [COMPARE_ALL],
                    value=NAV_COMMANDS[0],
                    label="Navigation command",
                    info=(
                        "rewritten into the prompt and into the expert's ego_status; "
                        f"{COMPARE_ALL} plans once per command and overlays them, 3x slower"
                    ),
                    scale=4,
                )
            with gr.Row():
                mode_rd = gr.Radio(
                    ["reasoning planning", "direct planning"],
                    value="reasoning planning",
                    label="Mode",
                    info="only reasoning planning produces a chain of thought",
                )
                planner_rd = gr.Radio(
                    list(PLANNERS),
                    value="planner-rl",
                    label="Planning Expert",
                    info="planner-rl was reward-optimized for reasoning planning only",
                )
                samples_sl = gr.Slider(1, 12, value=6, step=1, label="Trajectories to sample")
                seed_nb = gr.Number(value=42, precision=0, label="Noise seed")
            map_rd = gr.Radio(
                ["none", "predicted", "ground truth (vector)", "ground truth (raster)"],
                value="none",
                label="Map under the trajectory",
                info="nuScenes scenes only. predicted = BEV head on the same keyframe; "
                     "ground truth = nuScenes map expansion, drawn as vector geometry "
                     "(with lane centrelines) or as the rasterized grid",
            )
            plan_btn = gr.Button("Plan", variant="primary")
            plot_im = gr.Image(label="Camera ring and predicted trajectories", type="filepath")
            reason_tb = gr.Textbox(label="Chain of thought", lines=3)
            metrics_md = gr.Markdown()
            plan_btn.click(
                plan,
                [scene_dd, nav_rd, mode_rd, planner_rd, samples_sl, seed_nb, map_rd],
                [plot_im, reason_tb, metrics_md],
                api_name="plan",
            )

        with gr.Tab("Ask about the scene"):
            gr.Markdown(
                "`InferenceMode.VQA` — the VLM alone, over this scene's camera frames. "
                "The ego history and navigation command are **not** part of this prompt; "
                "only the images are."
            )
            with gr.Row():
                vqa_scene_dd = gr.Dropdown(LABELS, value=LABELS[0], label="Scene", scale=2)
                which_rd = gr.Radio(
                    ["all 12 frames", "front view only (4 frames)", "current frame, 3 views"],
                    value="all 12 frames",
                    label="Images to show the model",
                    scale=3,
                )
            question_tb = gr.Textbox(
                label="Question",
                value="What is the state of the traffic light ahead, and is it safe to proceed?",
            )
            ask_btn = gr.Button("Ask", variant="primary")
            answer_tb = gr.Textbox(label="Answer", lines=10)
            gallery = gr.Gallery(label="Frames sent to the model", columns=4, height=340)
            ask_btn.click(ask_scene, [vqa_scene_dd, question_tb, which_rd], answer_tb, api_name="ask_scene")
            vqa_scene_dd.change(camera_ring, vqa_scene_dd, gallery)
            ui.load(camera_ring, vqa_scene_dd, gallery)

        with gr.Tab("Perception frame"):
            gr.Markdown(
                "3D detection, occupancy and BEV map segmentation on the six bundled frames. "
                "The BEV head and the Planning Expert are **siblings** on one VLM, not a "
                "pipeline — the planner never reads the boxes or the map.\n\n"
                "Planning is not offered on these frames. A perception frame is a single "
                "timestep of the camera ring and carries no ego state, and planning from a "
                "synthesized ego history measured ADE 9.5 m against 1.9 m from a recorded "
                "one. Use the Planning tab, where every scene has real ego motion."
            )
            with gr.Row():
                frame_dd = gr.Dropdown(
                    PERCEPTION_FRAMES, value=PERCEPTION_FRAMES[0], label="Frame", scale=3
                )
                thr_sl = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Box score threshold")
            pf_btn = gr.Button("Run perception", variant="primary")
            pf_info = gr.Markdown()
            pf_vis = gr.Image(label="3D boxes, occupancy, BEV map", type="numpy")
            pf_btn.click(
                run_perception_frame, [frame_dd, thr_sl], [pf_vis, pf_info],
                api_name="run_perception_frame",
            )

        with gr.Tab("Trion mock"):
            gr.Markdown(
                "**Two-system stack, mocked end to end. No neural network runs.** "
                "A voice command goes to **Trion-Reason** (slow, symbolic: lane goal + window "
                "+ speed preference), the **Resolver** turns the symbols into lane corridors "
                "from the HD map and validates them, and **Trion-Action** (fast, geometric) "
                "chooses when inside the window and draws the manoeuvre. The resolver is real "
                "nuScenes map logic; the two systems are rule-based stand-ins for the models."
            )
            with gr.Row():
                trion_scene = gr.Dropdown(trion_scene_labels(), value=trion_scene_labels()[0],
                                          label="Scene (multi-lane, ego in the kerb lane)", scale=3)
                trion_pref = gr.Radio(["slower", "normal", "faster"], value="normal",
                                      label="Speed preference", scale=1)
            with gr.Row():
                trion_cmd = gr.Dropdown(list(TRION_COMMANDS), value=TRION_COMMANDS[1],
                                        label="Voice command", scale=2)
                trion_custom = gr.Textbox(label="...or type one", placeholder="e.g. pull over and stop",
                                          scale=3)
            trion_btn = gr.Button("Run Trion (mock)", variant="primary")
            trion_im = gr.Image(label="Input → Trion-Reason → Resolver → Trion-Action", type="filepath")
            trion_md = gr.Markdown()
            trion_btn.click(trion_run, [trion_scene, trion_cmd, trion_custom, trion_pref],
                            [trion_im, trion_md], api_name="trion")

        with gr.Tab("General VQA"):
            gr.Markdown(
                "`generate_text` — the same VLM on **any** images, nothing driving-specific. "
                "This is the path the MMBench / MMMU / RealWorldQA numbers come from."
            )
            up = gr.File(file_count="multiple", file_types=["image"], label="Images")
            gen_q = gr.Textbox(label="Question", value="Describe this image.")
            gen_btn = gr.Button("Ask", variant="primary")
            gen_a = gr.Textbox(label="Answer", lines=12)
            gen_btn.click(ask_images, [up, gen_q], gen_a, api_name="ask_images")
    return ui


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen-Drive-1.0-4B")
    parser.add_argument("--planner", default="planner-rl", choices=list(PLANNERS))
    parser.add_argument("--scenes", default="data/demo/planning_scenes.jsonl")
    parser.add_argument("--image-archive", default="data/demo/frames.parquet")
    parser.add_argument("--nuscenes-scenes", default="data/nuscenes_scenes.jsonl",
                        help="optional, built by tools/nuscenes_to_scenes.py")
    parser.add_argument("--nuscenes-root", default="data/nuscenes")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    SAMPLES, LABELS = load_scenes(args)
    print(f"loaded {len(SAMPLES)} scenes")
    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model = model.to("cuda").eval()
    select_planner(args.planner)
    print(f"model ready on {model.device}, planner {state['planner']}")

    build_ui().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=False,
        theme=gr.themes.Soft(),
    )
