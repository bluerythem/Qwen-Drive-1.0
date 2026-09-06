# Qwen-Drive-1.0 local setup notes

Set up on 2026-09-05, RTX PRO 6000 Blackwell (sm_120, 96 GB), Ubuntu 24.04, driver 595.84.

## What's here

| Path | What |
| --- | --- |
| `.venv/` | Python 3.12 venv (created with `uv`; `python3-venv` is not installed system-wide) |
| `Qwen-Drive-1.0-4B/` | 13 GB of weights from HF: VLM + `planner-sft/` + `planner-rl/` + `perception/` |
| `cuda128/`, `cuda-home/` | CUDA 12.8 `nvcc` toolchain, needed only by the perception mode |
| `env.sh` | exports `CUDA_HOME`, `PATH`, `PYTHONPATH` — source before any script |
| `run_demo.sh`, `run_perception_demo.sh` | the two verified demos |

## Deviations from the README

- **`uv` instead of conda.** `python3-venv` isn't installed and there's no sudo, so the venv
  came from `uv venv --python 3.12`. `pip install -e .` was skipped in favour of
  `PYTHONPATH=src`, exactly as the README's quick start does.
- **`flash-attn` from a prebuilt wheel**, not a source build — there's no system CUDA toolkit.
  `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312` runs fine on sm_120; verified directly.
- **`causal-conv1d` from a prebuilt wheel too**, same reason as flash-attn:
  `causal_conv1d-1.6.2.post1+cu12torch2.8cxx11abiTRUE-cp312`. Kernels verified on sm_120
  (`causal_conv1d_fn` against a `F.conv1d` reference, plus the `causal_conv1d_update` decode
  path). `transformers.utils.is_causal_conv1d_available()` returns True, so the Qwen3.5
  linear-attention layers use it instead of the torch fallback.
- **`opencv-python` unpinned** (the pinned 5.0.0.93 isn't on PyPI); only used by the
  perception visualizer.
- The perception ops JIT-compile on first run and are cached in `~/.cache/torch_extensions`.
  That first run takes a couple of minutes; later runs are instant.

## Rebuilding the CUDA prefix, if it ever breaks

`cuda-home/` is a symlink farm because no single source has a full toolkit layout:

    bin, nvvm  -> cuda128/           (micromamba: cuda-nvcc=12.8 cuda-cudart-dev cuda-cccl)
    lib64, lib -> cuda128/targets/x86_64-linux/lib
    include/   -> a merge of cuda128/targets/x86_64-linux/include
                  and .venv/.../site-packages/nvidia/*/include   (cusparse.h, cublas_v2.h, ...)

`TORCH_CUDA_ARCH_LIST=12.0` in `env.sh` keeps nvcc from building for every architecture.

## Verified working

- `./run_demo.sh` — VQA, direct planning, reasoning planning on scene 0.
  ADE 0.34 m / FDE 1.06 m against ground truth; writes `demo.png`.
- `./run_perception_demo.sh` — 300 boxes per frame on all 6 frames + occupancy and map
  segmentation; writes `outputs/perception_demo/vis/*.png`.
- `scripts/run_vqa.py` on a single image.

Benchmark runs (`scripts/run_planning.py` + `eval_*.py`) need NAVSIM / Waymo / PhysicalAI
scene files and frames, which aren't shipped — see `docs/data.md`.

## Interactive demo (`app.py`)

    source env.sh && python app.py        # then open http://127.0.0.1:7860

Gradio UI over the same four bundled scenes. The model loads once (~40 s) and stays
resident; every control is a different call into it. Three tabs:

- **Planning** — pick a scene and a navigation command, get the trajectory plot, the chain
  of thought and ADE/FDE. Nav command, planner (`planner-rl` / `planner-sft`, hot-swapped
  with `load_planner`), mode, sample count and noise seed are all live.
  The navigation command has a fourth choice, **`ALL THREE`**, beside the three real
  commands: it plans the scene once per command and draws them on one axis, colour-coded
  (straight blue, left green, right orange), with a reasoning trace and an ADE/FDE row per
  command. It runs three times, so it is 3x slower. Only the row matching the scene's
  recorded command is a fair score; the table marks it. This uses the `overlays` argument
  added to `plot_scene_summary`.
- **Ask about the scene** — `InferenceMode.VQA` over the scene's own frames, with a choice
  of which frames to show.
- **General VQA** — `generate_text` on images you upload; nothing driving-specific.

Overriding the navigation command rewrites all three places it lives: `nav_command`,
the `driving_command` one-hot inside `ego_status`, and the `[GO STRAIGHT]` token in
`instruction_text` (benchmark scenes carry their prompt verbatim, so the numeric fields
alone would not change what the VLM reads). When the command differs from the one the
scene was recorded under, the ADE/FDE line says so — the recorded future is no longer the
right reference.

Stop it with `kill $(ss -ltnp | grep 7860 | grep -o 'pid=[0-9]*' | cut -d= -f2)`. Note
`pkill -f "python app.py"` is a trap here: it also matches the shell that launched it.

### The "Perception frame" tab

3D detection, occupancy and BEV map segmentation on the six bundled frames. The perception
head is attached lazily to the VLM that is already resident (`head.attach(model.vlm, ...)`),
so there is only ever one copy of the 9.1 GB VLM in memory, and its result is cached per
frame.

**Planning is deliberately not offered on these frames.** A perception frame is a single
timestep of the camera ring and carries no ego state, so planning on one means synthesizing
the ego history — and that measured ADE 9.5 m against 1.9 m from a recorded history (table
below). The tab was built that way first and then cut; only scenes with real recorded ego
motion remain, on the Planning tab.

## Real nuScenes scenes (`tools/nuscenes_to_scenes.py`)

The two bundled perception frames are from nuScenes **trainval**, not mini, so their history
is not available locally. But `~/Downloads/v1.0-mini.tar` (already on this machine) has 10
scenes / 404 keyframes with everything a planning scene needs, so the scenes are built from
mini instead:

    tar xf ~/Downloads/v1.0-mini.tar -C data/nuscenes v1.0-mini samples   # 4.6 GB
    source env.sh
    python tools/nuscenes_to_scenes.py --root data/nuscenes --output data/nuscenes_scenes.jsonl
    # -> 255 scenes (209 straight, 31 left, 15 right)

Why mini works: nuScenes keyframes are 2 Hz, exactly the 0.5 s camera cadence the model
expects, and `ego_pose` is stored per sample_data, so the 20 Hz LIDAR_TOP stream gives a
dense pose timeline to interpolate a true 10 Hz history *and* a true 5 s future from.

Output is the scene-file format from `docs/data.md`, not a bespoke one, so the upstream
tooling reads it unchanged:

    python scripts/demo.py --model Qwen-Drive-1.0-4B --planner Qwen-Drive-1.0-4B/planner-rl \
        --scenes data/nuscenes_scenes.jsonl --image-root data/nuscenes --index 0
    python scripts/run_planning.py --scenes data/nuscenes_scenes.jsonl --image-root data/nuscenes ...

The prompt text is produced by `DrivingScene.instruction()` rather than hand-written, so the
wording matches the released files exactly.

### Does the real history help? Yes, about 5x

Same 40 scenes, `planner-rl`, reasoning planning, 6 samples, seed 42. "Zeroed ego state" is
what the perception-frame tab is forced to assume:

| variant | ADE | FDE |
| --- | --- | --- |
| real 10 Hz ego history | **1.917 m** | **5.413 m** |
| zeroed ego state | 9.512 m | 13.507 m |

### Caveats

- **`nav_command` is inferred from the recorded future** (heading change over 5 s, threshold
  0.35 rad, `--turn-threshold`). nuScenes ships no route command. This leaks ground truth:
  the model is told which way the vehicle actually turned. The released benchmarks take the
  command from a route planner instead, so these numbers are optimistic in that respect.
- **Not comparable to the reported numbers.** nuScenes is not one of the three planning
  benchmarks in the technical report, and its three forward cameras differ in FOV and
  mounting from the WOD-E2E rig. Treat ADE/FDE here as relative, not as a score.
- Only 255 of 404 keyframes qualify: a sample needs 1.5 s of camera history before it and
  5 s of recorded pose after it, which drops the ends of each scene.

### Overlaying a BEV map under the trajectory

The Planning tab has a **"Map under the trajectory"** choice: `none`, `predicted` (the BEV
head on the same keyframe), `ground truth (vector)` or `ground truth (raster)`. All the map
options are nuScenes-only.

The nuScenes map expansion is a **vector** map - polygons, polylines and a lane graph - so
`ground truth (vector)` draws it as geometry rather than pixels: filled polygons for
drivable surface, walkway and crosswalks, dashed lane dividers, the drivable-area boundary
as a road edge, and lane centrelines discretized from `arcline_path_3` at 1 m. Interior
rings are punched as real holes via a single `matplotlib.path.Path` per polygon. It is
sharper than the raster and shows lane structure the 0.15 m grid cannot.

`get_map_geom` clips each record to the patch and returns it already rotated into the ego
frame (X forward, Y left), so only the lane centrelines need transforming by hand - those
come back as global poses.

No transform is involved: `docs/perception.md` says the map raster is indexed in ego
coordinates with X forward and Y left, which is exactly the frame the trajectories use, so
it goes straight into `imshow` at `extent=(15, -15, -30, 30)` (60 m x 30 m at 0.15 m).

#### Ground-truth maps

    # 398 MB, from Motional's public bucket; the nuScenes non-commercial licence applies
    curl -L -o /tmp/map-expansion.zip \
      https://motional-nuscenes.s3.amazonaws.com/public/v1.0/nuScenes-map-expansion-v1.3.zip
    unzip -q /tmp/map-expansion.zip -d data/nuscenes/maps
    uv pip install nuscenes-devkit

`NuScenesMapGT.raster(sample_token)` calls the devkit's `get_map_mask` for a 60 m x 30 m
patch at the keyframe's ego pose and heading, then folds the nuScenes layers into the
model's six classes:

| model class | nuScenes source |
| --- | --- |
| driveable_surface | `drivable_area` |
| walkway | `walkway` |
| crosswalk | `ped_crossing` |
| road_line | `lane_divider` + `road_divider` |
| road_edge | boundary of `drivable_area` (nuScenes has no road-edge layer) |

Two things had to be checked rather than assumed:

- **Orientation.** The devkit returns `(h, w)` for `canvas_size=(200, 400)` with the patch
  box as `(x, y, height, width)`, so rows are Y and columns X, same as the model's grid.
  Confirmed by scoring drivable-surface IoU of prediction against ground truth over 12
  frames under every flip: as-is **0.799**, flip-Y 0.335, flip-X 0.530, flip-both 0.293.
  As-is wins by a wide margin, so no flip is applied.
- That 0.80 IoU also cross-validates both sides: the perception head and the rasterizer
  agree on where the road is.

Note the mini tar's own `maps/*.png` are only `semantic_prior` masks - a binary
drivable-area raster at 10 px/m, no lanes or crosswalks. The expansion pack is what carries
the vector layers.

### Overlaying the predicted BEV map under the trajectory

The `predicted` choice only works on the nuScenes scenes, because the BEV head needs the full camera ring and
calibration and the WOD-E2E demo scenes carry neither. Asking for it on a demo scene returns
a note saying so rather than failing.

    python tools/nuscenes_perception_frames.py    # 255 frames, 11 MB

builds a perception frame per planning scene, keyed by the same sample token. Calibration is
real, derived from nuScenes `calibrated_sensor` as `inv(lidar2ego) @ cam2ego`; images are
symlinked rather than copied; `gt.npz` is written empty and compressed, because nuScenes has
no occupancy or map rasters of its own (those need Occ3D and the map expansion) and inference
never reads it. Verified by rendering a full perception summary: the 3D boxes project tightly
onto the vehicles and barriers in all six cameras, so the extrinsics are right.

The overlay is drawn in `save_plan_figure()` in `app.py` on the figure `plot_scene_summary`
returns, rather than as another change to upstream `visualize.py`.


## Lane changes: there are none in nuScenes mini

`tools/nuscenes_lane_change.py` annotates a scene file in place with
`meta_info.lane_change`, and the app appends **`, LANE CHANGE`** to a scene's dropdown label
when that flag is set. Run it after building the scenes:

    python tools/nuscenes_lane_change.py --scenes data/nuscenes_scenes.jsonl

On the mini split it flags **0 of 255**, and that is the correct answer, not a broken
detector. Evidence:

- Intersecting each of the 10 full drives with the map's `lane_divider` polylines gives
  **one** crossing in total, in scene-0061 at 10.9 m into a 91 m drive - keyframe #2, which
  is before the first window that has the 1.5 s of history a scene needs, so no scene covers
  it. The other nine drives never cross a divider.
- 22 scenes do shift sideways by more than 1.5 m while staying near-straight, but none cross
  a divider. They are road curvature: idx 198 moves 2.8 m over ~70 m travelled at 14 m/s,
  a ~875 m radius curve, with only 0.07 rad of heading change.
- The largest lateral shifts (idx 0-22, up to +9.6 m) are one long left turn through a
  junction, with heading building to ~1 rad. A turn is not a lane change.

Mini is 10 scenes of dense urban driving, roughly 20 minutes, so this is unsurprising. The
tool is written against the full trainval split too: build scenes from it and any lane
changes will be flagged and labelled automatically.

### Two detectors, and why the first was wrong

The first version asked whether the end lane was reachable from the start lane through the
map's lane graph. It flagged 25 scenes - all false. Lane records are short and their
`outgoing` connectivity is sparse enough that an ordinary straight drive often ends in an
"unreachable" lane. The flagged scenes averaged **less** lateral motion than the unflagged
ones (0.47 m against 0.55 m), which is what exposed it.

The geometric test that replaced it needed one fix of its own: shapely's `crosses` requires
an interior-interior intersection, but dividers are short two-point segments that a path
frequently meets at an endpoint, so the test uses `intersects`. That was caught by probing a
divider with a synthetic path built to cross it and getting `False` back.


## Lane changes in the full val split

Detection needs no images: ego poses and the map are enough, and both come from
`v1.0-trainval_meta.tgz` (454 MB) plus the map expansion already downloaded. That matters
because the camera blobs are ~350 GB in total, cannot be fetched per file (individual image
URLs 404), and are **not** split-aligned - the 150 val scenes are spread across all ten, so
"val only" does not shrink the download. Scanning metadata first says which scenes would be
worth it.

    curl -L https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval_meta.tgz \
      | tar xz -C data/nuscenes-trainval          # 454 MB down, 2.5 GB on disk
    python tools/nuscenes_scan_lane_changes.py --root data/nuscenes-trainval \
      --version v1.0-trainval --split val --maps-from data/nuscenes \
      --output data/lane_changes_val.json

Result: **42 lane-change keyframes in 10 of the 150 val scenes**, from 4201 usable keyframes.
Merging consecutive keyframes (one manoeuvre is seen from several starting points) gives
**15 distinct lane changes**.

| scene | keyframes | scene | keyframes |
| --- | --- | --- | --- |
| scene-0273 | 7 | scene-0557 | 4 |
| scene-0914 | 6 | scene-0931 | 4 |
| scene-0962 | 6 | scene-0268 | 2 |
| scene-0638 | 5 | scene-0563 | 2 |
| scene-0783 | 5 | scene-0093 | 1 |

The detections look right: lateral shift averages 2.83 m - about one lane width - at a mean
heading change of 0.129 rad (7.4 deg), which is the signature of a lane change rather than a
turn. 26 are to the right, 16 to the left. scene-0783 keyframes 25-29 are the cleanest
example: -2.3 m of lateral movement with under 0.08 rad of heading change.

Building runnable scenes from these needs the image blobs, which is the 350 GB download.
`data/lane_changes_val.json` records every hit with its `sample_token`, so the scene builder
can be pointed straight at them once images are present.


## Where could the ego change lane? (mini)

`tools/nuscenes_multilane.py` probes sideways from the ego pose, snaps each probe to a lane,
and keeps only lanes whose heading agrees with the ego's - an oncoming lane across a centre
line is not somewhere you can move to. It also reports the marking type, so
`DOUBLE_DASHED_WHITE` (crossable) is distinguishable from `DOUBLE_SOLID_WHITE`.

    python tools/nuscenes_multilane.py --scenes data/nuscenes_scenes.jsonl

The dropdown label then carries the road width, e.g.
`nuscenes 199 - straight, 14.3 m/s, 3 lanes, room 2R`, or `junction`.

Of 255 scenes: **117 have the ego inside a junction**, and of the 138 on a real lane, **80
have an adjacent same-direction lane**. Only **26 have room on both sides**, and every one of
those is stationary.

| what | scenes | note |
| --- | --- | --- |
| widest while moving | idx 186-204 | 3-lane road, ego in the **left** lane, **2 lanes to its right**, 12-15 m/s, all `DOUBLE_DASHED_WHITE` |
| true middle lane (room 1L1R) | idx 49-74 | 3 lanes, but **stopped at a red light** in Boston, 0.0 m/s |

There is no scene in mini where the ego is *driving* with a lane on each side. Singapore is
left-hand traffic, so on idx 186-204 the two lanes to the right are the overtaking side.

Two corrections were needed to get here. Counting `lane_connector` records as neighbours made
every junction look multi-lane - they were 61% of all hits - so only `lane` records count now,
and scenes with the ego inside a junction are excluded. The dropdown label was then briefly
built from the raw counts rather than from `multi_lane`, which advertised "room" for 11
junction scenes; it is gated on `multi_lane` now.

A stationary ego also collapsed the trajectory panel to a flat line, since every point sits at
x ~ 0 and the equal aspect ratio squashes it. `_floor_longitudinal_span` gives the panel a
20 m minimum so the stopped scenes render.


## The "Trion mock" tab: two systems and a resolver, end to end

**No neural network runs on this tab.** It exercises the interface of a two-system stack:

    voice command -> Trion-Reason (slow, symbolic) -> Resolver (HD map) -> Trion-Action (fast)

- **Trion-Reason** (`tools/trion.py: reason`) is a rule-based stand-in for the reasoning
  model. A command in, a *symbolic* goal out: lane-relative `lateral` (KEEP / LEFT / RIGHT /
  PULL_OVER) with a **window** and a **deadline** in metres, a `cruise` preference relative to
  the speed limit, a planned stop, and which traffic light matters. Never geometry. It also
  deliberately does not check whether a lane exists - letting a bad request through is what
  makes the validation path visible.
- **Resolver** (`Resolver`) is the one real component: deterministic nuScenes map logic.
  It localizes the ego in a lane, walks `connectivity` to build the current corridor, probes
  the requested side every 4 m to classify what is there (lane / junction connector / nothing),
  fits the window to the stretch where a proper lane exists, checks the divider marking, the
  target's continuity, and that it is a *distinct* lane, then emits two corridors, the fitted
  window, a speed cap and any stop point. Any failed check rejects the request and falls back
  to KEEP, with the reason shown.
- **Trion-Action** (`act`) is a stand-in for the fast planner: a comfort-bounded speed profile
  towards the cap (or a 2.5 m/s^2 stop), and the lateral move placed *inside* the window - it
  "commits" 35% of the way in, standing in for gap acceptance.

The figure is a left-to-right story for a non-specialist: the three stages as cards, the
front camera with the target ribbon and the plan projected onto it, and a bird's-eye panel
with both corridors, the window and stop markers. The lateral axis of that panel is
stretched; it says so in the title.

What the four canned commands show on `nuscenes 147` (ego 11.9 m/s, kerb lane of three):

| command | resolver | outcome |
| --- | --- | --- |
| change to the right lane | window **fitted 12-45 -> 40-70 m**: a side-road junction sits beside the road at 16-36 m and you do not change lanes into a junction | drives RIGHT, -3.0 m, commits at 50 m |
| change to the left lane | **rejected**: the "lane" the map has on the left at 40 m is **0.0 m from the current centreline** - an overlapping record at a split, not a lane | fallback KEEP |
| pull over | kerb boundary 2.2 m left; stop at 28 m (2.5 m/s^2 from 11.9 m/s) | drives PULL_OVER, +1.3 m, rests at 0 m/s |
| keep going | localized, nothing else to check | KEEP, accelerates to the 50 km/h cap |

### Navigation input: the Route matcher

A nav app hands over a *road-level* route ("turn right in 180 m"); the stack needs
*lane-level* goals. `lane_route()` in `tools/trion.py` bridges them from the HD map: it finds
the same-direction lanes across the road at the ego (`lane_group`), walks each forward to the
first junction with a real turn - a `lane_connector` whose heading changes by more than 25
degrees - and reports which lanes make the requested turn, how many changes that is from the
ego's lane, and how far away it is. That is the `LaneRoute` message Trion-Reason reads.

Trion-Reason then does the two things a symbolic reasoner is for. It judges **feasibility**:
N changes need about N x (2.5 s of travel + 10 m) + 15 m; if the junction is closer than that,
the answer is "continue and ask the nav app to reroute", never "force the gap". And it
**arbitrates** route against voice: the route is the default, a voice request is a scoped
override that wins until done, and the message carries `source: ROUTE | VOICE` plus a
`pending` follow-up ("+1 RIGHT before the junction") so a two-change sequence is explicit.

Four scenes, chosen by running the matcher over every moving on-lane scene in mini:

| scene | route: turn right at the next junction | what it shows |
| --- | --- | --- |
| nuscenes 147 | at 13 m, needs 2 changes | out of reach -> KEEP, **reroute requested** |
| nuscenes 132 | at 103 m, needs 1 change | the canonical route-driven change; the right side is a junction until 60 m, so the resolver fits the window to 64-88 m |
| nuscenes 189 | at 127 m, needs 2 changes | a **sequence**: RIGHT now with `pending +1 RIGHT`, deadline 71 m |
| nuscenes 203 | at 33 m at 15 m/s, needs 2 | not possible -> reroute |

"turn left" on 147 shows the other branch: already in a valid lane, hold it. "arrive:
destination on the left" resolves to a PULL_OVER from source ROUTE - the same manoeuvre the
voice command produces. And voice "pull over" on 132 with the route wanting RIGHT shows the
override: PULL_OVER from VOICE, with `pending route: RIGHT for the turn`.

nuscenes 128 was the first pick for the single-change case and the resolver rejected it: its
right side is junction connectors for 60 of the first 90 m, with a real lane only at 24-32 m.
That is the map being right, and 132 replaced it.

Every line on the figure carries a provenance tag, and a key sits in the footer:

| tag | meaning | examples |
| --- | --- | --- |
| `in` | input the system receives | scene, ego speed, the voice command, history, ego marker |
| `msg` | field of a real inter-system message | everything Trion-Reason emits; every resolver check, corridor, window, cap and stop |
| `out` | real output of the system | the 50 x 3 trajectory, on the camera and the map |
| `viz` | derived from the output for display only | reach, lateral vs own lane, end speed - all computed from the trajectory |
| `mock` | exists only in this mock | "commits at N m": a fixed 35% into the window, which a real planner does not emit unless given a head for it |

The distinction matters for the room: the pictures on the right are what production would
look like - a real trajectory rendered two ways - and the summary numbers are honest
derivations, except the one red line.

Assumptions worth saying out loud in the room: the speed limit is **assumed** 50 km/h -
nuScenes' map has none; "commits at" is a fixed fraction of the window, not a gap model; and
`lateral` is measured against the current lane's centreline, so following a curving lane
reads as 0.0 and a lane change reads as one lane width.

Three things the build corrected along the way, each caught by looking at the output:
a majority vote across probe distances was needed because one stray hit 40 m ahead had
conjured a lane beside a kerb; lane records are ~40 m long, so the neighbour spans several
tokens and hits must be counted rather than tokens; and the pull-over's lateral move has to
finish *before* the stop, not after it.
### Drawing the plans on the camera images

Both the navigation target and the mocked trajectory are also projected onto the **current
frame** of each camera view, through the nuScenes calibration
(`tools/nuscenes_cameras.py`). Points are taken at ground level, `z = 0` in the ego frame -
the ego origin sits on the road, which is why the front camera's calibrated height is ~1.5 m.

The target is drawn as a **ribbon of constant metric width** (1.1 m), not a fixed-pixel line:
both edges are projected and the polygon between them filled, so perspective narrows it with
distance and it reads as something painted on the road. The trajectory stays a thin line.

Only the front camera usually shows anything. The side cameras are projected too, but a path
running up the ego's own lane falls outside their field of view and is clipped - which is
correct, not a failure.

Two things this needed:

- Plotting on a montage axes **autoscales it**, and a projected path runs far outside the
  frame - the point at the ego's own bumper lands ~600 px below the image. The first attempt
  shrank every camera picture to a corner of its panel. The axes limits are now captured
  before drawing and pinned back afterwards, so everything outside is simply clipped.
- `CameraFrame.load()` does not resize, so the images are the original 1600x900 and pixel
  coordinates map straight onto the montage without scaling.


### Layout of the mock tab

The figure is three columns: a small 3x4 montage, one large current front frame, and the
bird's-eye panel. `mock_figure()` in `app.py` builds it rather than `plot_scene_summary`,
because **enlarging one cell of a uniform grid does not work**: giving the front row and the
current column more room hands that room to every other cell in the same row and column, and
their images float in the middle of it with large gaps. The first attempt produced exactly
that. The montage stays uniform and small, and the frame worth looking at gets a column of
its own.

Rows are still views and columns still timestamps, and the montage's top row is labelled
`t-1.5s ... t-0.0s` so the sequence reads at thumbnail size. Only the front camera's current
frame carries the projected plan, so it is the only one enlarged.

The navigation target also stops just past where the plan ends (1.05x the distance the ego
covers, not 1.3x). Running it further stretched the bird's-eye panel to 80 m and squashed
everything worth seeing into a sliver.
