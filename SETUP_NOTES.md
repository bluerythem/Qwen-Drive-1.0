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

### Overlaying the predicted BEV map under the trajectory

The Planning tab has an **"Overlay the predicted BEV map under the trajectory"** checkbox.
It runs the BEV head on the same keyframe and draws the predicted map raster beneath the
plan. No transform is involved: `docs/perception.md` says the map raster is indexed in ego
coordinates with X forward and Y left, which is exactly the frame the trajectories use, so
it goes straight into `imshow` at `extent=(15, -15, -30, 30)` (60 m x 30 m at 0.15 m).

It only works on the nuScenes scenes, because the BEV head needs the full camera ring and
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
