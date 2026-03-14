## Quick Context

- Purpose: research codebase for pedestrian pose and gaze estimation using ONNX/PyTorch backbones.
- Primary entrypoints: `rtmpose_3d_merged.py` (main 3D pipeline + visualization), `rtmpose_jpeg.py` (single-image demo), `yolo_*.py` (detector variants).
- Models live in the `models/` directory (ONNX and some PyTorch `.pt` files). Pre/postprocessing variants are present under `onnx_no_prepost/` and `stcformer_*` folders.

## What to know before editing

- Scripts set environment variables at startup: `RTMLIB_HOME`, `TORCH_HOME`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE` to force model/cache locations into `models/`.
- There is no central config file — most parameters (e.g. `VIDEO_PATH`, thresholds, display sizes, smoothing constants) are top-level constants in each script. Edit the script you run.
- `rtmlib` is the main model wrapper used (e.g. `Wholebody3d`, `Body`). Code expects `rtmlib` and `onnxruntime` available in the environment.

## Architecture notes (high-value facts)

- Data flow in `rtmpose_3d_merged.py`: frame -> `Wholebody3d(frame)` -> returns (kp3d [mm], scores, kp2d [px]) -> filter by keypoint scores -> compute bounding boxes -> tracker -> smoothers -> project 3D to metric coordinates -> visualize + optional recording to `recordings/`.
- Indexing: RTM outputs 133-keypoint skeleton internally; this repo uses COCO-17 body subset (indices 0..16). See `project_3d()` for metric conversion logic and `SKELETON` constant for drawing order.
- Gaze logic: `calc_gaze()` uses nose + ear midpoint; gaze ray correction via `GAZE_ANGLE_CORR` constant.
- Tracking: `Tracker` is a hybrid IOU + centroid-distance tracker; confirm ID assignment logic when changing detection thresholds.

## Developer workflows & helpful commands

- Run the main pipeline (example):

  ```powershell
  python rtmpose_3d_merged.py
  ```

- Single-image demo:

  ```powershell
  python rtmpose_jpeg.py
  ```

- GPU sanity check (repo tool):

  ```powershell
  python tools/test_cuda.py
  ```

- If a model fails to load, check `models/` and the scripts that copy from cache (e.g. `rtmpose_jpeg.py` copies from `~/.cache/rtmlib/hub/checkpoints`).

## Project-specific conventions

- Files under `archiv/` are experiments/legacy implementations — prefer modifying top-level `rtmpose_*` files or adding new scripts at repo root.
- Visualization and IO are done inline (OpenCV windows + `cv2.VideoWriter`). Recording files are saved in `recordings/`.
- When adding new model variants, follow existing pattern: set `MODEL_DIR` relative to script, set the `RTMLIB_HOME`/`TORCH_HOME` env vars, and prefer `backend='onnxruntime'` where existing code uses it.

## Integration points & external dependencies

- Core libs: `rtmlib`, `onnxruntime`, `opencv-python`, `numpy`, (optional) `torch` for device detection and to run some scripts.
- Model files: `models/*.onnx` and a few `.pt` in `models/` — some workflows expect pre/postprocessing already baked into model variants (see `onnx_no_prepost/` vs `*_with_prepost/`).

## When making changes for AI agents

- Keep edits localized: prefer adding new scripts or modules rather than refactoring global constants across many top-level scripts.
- Preserve the environment-variable bootstrapping pattern (set `RTMLIB_HOME` etc.) so model resolution remains predictable in CI or other developer machines.
- If you change keypoint indexing, update `SKELETON`, `NOSE_IDX`, `LEFT_EAR_IDX`, `RIGHT_EAR_IDX` and any references in `project_3d()` and `calc_gaze()`.

## Quick pointers / examples to reference in code reviews

- `rtmpose_3d_merged.py` — main pipeline, metric projection: `project_3d()`.
- `rtmpose_jpeg.py` — compact demo showing how `Body` is initialized and how models may be copied from cache.
- `tools/test_cuda.py` — quick GPU check.

If anything here is unclear or you'd like more detail (for example: a mapping table of the 133→17 indices, or explicit environment setup steps for Windows GPU), tell me which section to expand.
