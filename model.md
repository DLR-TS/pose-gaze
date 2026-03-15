# pose-gaze — Internal Representation for LLM Assistance

## System in One Sentence
Reads a video file, runs RTMW3D pose estimation per frame, projects 17 joints to metric 3-D camera space via pinhole model + anthropometric scaling, tracks persons, estimates velocity via OLS regression, fits a ground plane via SVD, and renders overlays on-screen with optional MP4+NDJSON recording.

---

## File Dependency Graph
```
config.py   (no imports from project)
    ↓
backend.py  (imports from config only)
    ↓
frontend.py (imports from backend + config)
```
Entry point: `python frontend.py`

---

## config.py — Complete Constant Table

### Paths
| Symbol | Value | Type |
|--------|-------|------|
| SCRIPT_DIR | `Path(__file__).parent` | Path |
| MODEL_DIR | `SCRIPT_DIR/"models"` | Path |
| OUTPUT_DIR | `SCRIPT_DIR/"recordings"` | Path |
| CAMERA_JSON_DIR | `SCRIPT_DIR/"media"` | Path |
| VIDEO_PATH | `"media/video_1808x1392_mvBlueCOUGAR-X109b_crop205-391-2013-1783.mp4"` | str |
| FPS_FALLBACK | 13.2 | float |
| K_FALLBACK | 3×3 matrix, FX=FY=4637.7, CX=2056, CY=1088 | ndarray |

### Detection / Keypoint
| Symbol | Value | Effect |
|--------|-------|--------|
| MIN_KEYPOINT_CONFIDENCE | 0.25 | joint visibility gate |
| MIN_VISIBLE_KEYPOINTS | 6 | skip detection below this |
| MIN_PIXEL_HEIGHT_SPAN | 5.0 px | skip 3-D if too small |
| DETECTION_SCORE_THRESHOLD | 0.30 | YOLOX person detector gate |
| MIN_DETECTION_MEAN_CONFIDENCE | 0.20 | mean-joint score gate |

### Tracker
| Symbol | Value | Effect |
|--------|-------|--------|
| TRACKER_IOU_THRESHOLD | 0.25 | min IoU to match |
| TRACKER_MAX_INACTIVE_FRAMES | 90 | drop track after this many missed frames |
| TRACKER_MIN_HIT_STREAK | 2 | suppress new ID until seen this many frames |
| TRACKER_CENTROID_DIST_NORM | 100.0 px | normalisation for centroid distance score |
| TRACKER_CENTROID_DIST_WEIGHT | 0.3 | weight of centroid term in match score |

### Body Geometry (DO NOT CHANGE)
| Symbol | Value | Reason |
|--------|-------|--------|
| BODY_HEIGHT_PRIOR_M | 1.75 | De Leva scale anchor |
| RTMW3D_HEAD_ANKLE_Z_RATIO | 0.96 | model calibration |
| DE_LEVA_SEGMENTS | 16 triples (joint_a, joint_b, body_fraction) | De Leva 1996 Table 1 |
| VISIBLE_BODY_RATIO_ANKLES/KNEES/HIPS | 1.00/0.75/0.52 | fraction of body height visible |
| JOINT_DEPTH_MIN_M / MAX_M | 0.1 / 60.0 | clamp per-joint depth |
| BODY_HEIGHT_MIN_M / MAX_M | 0.5 / 2.5 | valid height range |
| HEIGHT_BUFFER_WINDOW_FRAMES | 60 | rolling median window for height |
| HEIGHT_BUFFER_MIN_SAMPLES | 5 | min samples before median is used |
| HEIGHT_OUTLIER_MAX_DEVIATION | 0.30 | reject if > 30% from running median |

### EMA Smoothing
| Symbol | Value | Applied to |
|--------|-------|-----------|
| EMA_ALPHA_PIXEL_HEIGHT | 0.30 | pixel body height span |
| EMA_ALPHA_DISPLAY | 0.35 | dist_m, elev_deg, azim_deg, height_m |

### Velocity
| Symbol | Value | Effect |
|--------|-------|--------|
| VELOCITY_BUFFER_S | 2.0 s | OLS regression window width |
| VELOCITY_MIN_SPAN_S | 0.5 s | min data span before computing |
| VELOCITY_MAX_DT_S | 3.0 s | flush buffer on gap > this (seek/pause) |
| VELOCITY_MAX_SPEED_MS | 8.0 m/s | hard cap before EMA (≈29 km/h) |
| VELOCITY_EMA_TAU_S | 2.0 s | EMA time constant, fps-invariant |
| VELOCITY_BUFFER_SIZE | 60 | deque maxlen safety limit |

### Label Display
| Symbol | Value | Effect |
|--------|-------|--------|
| LABEL_SMOOTH_WINDOW_S | 2.0 s | trailing-average window for all overlay values |

### Display / Window
| Symbol | Value |
|--------|-------|
| DISPLAY_MAX_WIDTH_PX | 1580 |
| DISPLAY_MAX_HEIGHT_PX | 840 |
| SEEK_STEP_SECONDS | 5 |

### Joint Index Map (RTMW3D COCO-17)
```
0=NOSE  1=L_EYE  2=R_EYE  3=L_EAR  4=R_EAR
5=L_SHOULDER  6=R_SHOULDER  7=L_ELBOW  8=R_ELBOW
9=L_WRIST  10=R_WRIST  11=L_HIP  12=R_HIP
13=L_KNEE  14=R_KNEE  15=L_ANKLE  16=R_ANKLE
```
Named constants defined for: NOSE(0), L_EAR(3), R_EAR(4), L_SHOULDER(5), R_SHOULDER(6),
L_HIP(11), R_HIP(12), L_KNEE(13), R_KNEE(14), L_ANKLE(15), R_ANKLE(16).

### Ground Plane
| Symbol | Value |
|--------|-------|
| GROUND_PLANE_ENABLED | True |
| GROUND_PLANE_MAX_DEPTH_M | 20.0 |
| GROUND_PLANE_SAMPLE_INTERVAL_S | 0.5 |
| GROUND_PLANE_SAMPLE_BUFFER | 200 |
| GROUND_PLANE_MIN_SAMPLES | 20 |
| GROUND_PLANE_FILL_ALPHA | 0.25 |
| GROUND_PLANE_X_STEP_M | 1.0 |
| GROUND_PLANE_X_HALF_M | 4.0 |
| GROUND_PLANE_NEAR_FRAC | 0.75 (y_near = h*0.75) |
| GROUND_PLANE_FAR_FRAC | 0.25 (y_far  = h*0.25) |

### Colors (BGR)
| Symbol | BGR | Used for |
|--------|-----|---------|
| COLOR_KEYPOINT_NOSE | (0,0,255) red | nose dot |
| COLOR_KEYPOINT_EAR | (255,0,0) blue | ear dots |
| COLOR_KEYPOINT_OTHER | (0,255,0) green | remaining joints |
| COLOR_EAR_CONNECTOR | (150,150,150) grey | ear-to-ear line |
| COLOR_GAZE_ORIGIN_DOT | (0,255,0) green | nose dot on gaze ray |
| COLOR_GAZE_LINE | (0,255,255) yellow | gaze ray line |
| COLOR_GAZE_ENDPOINT | (0,0,255) red | gaze endpoint dot |
| COLOR_LABEL_DIST | (255,255,0) cyan | H/distance labels |
| COLOR_LABEL_ANGLES | (255,0,255) magenta | elev/azim/spd labels |
| COLOR_LABEL_GAZE | (0,255,255) yellow | gaze/id/conf labels |
| COLOR_BEARING_RAY | (255,200,0) light-cyan | bearing ray + dist label |
| COLOR_GROUND_LONG_LINES | (80,80,80) dark-grey | grid lines |
| COLOR_GROUND_PLANE_FILL | (30,80,30) dark-green | fill polygon |
| COLOR_HUD_RECORDING | (0,0,255) red | status bar when REC |
| COLOR_HUD_PAUSED | (0,220,255) yellow | status bar when PAUSED |
| COLOR_HUD_RUNNING | (0,255,0) green | status bar normal |

---

## config.py — Dataclasses

### GroundPlane
```python
centroid:  ndarray(3,) f32   # mean of joint samples, camera frame [m]
normal:    ndarray(3,) f32   # unit normal, always normal[1] < 0 (Y_cam = down)
n_samples: int
```

### ObjectVelocity
```python
vx: float  # forward  = +Z_cam [m/s]
vy: float  # lateral  = +X_cam (right+) [m/s]
vz: float  # vertical = -Y_cam (up+) [m/s]
rx, ry, rz: float = 0  # rotational, unused
covariance: List[float] = []
```

### DetectedObject.to_dict()
Renames `object_class` → `"class"`, drops None fields.

### DetectionMessage.to_json()
Compact JSON, no spaces: `separators=(",",":")`.

---

## backend.py — Module-Level State

```python
_tracker:        PersonTracker          # singleton, persistent across frames
_profiles:       PersonProfiles         # singleton, per-person smoothing state
_camera_params:  Optional[dict]         # set by init_camera()
_ground_ref_buf: deque(maxlen=200)      # rolling 3-D joint positions
_last_ground_t:  float                  # timestamp of last SVD fit
_ground_plane_g: Optional[GroundPlane]  # latest fitted plane
_last_det_msg:   Optional[DetectionMessage]
_inference_model: Wholebody3d           # rtmlib model, loaded at import time
_DELEVA_JOINT_A/B/RATIO: ndarray       # precomputed from DE_LEVA_SEGMENTS
_TORSO_JOINTS:   [5, 6, 11, 12]        # L/R shoulder + L/R hip, module-level const
DEVICE:          str                    # "cuda" or "cpu"
```

---

## backend.py — Function Reference

### init_camera(w, h, fps) → dict
Search order for calibration JSON:
1. `camera_{w}x{h}_*.json` (±1 pixel padding for codec)
2. Content-scan: WIDTH/HEIGHT fields in JSON ±1
3. Same aspect-ratio JSON scaled to target resolution

Returns dict with keys: `FX FY CX CY W H FPS K D source crop`.
Sets `_camera_params`. Must be called before `process_frame()`.

### process_frame(bgr_frame, timestamp_s=0.0) → List[PersonData]
Per-frame pipeline (in order):
1. `_inference_model(bgr_frame)` → `kp2d(N,17,2)`, `kp3d_norm(N,17,3)`, `scores(N,17)`
2. Filter detections: MIN_VISIBLE_KEYPOINTS, MIN_DETECTION_MEAN_CONFIDENCE
3. `_tracker.update(bboxes, confs)` → `[(person_id, conf)]`
4. Per person:
   - `_pixel_body_height_span()` → `raw_ph`; `_profiles.smooth_pixel_height()` → `sm_ph`
   - `compute_3d_pose(..., pixel_height_hint=sm_ph)` → `(jm, h_m, centroid_3d, (dist,elev,azim))`
   - `_profiles.push_height_sample()` (only if head+ankle both visible)
   - `_profiles.smooth_display_values()` → smoothed dist/elev/azim/h
   - `compute_gaze_ray_3d()` → `(gaze_origin_m, gaze_unit)`
   - `_profiles.compute_velocity(person_id, centroid_3d, timestamp_s)`
   - Append `PersonData`
5. Ground plane update every GROUND_PLANE_SAMPLE_INTERVAL_S seconds

### compute_3d_pose(px, k3d_norm, sc, fx, fy, cx, cy, h_ref, px_h_hint)
Returns `(jm, h_m, centroid_3d, (dist, elev, azim))` or `(None, 0., None, None)`.

**Scaling (units_to_metres)**:
Confidence-weighted mean of De Leva segment ratios:
`u2m = Σ(ratio[i] * BODY_HEIGHT_PRIOR / span[i] * conf_product[i]) / Σconf_product[i]`

**Depth anchor**:
`z_root = FY * height_reference_m * visible_body_ratio / pixel_height_span`
`depth[j] = clip(z_root + k3d_norm[j,2] * u2m, JOINT_DEPTH_MIN, JOINT_DEPTH_MAX)`

**Joint priority** for `visible_body_ratio`:
ankles (1.00) > knees (0.75) > hips (0.52) — first available group wins.

**Centroid**:
Mean of visible _TORSO_JOINTS [5,6,11,12]; falls back to all visible joints if <2 torso joints visible. This centroid is used for distance/elevation/azimuth display AND velocity input.

**Height measurement** (returned for profile update):
`h_m = |mean(k3d_norm[head,2]) − mean(k3d_norm[ankles,2])| * u2m / RTMW3D_HEAD_ANKLE_Z_RATIO`
Only used if both head and ankle joints visible.

### compute_gaze_ray_3d(joints_meters, joint_scores)
Requires NOSE(0), L_EAR(3), R_EAR(4) all above MIN_KEYPOINT_CONFIDENCE.
`gaze_unit = normalise(nose_3d − ear_midpoint_3d)`
Returns `(nose_3d_copy, gaze_unit.astype(f32))` or `(None, None)`.
No pitch correction applied.

### gaze_angles_3d(gaze_unit_dir) → (azimuth_deg, elevation_deg)
`azimuth   = atan2(d[0],  d[2])`  — 0° = same dir as camera, +90° = right
`elevation = atan2(-d[1], hypot(d[0],d[2]))` — positive = looking up

### PersonProfiles.compute_velocity(person_id, centroid_m, timestamp_s) → ObjectVelocity
1. Flush `_pos_history[pid]` and `_vel_ema[pid]` if gap > VELOCITY_MAX_DT_S
2. Append `(timestamp_s, centroid_m.copy())`
3. Collect samples with `t >= timestamp_s − VELOCITY_BUFFER_S`
4. Return last EMA (or zero) if < 3 samples or span < VELOCITY_MIN_SPAN_S
5. OLS: `Tc = T − mean(T)`, `Pc = P − mean(P, axis=0)`, `raw = (Tc @ Pc) / ||Tc||²`
6. Hard cap: if `||raw|| > VELOCITY_MAX_SPEED_MS`: `raw *= cap/||raw||`
7. Time-based EMA: `α = 1 − exp(−dt / VELOCITY_EMA_TAU_S)` where dt = last frame gap
8. Return `ObjectVelocity(vx=ema[2], vy=ema[0], vz=-ema[1])`
   — axis remap: cam_Z→vx(fwd), cam_X→vy(lat), −cam_Y→vz(up)

### PersonProfiles.smooth_display_values(pid, dist, elev, azim, height) → tuple
EMA(α=0.35) on all four scalars. Azimuth uses wrap-around correction (±180°).
First call initialises EMA to raw values (no lag on first appearance).

### PersonTracker.update(bboxes, confs) → List[(person_id, conf)]
Match score per pair: `IoU + (1 − min(1, centroid_dist/100)) * 0.3`
Match accepted if: `score > best_score AND (IoU > 0.25 OR centroid_dist < 100)`
New track suppressed until `hit_streak >= TRACKER_MIN_HIT_STREAK (2)`.
Track dropped after `inactive_frames > TRACKER_MAX_INACTIVE_FRAMES (90)`.

### proj3d2d(point_3d, fx, fy, cx, cy) → Optional[ndarray(2,)]
Returns None if `point_3d[2] <= 0`.
`[x/z * fx + cx, y/z * fy + cy]`

### _fit_ground_plane()
SVD on `(_ground_ref_buf − centroid)`. Smallest singular vector = normal.
Flip normal so `normal[1] < 0`. Min GROUND_PLANE_MIN_SAMPLES (20) required.

### build_detection_message(persons, t_s, frame_id="CAMERA") → DetectionMessage
Builds DetectionMessage from List[PersonData]. Stores in `_last_det_msg`.
Called by frontend only when recording (R key active).

---

## PersonData Fields

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| person_id | int | tracker | persistent ID |
| pixel_coords | ndarray(17,2) f32 | inference | raw 2-D keypoints [px] |
| joint_scores | ndarray(17,) f32 | inference | confidence [0,1] |
| joint_visible | ndarray(17,) bool | derived | score > MIN_KEYPOINT_CONFIDENCE |
| joints_meters | ndarray(17,3) f32 | compute_3d_pose | 3-D positions, camera frame [m] |
| distance_m | float | EMA-smoothed | torso centroid radial distance [m] |
| elevation_deg | float | EMA-smoothed | vertical angle above horizon [°] |
| azimuth_deg | float | EMA-smoothed | horizontal bearing [°] |
| height_m | float | rolling median | body height estimate [m] |
| gaze_origin_m | ndarray(3,) or None | compute_gaze | nose 3-D position [m] |
| gaze_direction | ndarray(3,) or None | compute_gaze | unit vector, camera frame |
| velocity_m_per_s | ObjectVelocity or None | compute_velocity | None until VELOCITY_MIN_SPAN_S of data |

---

## Coordinate Systems

| Frame | X | Y | Z | Origin |
|-------|---|---|---|--------|
| Pixel | right→ | down↓ | — | top-left |
| Camera-3D | right→ | down↓ | forward→ | lens |
| Message | forward→ | right→ | up↑ | lens |

Mapping Camera→Message: `msg.x=Z_cam, msg.y=X_cam, msg.z=−Y_cam`
Mapping in ObjectVelocity: `vx=ema[2](Z_cam), vy=ema[0](X_cam), vz=−ema[1](−Y_cam)`

Camera intrinsics dict keys: `FX FY CX CY W H FPS K(3×3) D(5,) source crop`

---

## frontend.py — Structure

### LabelSmoother
```python
_buf: dict  # (person_id, field_name) -> deque[(t_s, value)]
push(pid, field, value, t_s) -> float  # returns rolling mean over LABEL_SMOOTH_WINDOW_S
```
No frozen display, no update interval — returns live rolling mean every call.
Speed additionally quantised to 0.5 km/h steps in `draw_person`.

### draw_ground_plane(img, gp, fx, fy, cx, cy, ds)
Row→depth via plane intersection: `t = −(n·c) / (n·ray)` where `ray=[0,(y−cy)/fy,1]`.
Fallback (no plane yet): `z = max(0.3, 1.2*fy/dy)` for `dy = y−cy > 2`.
Guard: if `z_far <= z_near * 1.05` → use `(1.5, GROUND_PLANE_MAX_DEPTH_M)`.
Draws: fillPoly trapezoid (addWeighted α=0.25), then longitudinal lines on top.

### draw_person(img, p, fx, fy, cx, cy, ds, timestamp_s)
Draw order: bearing_ray → skeleton → gaze_ray → joint_dots → label_block.
**Bearing ray**: from `(cx,cy)` to projected `(hd*sin(az), −dist*sin(el), hd*cos(el))`.
**Label block** (6 lines, bottom→top above topmost joint):
```
line 0 (ay):     #pid  conf%                COLOR_LABEL_GAZE
line 1 (ay-g):   gaze:az/el  or gaze:n/a   COLOR_LABEL_GAZE
line 2 (ay-2g):  ver:±XX.X                  COLOR_LABEL_ANGLES
line 3 (ay-3g):  hor:±XX.X                  COLOR_LABEL_ANGLES
line 4 (ay-4g):  spd:X.X km/h               COLOR_LABEL_ANGLES
line 5 (ay-5g):  H:XXXcm                    COLOR_LABEL_DIST
```
`g = round(LABEL_LINE_SPACING_PX * ds)`,  `ax = px_x[topmost_joint] − round(30*ds)`

**Pixel projection**: `px_x = (jm[:,0] / safe_depth) * fx + cx`
where `safe_depth = where(joint_visible, jm[:,2], 1.0)` — invisible joints project to garbage, not drawn.

### draw_hud(img, n, t_s, total_s, rec, paused, cam_src, h, ds)
Row1 = `max(20, round(30*ds))`: status string, colour = rec/paused/running.
Row2 = `max(36, round(56*ds))`: `CAM:<source>`.
Bottom = `h − max(8, round(12*ds))`: keyboard hint.

### main() loop
```
init → VideoCapture → init_camera → compute ds/scale
while True:
  if not paused: cap.read() → process_frame() → log FPS every 1s
  build canvas (copy frame or freeze last_canvas)
  if not paused: draw_ground_plane(); draw_person() per p
  draw_hud()
  if rec: vw.write(canvas); nw.write(DetectionMessage.to_json())
  cv2.imshow → waitKey(100 if paused else 1)
  handle keys: q/Esc quit | Space pause | R rec | A/D seek
```
`ds` = display scale factor: `1.0 + (sqrt(W*H / (1808*1392)) − 1) * 0.6`
— scales drawing sizes proportionally to resolution, anchored at 1808×1392.

---

## Invariants and Constraints

1. `init_camera()` must be called before `process_frame()` — raises RuntimeError otherwise.
2. `_TORSO_JOINTS = [5,6,11,12]` is module-level; never redefined per-call.
3. `compute_3d_pose` returns 4-tuple `(jm, h_m, centroid_3d, (dist,elev,azim))` or `(None,0.,None,None)`. Caller checks `result[0] is None`.
4. `gaze_direction` is always a unit vector or None — never zero-length.
5. Ground plane `normal[1] < 0` always (camera Y = down; normal points up).
6. `PersonTracker` IDs are monotonically increasing integers from 0, never reused.
7. `velocity_m_per_s` is None until `VELOCITY_MIN_SPAN_S` (0.5s) of torso-centroid history exists; never zero-filled before that.
8. `joint_visible[j]` ↔ `joint_scores[j] > MIN_KEYPOINT_CONFIDENCE` — always consistent.
9. `joints_meters` is always shape (17,3) f32; rows for invisible joints contain mathematically projected values but are masked by `joint_visible` before any use.
10. Skeleton edge indices in `SKELETON_EDGES` are 1-based (subtract 1 before indexing).

---

## Camera JSON Format
```json
{
  "camera_intrinsics": {
    "WIDTH": int, "HEIGHT": int, "FPS": float,
    "FX": float, "FY": float, "CX": float, "CY": float,
    "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
    "D": [k1,k2,p1,p2,k3]
  },
  "crop": {"x0_px":int,"y0_px":int,"x1_px":int,"y1_px":int,"percent":float} | null,
  "warnings": {"anamorphic": bool}
}
```
Filename pattern: `camera_{W}x{H}_{device}.json` in `media/`.

---

## Recording Output Format
- **MP4**: full-resolution canvas frames, same fps as input, codec `mp4v`.
- **NDJSON**: one JSON object per frame (only when recording), newline-delimited.
  Schema: `{"header":{"sec":int,"nanosec":int,"frame_id":"CAMERA"},"objects":[...]}`
  Each object: `id, class, score, pose{x,y,z,yaw,pitch,roll,covariance}, size{l,w,h}, velocity{vx,vy,vz,...}, label{text,value}`.
  `size.w=0.45` (shoulder width), `size.l=0.25` (body depth) — constants.
  `pose.yaw` = gaze azimuth (rad) if gaze available, else camera bearing.

---

## Keyboard Controls
| Key | Action |
|-----|--------|
| Space | pause / resume |
| R | toggle recording (MP4 + NDJSON) |
| A | seek −5 s |
| D | seek +5 s |
| Q / Esc | quit, flush recording |
