"""backend.py – inference, tracking, 3-D geometry.

Public API:
    camera_params = init_camera(width, height, fps)
    persons       = process_frame(bgr_frame, timestamp_s) -> List[PersonData]
    ground_plane  = get_ground_plane() -> Optional[GroundPlane]
"""
import math, os, json, warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import onnxruntime as ort
from collections import deque
from pathlib import Path
from typing import Optional, List, Tuple

from scipy.optimize import linear_sum_assignment

from basics import *
from basics import _Ema  # not exported by * due to underscore prefix

# ── Environment ───────────────────────────────────────────────────────────────
MODEL_DIR.mkdir(parents=True, exist_ok=True)
for _k, _v in [("RTMLIB_HOME",          str(MODEL_DIR)),
               ("TORCH_HOME",            str(MODEL_DIR)),
               ("HF_HOME",               str(MODEL_DIR / "hub")),
               ("HUGGINGFACE_HUB_CACHE", str(MODEL_DIR / "hub"))]:
    os.environ[_k] = _v
ort.set_default_logger_severity(4)

print(f"Backend : {DEVICE.upper()}")

from rtmlib import Wholebody3d
_inference_model: Optional["Wholebody3d"] = None  # populated on first call to init_inference()


def init_inference() -> None:
    """Load detection + pose models. Called once from init_camera().
    Safe to call multiple times; no-op if already initialised.
    """
    global _inference_model
    if _inference_model is not None:
        return
    _inference_model = Wholebody3d(mode="balanced", backend="onnxruntime", device=DEVICE)
    _inference_model.det_model.score_thr = DETECTION_SCORE_THRESHOLD
    print(f"Backend : YOLOX score_thr -> {DETECTION_SCORE_THRESHOLD:.2f}")

# Module-level arrays for vectorised De Leva (1996) segment calculations.
# Precomputed once so process_frame() avoids repeated list construction.
_DELEVA_JOINT_A = np.array([seg[0] for seg in DE_LEVA_SEGMENTS], dtype=np.int32)
_DELEVA_JOINT_B = np.array([seg[1] for seg in DE_LEVA_SEGMENTS], dtype=np.int32)
_DELEVA_RATIO   = np.array([seg[2] for seg in DE_LEVA_SEGMENTS], dtype=np.float32)
_TORSO_JOINTS   = (JOINT_LEFT_SHOULDER, JOINT_RIGHT_SHOULDER,
                   JOINT_LEFT_HIP,      JOINT_RIGHT_HIP)
_JOINT_HEAD     = (JOINT_NOSE, JOINT_LEFT_EYE, JOINT_RIGHT_EYE,
                   JOINT_LEFT_EAR, JOINT_RIGHT_EAR)
_JOINT_ANKLES   = (JOINT_LEFT_ANKLE, JOINT_RIGHT_ANKLE)
_JOINT_KNEES    = (JOINT_LEFT_KNEE,  JOINT_RIGHT_KNEE)
_JOINT_HIPS     = (JOINT_LEFT_HIP,   JOINT_RIGHT_HIP)


def _vis_joints(scores: np.ndarray, indices: list) -> list:
    """Return indices from `indices` where score exceeds MIN_KEYPOINT_CONFIDENCE."""
    return [j for j in indices if scores[j] > MIN_KEYPOINT_CONFIDENCE]


# ── Camera helpers ────────────────────────────────────────────────────────────
def _camera_search_dirs(search_dir: Path):
    raw = [search_dir, search_dir.parent, Path.cwd(), SCRIPT_DIR]
    return list(dict.fromkeys(d.resolve() for d in raw if d.exists()))

def find_camera_json(frame_width: int, frame_height: int,
                     search_dir: Path) -> Optional[Path]:
    dirs = _camera_search_dirs(search_dir)
    for w, h in [(frame_width, frame_height),
                 (frame_width, frame_height + 1), (frame_width, frame_height - 1),
                 (frame_width + 1, frame_height), (frame_width - 1, frame_height)]:
        for d in dirs:
            hits = sorted(d.glob(f"camera_{w}x{h}_*.json"))
            if hits:
                if (w, h) != (frame_width, frame_height):
                    print(f" [calibration] codec-padded: "
                          f"video={frame_width}x{frame_height}, using {hits[0].name}")
                return hits[0]
    print(f" [calibration] no filename match for {frame_width}x{frame_height} (+-1); "
          f"scanning JSON contents ...")
    for d in dirs:
        for candidate in sorted(d.glob("camera_*.json")):
            try:
                ci = json.loads(candidate.read_text(encoding="utf-8")
                                ).get("camera_intrinsics", {})
                jw, jh = int(ci.get("WIDTH", 0)), int(ci.get("HEIGHT", 0))
                if abs(jw - frame_width) <= 1 and abs(jh - frame_height) <= 1:
                    print(f" [calibration] level-3 match: {candidate.name} "
                          f"(JSON={jw}x{jh})")
                    return candidate
            except Exception:
                pass
    searched  = [str(d) for d in dirs]
    all_jsons = [str(p) for d in dirs for p in sorted(d.glob("camera_*.json"))]
    print(f" [calibration] no direct match -- dirs: {searched}")
    print(f" [calibration] available JSONs: {all_jsons or 'none'}")
    return None

def _find_scalable_params(frame_width: int, frame_height: int,
                          search_dir: Path) -> Optional[dict]:
    dirs     = _camera_search_dirs(search_dir)
    ar_video = frame_width / frame_height
    for d in dirs:
        for candidate in sorted(d.glob("camera_*.json")):
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                ci   = data.get("camera_intrinsics", {})
                jw, jh = int(ci.get("WIDTH", 0)), int(ci.get("HEIGHT", 0))
                if jw == 0 or jh == 0:
                    continue
                if abs(jw / jh - ar_video) / ar_video > 0.01:
                    continue
                sx, sy = frame_width / jw, frame_height / jh
                print(f" [calibration] level-4 scale: {candidate.name} "
                      f"{jw}x{jh}->{frame_width}x{frame_height} "
                      f"(sx={sx:.4f}, sy={sy:.4f})")
                return {
                    "FX": float(ci["FX"]) * sx, "FY": float(ci["FY"]) * sy,
                    "CX": float(ci["CX"]) * sx, "CY": float(ci["CY"]) * sy,
                    "W": frame_width, "H": frame_height,
                    "FPS": float(ci.get("FPS", 0)),
                    "K": np.array([[float(ci["FX"]) * sx, 0., float(ci["CX"]) * sx],
                                   [0., float(ci["FY"]) * sy, float(ci["CY"]) * sy],
                                   [0., 0., 1.]], dtype=np.float64),
                    "D": np.array(ci.get("D", [0, 0, 0, 0, 0]), dtype=np.float64),
                    "source": (f"{candidate.name} "
                               f"[scaled {jw}x{jh}->{frame_width}x{frame_height}]"),
                    "crop": None,
                }
            except Exception:
                pass
    return None

def load_camera_params(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    ci = data["camera_intrinsics"]
    if data.get("warnings", {}).get("anamorphic"):
        print(f" [{json_path.name}] WARNING: anamorphic sensor (FX != FY).")
    return {
        "FX": float(ci["FX"]), "FY": float(ci["FY"]),
        "CX": float(ci["CX"]), "CY": float(ci["CY"]),
        "W":  int(ci["WIDTH"]), "H": int(ci["HEIGHT"]),
        "FPS": float(ci["FPS"]),
        "K":  np.array(ci["K"], dtype=np.float64),
        "D":  np.array(ci["D"], dtype=np.float64),
        "source": json_path.name, "crop": data.get("crop"),
    }

def fallback_camera_params(frame_width: int, frame_height: int, fps: float) -> dict:
    """Construct camera intrinsics for a perfect pinhole camera.

    Used when no calibration JSON is available. Assumes:
      - principal point at the image centre (CX = W/2, CY = H/2)
      - focal length equal to the image diagonal in pixels, which
        corresponds to a ~53 degree diagonal field of view
      - zero lens distortion
    """
    fx = fy = math.sqrt(frame_width ** 2 + frame_height ** 2)
    cx, cy  = frame_width / 2.0, frame_height / 2.0
    K = np.array([[fx, 0., cx],
                  [0., fy, cy],
                  [0., 0., 1.]], dtype=np.float64)
    return {
        "FX": fx, "FY": fy, "CX": cx, "CY": cy,
        "W": frame_width, "H": frame_height, "FPS": fps,
        "K": K, "D": np.zeros(5, dtype=np.float64),
        "source": "perfect pinhole fallback (no calibration JSON found)",
        "crop": None,
    }

# ── Lens-distortion correction ────────────────────────────────────────────────
def _undistort_keypoints(pixel_coords: np.ndarray,
                         K: np.ndarray,
                         D: np.ndarray) -> np.ndarray:
    """Map distorted pixel coordinates to their undistorted equivalents.

    P=K keeps the output in pixel space (same intrinsics, no normalisation).
    When D is all-zero the result is numerically identical to the input.
    """
    pts = pixel_coords[:, :2].reshape(-1, 1, 2).astype(np.float64)
    out = cv2.undistortPoints(pts, K, D, P=K)
    return out.reshape(-1, 2).astype(np.float32)

# ── Tracker ───────────────────────────────────────────────────────────────────
class PersonTracker:
    """IoU + centroid-distance tracker with Hungarian assignment.

    update() returns List[Tuple[person_id, det_idx]] where det_idx indexes
    into the caller's bounding_boxes list directly.
    """

    def __init__(self,
                 iou_threshold:       float = TRACKER_IOU_THRESHOLD,
                 max_inactive_frames: int   = TRACKER_MAX_INACTIVE_FRAMES,
                 min_hit_streak:      int   = TRACKER_MIN_HIT_STREAK):
        self._tracks:         dict = {}
        self._next_person_id: int  = 0
        self._iou_threshold         = iou_threshold
        self._max_inactive          = max_inactive_frames
        self._min_hit_streak        = min_hit_streak

    @staticmethod
    def _bbox_iou(a, b) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / union if union > 0 else 0.

    @staticmethod
    def _centroid_dist(a, b) -> float:
        return math.hypot((a[0]+a[2])/2 - (b[0]+b[2])/2,
                          (a[1]+a[3])/2 - (b[1]+b[3])/2)

    def update(self,
               bounding_boxes: list,
               confidences:    list) -> List[Tuple[int, int]]:
        for pid in list(self._tracks):
            self._tracks[pid]["inactive_frames"] += 1
            if self._tracks[pid]["inactive_frames"] > self._max_inactive:
                del self._tracks[pid]

        if not bounding_boxes:
            return []

        result:           List[Tuple[int, int]] = []
        matched_det_idxs: set                   = set()
        existing_ids = list(self._tracks.keys())

        if existing_ids:
            n_tr  = len(existing_ids)
            n_det = len(bounding_boxes)
            cost  = np.full((n_tr, n_det), fill_value=1e6, dtype=np.float64)
            for i, pid in enumerate(existing_ids):
                track = self._tracks[pid]
                for j, bbox in enumerate(bounding_boxes):
                    iou   = self._bbox_iou(bbox, track["bbox"])
                    cdist = self._centroid_dist(bbox, track["bbox"])
                    if iou > self._iou_threshold or cdist < TRACKER_CENTROID_DIST_NORM:
                        score      = (iou
                                      + (1. - min(1., cdist / TRACKER_CENTROID_DIST_NORM))
                                      * TRACKER_CENTROID_DIST_WEIGHT)
                        cost[i, j] = -score
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] >= TRACKER_MATCH_GATE:
                    continue
                pid = existing_ids[i]
                self._tracks[pid].update(
                    {"bbox": bounding_boxes[j], "inactive_frames": 0,
                     "conf": confidences[j]})
                self._tracks[pid]["hit_streak"] += 1
                matched_det_idxs.add(j)
                if self._tracks[pid]["hit_streak"] >= self._min_hit_streak:
                    result.append((pid, j))

        for j, (bbox, conf) in enumerate(zip(bounding_boxes, confidences)):
            if j in matched_det_idxs:
                continue
            pid = self._next_person_id
            self._tracks[pid] = {
                "bbox": bbox, "inactive_frames": 0,
                "conf": conf, "hit_streak": 1,
            }
            self._next_person_id += 1
            if self._tracks[pid]["hit_streak"] >= self._min_hit_streak:
                result.append((pid, j))

        return result

# ── PersonProfiles ────────────────────────────────────────────────────────────
class PersonProfiles:
    def __init__(self):
        self._height_history:   dict = {}
        self._height_published: dict = {}  # maps person_id to (timestamp, published_value); throttles height output to HEIGHT_PUBLISH_INTERVAL_S
        self._pos_history:      dict = {}
        self._vel_ema:          dict = {}
        self._ema_pixel_h = _Ema(TAU_PIXEL_HEIGHT_S)
        self._ema_display = _Ema(TAU_DISPLAY_S)

    def push_height_sample(self, person_id: int, height_m: float,
                           head_visible: bool, ankle_visible: bool) -> None:
        if not (head_visible and ankle_visible
                and BODY_HEIGHT_MIN_M <= height_m <= BODY_HEIGHT_MAX_M):
            return
        history = self._height_history.setdefault(
            person_id, deque(maxlen=HEIGHT_BUFFER_WINDOW_FRAMES))
        if len(history) >= HEIGHT_BUFFER_MIN_SAMPLES:
            running_median = float(np.median(history))
            if abs(height_m - running_median) > HEIGHT_OUTLIER_MAX_DEVIATION * running_median:
                return
        history.append(height_m)

    def get_height_reference(self, person_id: int) -> float:
        history = self._height_history.get(person_id, [])
        return (float(np.median(history))
                if len(history) >= HEIGHT_BUFFER_MIN_SAMPLES
                else BODY_HEIGHT_PRIOR_M)

    def get_published_height(self, person_id: int, timestamp_s: float) -> float:
        """Return the height reference throttled to HEIGHT_PUBLISH_INTERVAL_S.
        Internal depth anchor always uses get_height_reference() directly.
        """
        current = self.get_height_reference(person_id)
        cached  = self._height_published.get(person_id)
        if cached is None or timestamp_s - cached[0] >= HEIGHT_PUBLISH_INTERVAL_S:
            self._height_published[person_id] = (timestamp_s, current)
            return current
        return cached[1]

    def smooth_pixel_height(self, person_id: int,
                            pixel_height: float,
                            timestamp_s:  float) -> float:
        return float(self._ema_pixel_h(person_id, pixel_height, timestamp_s))

    def smooth_display_values(self, person_id: int,
                              distance_m:    float,
                              elevation_deg: float,
                              azimuth_deg:   float,
                              timestamp_s:   float) -> tuple:
        prev = self._ema_display._v.get(person_id)
        if prev is not None:
            delta = azimuth_deg - prev[2]
            if   delta >  180.: azimuth_deg -= 360.
            elif delta < -180.: azimuth_deg += 360.
        val = np.array([distance_m, elevation_deg, azimuth_deg], dtype=np.float64)
        out = self._ema_display(person_id, val, timestamp_s)
        return float(out[0]), float(out[1]), float(out[2])

    def compute_velocity(self, person_id: int,
                         centroid_m:  np.ndarray,
                         timestamp_s: float) -> ObjectVelocity:
        buf = self._pos_history.setdefault(
            person_id, deque(maxlen=VELOCITY_BUFFER_SIZE))
        if buf and (timestamp_s - buf[-1][0]) > VELOCITY_MAX_DT_S:
            buf.clear()
            self._vel_ema.pop(person_id, None)
        buf.append((timestamp_s, centroid_m.copy()))
        cutoff  = timestamp_s - VELOCITY_BUFFER_S
        samples = [(t, p) for t, p in buf if t >= cutoff]
        prev    = self._vel_ema.get(person_id)
        if (len(samples) < 3
                or samples[-1][0] - samples[0][0] < VELOCITY_MIN_SPAN_S):
            if prev is None:
                return ObjectVelocity()
            return ObjectVelocity(vx=float(prev[2]),
                                  vy=float(prev[0]),
                                  vz=float(-prev[1]))
        T  = np.array([s[0] for s in samples], dtype=np.float64)
        P  = np.array([s[1] for s in samples], dtype=np.float64)
        Tc = T - T.mean()
        Pc = P - P.mean(axis=0)
        raw     = (Tc @ Pc) / float((Tc ** 2).sum())
        raw_spd = float(np.linalg.norm(raw))
        if raw_spd > VELOCITY_MAX_SPEED_MS:
            raw = raw * (VELOCITY_MAX_SPEED_MS / raw_spd)
        if prev is not None and len(buf) >= 2:
            dt    = float(buf[-1][0] - buf[-2][0])
            alpha = 1.0 - math.exp(-dt / VELOCITY_EMA_TAU_S) if dt > 1e-4 else 0.1
            ema   = alpha * raw + (1.0 - alpha) * prev
        else:
            ema = raw.copy()
        self._vel_ema[person_id] = ema
        # Camera coordinate convention: X=right, Y=down, Z=forward.
        # ObjectVelocity uses X=forward, Y=right, Z=up, so axes are remapped here.
        return ObjectVelocity(vx=float(ema[2]),
                              vy=float(ema[0]),
                              vz=float(-ema[1]))

# ── 3-D geometry ──────────────────────────────────────────────────────────────
def _pixel_body_height_span(pixel_coords, joint_scores) -> Optional[float]:
    """Vertical pixel span from the topmost confident head joint to the lowest
    confident ankle/knee/hip joint.  Operates on undistorted coordinates."""
    top_y = None
    for group in ((JOINT_NOSE,), (JOINT_LEFT_EYE, JOINT_RIGHT_EYE), (JOINT_LEFT_EAR, JOINT_RIGHT_EAR)):
        ys = [float(pixel_coords[j, 1])
              for j in group if joint_scores[j] > MIN_KEYPOINT_CONFIDENCE]
        if ys:
            top_y = float(np.mean(ys)); break
    bot_y = None
    for group in (_JOINT_ANKLES, _JOINT_KNEES, _JOINT_HIPS):
        ys = [float(pixel_coords[j, 1])
              for j in group if joint_scores[j] > MIN_KEYPOINT_CONFIDENCE]
        if ys:
            bot_y = float(np.mean(ys)); break
    return abs(bot_y - top_y) if top_y is not None and bot_y is not None else None


def compute_3d_pose(pixel_coords, kp3d_normalized, joint_scores,
                    focal_x, focal_y, principal_x, principal_y,
                    height_reference_m: float = BODY_HEIGHT_PRIOR_M,
                    pixel_height_hint:  Optional[float] = None):
    """Back-project undistorted 2-D keypoints into metric 3-D camera space.

    Returns (joints_meters, height_m, centroid_3d, (dist, elev_deg, azim_deg))
    or (None, 0., None, None) on failure.
    """
    sa    = joint_scores[_DELEVA_JOINT_A]
    sb    = joint_scores[_DELEVA_JOINT_B]
    span  = np.abs(kp3d_normalized[_DELEVA_JOINT_A, 2]
                   - kp3d_normalized[_DELEVA_JOINT_B, 2])
    cp    = sa * sb
    valid = (sa > MIN_KEYPOINT_CONFIDENCE) & (sb > MIN_KEYPOINT_CONFIDENCE) & (span > 1e-3)
    if not valid.any():
        return None, 0., None, None
    wt = float(cp[valid].sum())
    if wt < 1e-6:
        return None, 0., None, None
    # u2m converts the model's dimensionless z-axis span into metres.
    # It is derived from De Leva segment ratios weighted by joint confidence,
    # anchored to BODY_HEIGHT_PRIOR_M until a personal height estimate is available.
    u2m = float((_DELEVA_RATIO[valid]
                 * (BODY_HEIGHT_PRIOR_M / span[valid])
                 * cp[valid]).sum() / wt)

    head_joints  = _vis_joints(joint_scores, _JOINT_HEAD)
    ankle_joints = _vis_joints(joint_scores, _JOINT_ANKLES)

    measured_height_m = 0.
    if head_joints and ankle_joints:
        measured_height_m = float(np.clip(
            abs(kp3d_normalized[head_joints,  2].mean()
                - kp3d_normalized[ankle_joints, 2].mean())
            * u2m / RTMW3D_HEAD_ANKLE_Z_RATIO,
            BODY_HEIGHT_MIN_M, BODY_HEIGHT_MAX_M))

    pixel_height = (pixel_height_hint
                    if pixel_height_hint and pixel_height_hint > MIN_PIXEL_HEIGHT_SPAN
                    else _pixel_body_height_span(pixel_coords, joint_scores))
    if pixel_height is None or pixel_height < MIN_PIXEL_HEIGHT_SPAN:
        return None, 0., None, None

    knee_joints = _vis_joints(joint_scores, _JOINT_KNEES)
    hip_joints  = _vis_joints(joint_scores, _JOINT_HIPS)
    foot_joints = ankle_joints or knee_joints or hip_joints
    if not head_joints or not foot_joints:
        return None, 0., None, None

    visible_body_ratio = (VISIBLE_BODY_RATIO_ANKLES if ankle_joints else
                          VISIBLE_BODY_RATIO_HIPS   if hip_joints   else
                          VISIBLE_BODY_RATIO_KNEES)
    # Depth anchor via pinhole camera model: a person of known height h at depth z
    # projects to pixel_height = focal_y * h / z, so z = focal_y * h / pixel_height.
    # VISIBLE_BODY_RATIO corrects for partially-occluded bodies.
    z_root = focal_y * height_reference_m * visible_body_ratio / pixel_height

    depth  = np.clip(z_root + kp3d_normalized[:17, 2] * u2m,
                     JOINT_DEPTH_MIN_M, JOINT_DEPTH_MAX_M).astype(np.float32)
    jm     = np.empty((17, 3), dtype=np.float32)
    jm[:, 2] = depth
    jm[:, 0] = (pixel_coords[:, 0] - principal_x) * depth / focal_x
    jm[:, 1] = (pixel_coords[:, 1] - principal_y) * depth / focal_y

    torso_vis   = _vis_joints(joint_scores, _TORSO_JOINTS)
    centroid_3d = (jm[torso_vis].mean(axis=0) if len(torso_vis) >= 2
                   else jm[joint_scores[:17] > MIN_KEYPOINT_CONFIDENCE].mean(axis=0))

    r  = math.sqrt(float(centroid_3d[0])**2
                   + float(centroid_3d[1])**2
                   + float(centroid_3d[2])**2)
    hd = math.sqrt(float(centroid_3d[0])**2 + float(centroid_3d[2])**2)
    return (jm, measured_height_m, centroid_3d,
            (r,
             math.degrees(math.atan2(-float(centroid_3d[1]), hd)),
             math.degrees(math.atan2( float(centroid_3d[0]), float(centroid_3d[2])))))



# ── Gaze ──────────────────────────────────────────────────────────────────────
def compute_gaze_ray_3d(joints_meters: np.ndarray,
                        joint_scores:  np.ndarray) -> tuple:
    """Return (origin_m, unit_direction) from the ear-midpoint to the nose tip.

    All three landmarks (nose, left ear, right ear) must exceed
    MIN_KEYPOINT_CONFIDENCE.  Returns (None, None) otherwise.
    """
    for j in [JOINT_NOSE, JOINT_LEFT_EAR, JOINT_RIGHT_EAR]:
        if joint_scores[j] <= MIN_KEYPOINT_CONFIDENCE:
            return None, None
    ear_mid  = (joints_meters[JOINT_LEFT_EAR] + joints_meters[JOINT_RIGHT_EAR]) * 0.5
    gaze_vec = joints_meters[JOINT_NOSE] - ear_mid
    length   = float(np.linalg.norm(gaze_vec))
    if length < 1e-6:
        return None, None
    return (joints_meters[JOINT_NOSE].copy(),
            (gaze_vec / length).astype(np.float32))



# ── Module state ──────────────────────────────────────────────────────────────
_tracker        = PersonTracker()
_profiles       = PersonProfiles()
_camera_params: Optional[dict] = None

# ── Ground plane ──────────────────────────────────────────────────────────────
_ground_ref_buf: deque                 = deque(maxlen=GROUND_PLANE_SAMPLE_BUFFER)
_last_ground_t:  float                 = -GROUND_PLANE_SAMPLE_INTERVAL_S
_last_fit_t:     float                 = -GROUND_PLANE_FIT_INTERVAL_S
_ground_plane_g: Optional[GroundPlane] = None
_ransac_rng = np.random.default_rng(42)  # fixed seed for reproducible RANSAC plane fits across runs


def _fit_ground_plane() -> None:
    """Fit a ground plane via RANSAC with temporally-decayed inlier refit.

    RANSAC selects the plane with the most inliers within
    GROUND_PLANE_RANSAC_INLIER_DIST metres.  The final plane is refit by
    weighted SVD where each inlier's weight decays as
    exp(-age_s / GROUND_PLANE_TEMPORAL_DECAY_S).  Normal orientation is
    forced to point downward (positive Y in camera frame).
    """
    global _ground_plane_g
    if len(_ground_ref_buf) < GROUND_PLANE_MIN_SAMPLES:
        return

    times = np.array([t for t, _ in _ground_ref_buf], dtype=np.float64)
    pts   = np.array([p for _, p in _ground_ref_buf], dtype=np.float64)
    now   = times[-1]
    n     = len(pts)

    best_inliers: Optional[np.ndarray] = None

    for _ in range(GROUND_PLANE_RANSAC_ITERATIONS):
        idx = _ransac_rng.choice(n, 3, replace=False)
        ab  = pts[idx[1]] - pts[idx[0]]
        ac  = pts[idx[2]] - pts[idx[0]]
        nrm = np.cross(ab, ac)
        nlen = float(np.linalg.norm(nrm))
        if nlen < 1e-6:
            continue
        nrm /= nlen
        d       = -float(np.dot(nrm, pts[idx[0]]))
        dists   = np.abs(pts @ nrm + d)
        inliers = np.where(dists < GROUND_PLANE_RANSAC_INLIER_DIST)[0]
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers

    if best_inliers is None or len(best_inliers) < 3:
        return

    ipts    = pts[best_inliers]
    weights = np.exp(-(now - times[best_inliers]) / GROUND_PLANE_TEMPORAL_DECAY_S)
    weights /= weights.sum()

    centroid = (weights[:, None] * ipts).sum(axis=0)
    centered = ipts - centroid
    _, _, Vt = np.linalg.svd(centered * np.sqrt(weights[:, None]))
    normal   = Vt[-1].astype(np.float32)
    if normal[1] > 0:
        normal = -normal
    if abs(float(normal[1])) < GROUND_PLANE_NORMAL_MIN_Y:
        return

    # Evaluate fit quality: RMSE over inliers (temporally weighted) and inlier ratio.
    # Both are stored in GroundPlane and exposed for downstream confidence gating.
    d_final      = -float(np.dot(normal.astype(np.float64), centroid))
    residuals    = np.abs(ipts @ normal.astype(np.float64) + d_final)
    fit_rmse     = float(np.sqrt((weights * residuals ** 2).sum()))
    inlier_ratio = len(best_inliers) / n

    was_none = _ground_plane_g is None
    _ground_plane_g = GroundPlane(
        centroid=centroid.astype(np.float32),
        normal=normal,
        inlier_ratio=float(inlier_ratio),
        fit_rmse=float(fit_rmse),
    )
    if was_none:
        print(f"Ground plane: first fit "
              f"n_samples={len(best_inliers)} inlier_ratio={inlier_ratio:.2f} "
              f"fit_rmse={fit_rmse:.3f}m normal_y={float(normal[1]):.3f}")


def get_ground_plane() -> Optional[GroundPlane]:
    return _ground_plane_g


# ── Camera init ───────────────────────────────────────────────────────────────
def init_camera(frame_width: int, frame_height: int, fps: float) -> dict:
    global _camera_params
    init_inference()
    json_path = find_camera_json(frame_width, frame_height, CAMERA_JSON_DIR)
    if json_path:
        _camera_params = load_camera_params(json_path)
    else:
        scaled         = _find_scalable_params(frame_width, frame_height, CAMERA_JSON_DIR)
        _camera_params = (scaled if scaled
                          else fallback_camera_params(frame_width, frame_height, fps))
    print(f"Camera : {_camera_params['source']}")
    if _camera_params.get("crop"):
        c = _camera_params["crop"]
        print(f"  crop {c['percent']}% "
              f"({c['x0_px']},{c['y0_px']})-({c['x1_px']},{c['y1_px']})")
    print(f"  FX={_camera_params['FX']:.1f} FY={_camera_params['FY']:.1f}"
          f" CX={_camera_params['CX']:.1f} CY={_camera_params['CY']:.1f}")
    d_norm = float(np.linalg.norm(_camera_params["D"]))
    print(f"  |D|={d_norm:.4f}"
          f"  ({'distortion active' if d_norm > 1e-6 else 'zero – undistort is identity'})")
    return _camera_params


# ── Main inference loop ───────────────────────────────────────────────────────
def process_frame(bgr_frame, timestamp_s: float = 0.0) -> List[PersonData]:
    if _camera_params is None:
        raise RuntimeError("call init_camera() before process_frame()")
    if _inference_model is None:
        raise RuntimeError("call init_camera() before process_frame() – model not loaded")
    try:
        kp3d_all, scores_all, _, kp2d_all = _inference_model(bgr_frame)
    except Exception:
        import traceback; traceback.print_exc(); return []

    K = _camera_params["K"]
    D = _camera_params["D"]

    bboxes, confs, det_data = [], [], []
    for i in range(len(kp3d_all)):
        sc      = scores_all[i, :17].astype(np.float32)
        k3      = kp3d_all[i,  :17].astype(np.float32)
        px_dist = kp2d_all[i,  :17, :2].astype(np.float32)
        px      = _undistort_keypoints(px_dist, K, D)  # identity when distortion coefficients are zero

        if (sc > MIN_KEYPOINT_CONFIDENCE).sum() < MIN_VISIBLE_KEYPOINTS:
            continue
        det_conf = float(sc[sc > MIN_KEYPOINT_CONFIDENCE].mean())
        if det_conf < MIN_DETECTION_MEAN_CONFIDENCE:
            continue
        vis_px = px[sc > MIN_KEYPOINT_CONFIDENCE]
        x1, y1 = vis_px.min(axis=0)
        x2, y2 = vis_px.max(axis=0)
        bboxes.append([x1, y1, x2, y2])
        confs.append(det_conf)
        det_data.append((px, k3, sc))

    fx, fy = _camera_params["FX"], _camera_params["FY"]
    cx, cy = _camera_params["CX"], _camera_params["CY"]
    results: List[PersonData] = []

    for pid, det_idx in _tracker.update(bboxes, confs):
        if det_idx >= len(det_data):
            continue
        px, k3, sc = det_data[det_idx]

        raw_ph = _pixel_body_height_span(px, sc)
        sm_ph  = (_profiles.smooth_pixel_height(pid, raw_ph, timestamp_s)
                  if raw_ph is not None else None)

        result = compute_3d_pose(px, k3, sc, fx, fy, cx, cy,
                                 _profiles.get_height_reference(pid),
                                 pixel_height_hint=sm_ph)
        if result[0] is None:
            continue
        jm, h_m, centroid_3d, (dist, elev, azim) = result

        _profiles.push_height_sample(
            pid, h_m,
            head_visible  = any(sc[j] > MIN_KEYPOINT_CONFIDENCE for j in _JOINT_HEAD),
            ankle_visible = any(sc[j] > MIN_KEYPOINT_CONFIDENCE for j in _JOINT_ANKLES))

        sm_dist, sm_elev, sm_azim = _profiles.smooth_display_values(
            pid, dist, elev, azim, timestamp_s)
        sm_h = _profiles.get_published_height(pid, timestamp_s)

        gaze_origin, gaze_dir = compute_gaze_ray_3d(jm, sc)
        velocity               = _profiles.compute_velocity(pid, centroid_3d, timestamp_s)

        results.append(PersonData(
            person_id=pid,        pixel_coords=px,   joint_scores=sc,
            joints_meters=jm,
            distance_m=sm_dist,   elevation_deg=sm_elev,
            azimuth_deg=sm_azim,  height_m=sm_h,
            gaze_origin_m=gaze_origin, gaze_direction=gaze_dir,
            joint_visible=sc > MIN_KEYPOINT_CONFIDENCE,
            velocity_m_per_s=velocity,
        ))

    global _last_ground_t, _last_fit_t
    if (GROUND_PLANE_ENABLED
            and timestamp_s - _last_ground_t >= GROUND_PLANE_SAMPLE_INTERVAL_S):
        for p in results:
            jv = p.joint_visible
            if jv is None:
                continue
            hip_vis = [j for j in [JOINT_LEFT_HIP, JOINT_RIGHT_HIP] if jv[j]]
            if hip_vis:
                h_ref  = _profiles.get_height_reference(p.person_id)
                gnd_pt = p.joints_meters[hip_vis].mean(axis=0).copy()
                # Estimate the floor position beneath each person by shifting the hip
                # joint downward by the remaining body fraction below the hips.
                # Y+ points downward in camera space.
                gnd_pt[1] += (1.0 - VISIBLE_BODY_RATIO_HIPS) * h_ref
                _ground_ref_buf.append((timestamp_s, gnd_pt))
        _last_ground_t = timestamp_s
        if (GROUND_PLANE_FIT_INTERVAL_S <= 0
                or timestamp_s - _last_fit_t >= GROUND_PLANE_FIT_INTERVAL_S):
            _last_fit_t = timestamp_s
            _fit_ground_plane()

    return results
