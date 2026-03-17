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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

from scipy.optimize import linear_sum_assignment

from basics import (
    SCRIPT_DIR, MODEL_DIR, CAMERA_JSON_DIR, K_FALLBACK,
    MIN_KEYPOINT_CONFIDENCE, MIN_VISIBLE_KEYPOINTS, MIN_DETECTION_MEAN_CONFIDENCE,
    BODY_HEIGHT_PRIOR_M, TAU_PIXEL_HEIGHT_S, TAU_DISPLAY_S,
    DETECTION_SCORE_THRESHOLD,
    JOINT_NOSE, JOINT_LEFT_EAR, JOINT_RIGHT_EAR,
    JOINT_LEFT_HIP, JOINT_RIGHT_HIP,
    JOINT_LEFT_SHOULDER, JOINT_RIGHT_SHOULDER,
    TRACKER_IOU_THRESHOLD, TRACKER_MAX_INACTIVE_FRAMES, TRACKER_MIN_HIT_STREAK,
    TRACKER_CENTROID_DIST_NORM, TRACKER_CENTROID_DIST_WEIGHT,
    HEIGHT_BUFFER_WINDOW_FRAMES, HEIGHT_BUFFER_MIN_SAMPLES, HEIGHT_OUTLIER_MAX_DEVIATION,
    BODY_HEIGHT_MIN_M, BODY_HEIGHT_MAX_M,
    RTMW3D_HEAD_ANKLE_Z_RATIO, JOINT_DEPTH_MIN_M, JOINT_DEPTH_MAX_M,
    VISIBLE_BODY_RATIO_ANKLES, VISIBLE_BODY_RATIO_KNEES, VISIBLE_BODY_RATIO_HIPS,
    MIN_PIXEL_HEIGHT_SPAN, DE_LEVA_SEGMENTS,
    VELOCITY_EMA_TAU_S, VELOCITY_BUFFER_S, VELOCITY_MIN_SPAN_S,
    VELOCITY_MAX_DT_S, VELOCITY_MAX_SPEED_MS, VELOCITY_BUFFER_SIZE,
    GROUND_PLANE_ENABLED, GROUND_PLANE_SAMPLE_INTERVAL_S,
    GROUND_PLANE_SAMPLE_BUFFER, GROUND_PLANE_MIN_SAMPLES,
    GROUND_PLANE_RANSAC_ITERATIONS, GROUND_PLANE_RANSAC_INLIER_DIST,
    GROUND_PLANE_TEMPORAL_DECAY_S,
    GroundPlane,
    MessageHeader, ObjectSize, ObjectPose, ObjectVelocity, ObjectLabel,
    DetectedObject, DetectionMessage,
)

# ── Environment ───────────────────────────────────────────────────────────────
MODEL_DIR.mkdir(parents=True, exist_ok=True)
for _k, _v in [("RTMLIB_HOME",          str(MODEL_DIR)),
               ("TORCH_HOME",            str(MODEL_DIR)),
               ("HF_HOME",               str(MODEL_DIR / "hub")),
               ("HUGGINGFACE_HUB_CACHE", str(MODEL_DIR / "hub"))]:
    os.environ[_k] = _v
ort.set_default_logger_severity(4)

try:
    import torch as _torch
    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
print(f"Backend : {DEVICE.upper()}")

from rtmlib import Wholebody3d
_inference_model = Wholebody3d(mode="balanced", backend="onnxruntime", device=DEVICE)
_inference_model.det_model.score_thr = DETECTION_SCORE_THRESHOLD
print(f"Backend : YOLOX score_thr -> {DETECTION_SCORE_THRESHOLD:.2f}")

_DELEVA_JOINT_A = np.array([seg[0] for seg in DE_LEVA_SEGMENTS], dtype=np.int32)
_DELEVA_JOINT_B = np.array([seg[1] for seg in DE_LEVA_SEGMENTS], dtype=np.int32)
_DELEVA_RATIO   = np.array([seg[2] for seg in DE_LEVA_SEGMENTS], dtype=np.float32)
_TORSO_JOINTS   = [JOINT_LEFT_SHOULDER, JOINT_RIGHT_SHOULDER,
                   JOINT_LEFT_HIP,      JOINT_RIGHT_HIP]

# ── Data structure ────────────────────────────────────────────────────────────
@dataclass
class PersonData:
    person_id:        int
    pixel_coords:     np.ndarray
    joint_scores:     np.ndarray
    joints_meters:    np.ndarray
    distance_m:       float
    elevation_deg:    float
    azimuth_deg:      float
    height_m:         float
    gaze_origin_m:    Optional[np.ndarray]
    gaze_direction:   Optional[np.ndarray]
    joint_visible:    np.ndarray               = None
    velocity_m_per_s: Optional[ObjectVelocity] = None

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
                    "D":      np.zeros(5, dtype=np.float64),
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
    return {
        "FX": K_FALLBACK[0, 0], "FY": K_FALLBACK[1, 1],
        "CX": K_FALLBACK[0, 2], "CY": K_FALLBACK[1, 2],
        "W": frame_width, "H": frame_height, "FPS": fps,
        "K": K_FALLBACK.copy(), "D": np.zeros(5, dtype=np.float64),
        "source": "K_FALLBACK (no calibration JSON found)", "crop": None,
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
_MATCH_GATE = 0.99

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
                if cost[i, j] >= _MATCH_GATE:
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
        self._pixel_height_ema: dict = {}
        self._pixel_height_t:   dict = {}
        self._display_ema:      dict = {}
        self._display_t:        dict = {}
        self._pos_history:      dict = {}
        self._vel_ema:          dict = {}

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

    def smooth_pixel_height(self, person_id: int,
                            pixel_height: float,
                            timestamp_s:  float) -> float:
        prev   = self._pixel_height_ema.get(person_id)
        last_t = self._pixel_height_t.get(person_id)
        if prev is None or last_t is None:
            self._pixel_height_ema[person_id] = pixel_height
            self._pixel_height_t[person_id]   = timestamp_s
            return pixel_height
        dt       = max(0.0, timestamp_s - last_t)
        alpha    = 1.0 - math.exp(-dt / TAU_PIXEL_HEIGHT_S) if dt > 1e-4 else 0.0
        smoothed = alpha * pixel_height + (1.0 - alpha) * prev
        self._pixel_height_ema[person_id] = smoothed
        self._pixel_height_t[person_id]   = timestamp_s
        return smoothed

    def smooth_display_values(self, person_id: int,
                              distance_m:    float,
                              elevation_deg: float,
                              azimuth_deg:   float,
                              height_m:      float,
                              timestamp_s:   float) -> tuple:
        last_t = self._display_t.get(person_id)
        if person_id not in self._display_ema or last_t is None:
            self._display_ema[person_id] = (distance_m, elevation_deg,
                                            azimuth_deg, height_m)
            self._display_t[person_id]   = timestamp_s
            return distance_m, elevation_deg, azimuth_deg, height_m
        dt    = max(0.0, timestamp_s - last_t)
        alpha = 1.0 - math.exp(-dt / TAU_DISPLAY_S) if dt > 1e-4 else 0.0
        pd, pe, pa, ph = self._display_ema[person_id]
        delta = azimuth_deg - pa
        if   delta >  180.: azimuth_deg -= 360.
        elif delta < -180.: azimuth_deg += 360.
        smoothed = (alpha * distance_m    + (1.0 - alpha) * pd,
                    alpha * elevation_deg + (1.0 - alpha) * pe,
                    alpha * azimuth_deg   + (1.0 - alpha) * pa,
                    alpha * height_m      + (1.0 - alpha) * ph)
        self._display_ema[person_id] = smoothed
        self._display_t[person_id]   = timestamp_s
        return smoothed

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
        return ObjectVelocity(vx=float(ema[2]),
                              vy=float(ema[0]),
                              vz=float(-ema[1]))

# ── 3-D geometry ──────────────────────────────────────────────────────────────
def _pixel_body_height_span(pixel_coords, joint_scores) -> Optional[float]:
    """Vertical pixel span from the topmost confident head joint to the lowest
    confident ankle/knee/hip joint.  Operates on undistorted coordinates."""
    top_y = None
    for group in ([0], [1, 2], [3, 4]):
        ys = [float(pixel_coords[j, 1])
              for j in group if joint_scores[j] > MIN_KEYPOINT_CONFIDENCE]
        if ys:
            top_y = float(np.mean(ys)); break
    bot_y = None
    for group in ([15, 16], [13, 14], [11, 12]):
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
    u2m = float((_DELEVA_RATIO[valid]
                 * (BODY_HEIGHT_PRIOR_M / span[valid])
                 * cp[valid]).sum() / wt)

    def _vis(idx): return [j for j in idx if joint_scores[j] > MIN_KEYPOINT_CONFIDENCE]

    head_joints  = _vis([0, 1, 2, 3, 4])
    ankle_joints = _vis([15, 16])

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

    foot_joints = _vis([15, 16, 13, 14, 11, 12])
    if not head_joints or not foot_joints:
        return None, 0., None, None

    visible_body_ratio = (VISIBLE_BODY_RATIO_ANKLES
                          if any(j in foot_joints for j in [15, 16]) else
                          VISIBLE_BODY_RATIO_HIPS
                          if any(j in foot_joints for j in [11, 12]) else
                          VISIBLE_BODY_RATIO_KNEES)
    z_root = focal_y * height_reference_m * visible_body_ratio / pixel_height

    depth  = np.clip(z_root + kp3d_normalized[:17, 2] * u2m,
                     JOINT_DEPTH_MIN_M, JOINT_DEPTH_MAX_M).astype(np.float32)
    jm     = np.empty((17, 3), dtype=np.float32)
    jm[:, 2] = depth
    jm[:, 0] = (pixel_coords[:, 0] - principal_x) * depth / focal_x
    jm[:, 1] = (pixel_coords[:, 1] - principal_y) * depth / focal_y

    torso_vis   = [j for j in _TORSO_JOINTS if joint_scores[j] > MIN_KEYPOINT_CONFIDENCE]
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


def proj3d2d(point_3d, focal_x, focal_y,
             principal_x, principal_y) -> Optional[np.ndarray]:
    if point_3d[2] <= 0.:
        return None
    return np.array([point_3d[0] / point_3d[2] * focal_x + principal_x,
                     point_3d[1] / point_3d[2] * focal_y + principal_y],
                    dtype=np.float32)


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


def gaze_angles_3d(gaze_unit_dir) -> tuple:
    """Return (azimuth_deg, elevation_deg) for a unit gaze direction vector."""
    h = math.hypot(float(gaze_unit_dir[0]), float(gaze_unit_dir[2]))
    return (math.degrees(math.atan2( float(gaze_unit_dir[0]), float(gaze_unit_dir[2]))),
            math.degrees(math.atan2(-float(gaze_unit_dir[1]), h)))


# ── Module state ──────────────────────────────────────────────────────────────
_tracker        = PersonTracker()
_profiles       = PersonProfiles()
_camera_params: Optional[dict] = None

# ── Ground plane ──────────────────────────────────────────────────────────────
_ground_ref_buf: deque                 = deque(maxlen=GROUND_PLANE_SAMPLE_BUFFER)
_last_ground_t:  float                 = -GROUND_PLANE_SAMPLE_INTERVAL_S
_ground_plane_g: Optional[GroundPlane] = None
_ransac_rng                            = np.random.default_rng(42)


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

    _ground_plane_g = GroundPlane(centroid=centroid.astype(np.float32),
                                  normal=normal,
                                  n_samples=int(len(best_inliers)))


def get_ground_plane() -> Optional[GroundPlane]:
    return _ground_plane_g

# ── Detection message ─────────────────────────────────────────────────────────
_BODY_DEPTH_M = 0.25
_SHOULDER_W_M = 0.45
_last_det_msg = None

def _person_to_obj(p: PersonData) -> DetectedObject:
    r, er, ar = p.distance_m, math.radians(p.elevation_deg), math.radians(p.azimuth_deg)
    hd  = r * math.cos(er)
    yaw = (float(math.atan2(float(p.gaze_direction[0]), float(p.gaze_direction[2])))
           if p.gaze_direction is not None else float(ar))
    jv   = p.joint_visible
    conf = (float(p.joint_scores[jv].mean()) if jv is not None and jv.any()
            else float(p.joint_scores.mean()))
    spd  = (math.sqrt(p.velocity_m_per_s.vx**2 + p.velocity_m_per_s.vy**2)
            if p.velocity_m_per_s is not None else 0.0)
    return DetectedObject(
        id=p.person_id, object_class="PEDESTRIAN", score=conf,
        pose=ObjectPose(x=hd * math.cos(ar), y=hd * math.sin(ar),
                        z=r * math.sin(er), yaw=yaw, pitch=0.0, roll=0.0),
        size=ObjectSize(l=_BODY_DEPTH_M, w=_SHOULDER_W_M, h=p.height_m),
        velocity=p.velocity_m_per_s,
        label=ObjectLabel(text=f"H:{p.height_m*100:.0f}cm spd:{spd*3.6:.1f}km/h",
                          value=round(p.height_m, 3)),
    )

def build_detection_message(persons: List[PersonData], timestamp_s: float,
                             frame_id: str = "CAMERA") -> DetectionMessage:
    global _last_det_msg
    sec  = int(timestamp_s)
    nsec = int((timestamp_s - sec) * 1_000_000_000)
    _last_det_msg = DetectionMessage(
        header=MessageHeader(sec=sec, nanosec=nsec, frame_id=frame_id),
        objects=[_person_to_obj(p) for p in persons],
    )
    return _last_det_msg

def get_last_detection_message():
    return _last_det_msg

# ── Camera init ───────────────────────────────────────────────────────────────
def init_camera(frame_width: int, frame_height: int, fps: float) -> dict:
    global _camera_params
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

def get_camera() -> Optional[dict]:
    return _camera_params

# ── Main inference loop ───────────────────────────────────────────────────────
def process_frame(bgr_frame, timestamp_s: float = 0.0) -> List[PersonData]:
    if _camera_params is None:
        raise RuntimeError("call init_camera() before process_frame()")
    try:
        kp3d_all, scores_all, _, kp2d_all = _inference_model(bgr_frame)
    except Exception as e:
        print(f"Inference error: {e}"); return []

    K = _camera_params["K"]
    D = _camera_params["D"]

    bboxes, confs, det_data = [], [], []
    for i in range(len(kp3d_all)):
        sc      = scores_all[i, :17].astype(np.float32)
        k3      = kp3d_all[i,  :17].astype(np.float32)
        px_dist = kp2d_all[i,  :17, :2].astype(np.float32)
        px      = _undistort_keypoints(px_dist, K, D)

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
            head_visible  = any(sc[j] > MIN_KEYPOINT_CONFIDENCE for j in [0, 1, 2, 3, 4]),
            ankle_visible = any(sc[j] > MIN_KEYPOINT_CONFIDENCE for j in [15, 16]))

        sm_dist, sm_elev, sm_azim, sm_h = _profiles.smooth_display_values(
            pid, dist, elev, azim,
            _profiles.get_height_reference(pid), timestamp_s)

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

    global _last_ground_t
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
                gnd_pt[1] += (1.0 - VISIBLE_BODY_RATIO_HIPS) * h_ref
                _ground_ref_buf.append((timestamp_s, gnd_pt))
        _last_ground_t = timestamp_s
        _fit_ground_plane()

    return results
