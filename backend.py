"""backend.py – inference, tracking, 3-D geometry.

Public API:
    camera_params = init_camera(width, height, fps)
    persons       = process_frame(bgr_frame, timestamp_s)  -> List[PersonData]
    ground_plane  = get_ground_plane()                      -> Optional[GroundPlane]
"""
import math, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import onnxruntime as ort
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from config import (
    SCRIPT_DIR, MODEL_DIR, CAMERA_JSON_DIR, K_FALLBACK,
    MIN_KEYPOINT_CONFIDENCE, MIN_VISIBLE_KEYPOINTS, MIN_DETECTION_MEAN_CONFIDENCE,
    BODY_HEIGHT_PRIOR_M, EMA_ALPHA_PIXEL_HEIGHT, EMA_ALPHA_DISPLAY,
    DETECTION_SCORE_THRESHOLD,
    JOINT_NOSE, JOINT_LEFT_EAR, JOINT_RIGHT_EAR,
    JOINT_LEFT_SHOULDER, JOINT_RIGHT_SHOULDER,
    JOINT_LEFT_HIP, JOINT_RIGHT_HIP,
    JOINT_LEFT_KNEE, JOINT_RIGHT_KNEE,
    JOINT_LEFT_ANKLE, JOINT_RIGHT_ANKLE,
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
    GroundPlane,
    MessageHeader, ObjectSize, ObjectPose, ObjectVelocity, ObjectLabel,
    DetectedObject, DetectionMessage,
)

# ── Environment ───────────────────────────────────────────────────────────────
MODEL_DIR.mkdir(parents=True, exist_ok=True)
for _k, _v in [("RTMLIB_HOME",           str(MODEL_DIR)),
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
    joint_visible:    np.ndarray = None
    velocity_m_per_s: Optional[ObjectVelocity] = None


# ── Camera helpers ────────────────────────────────────────────────────────────
def _camera_search_dirs(search_dir: Path):
    raw = [search_dir, search_dir.parent, Path.cwd(), SCRIPT_DIR]
    return list(dict.fromkeys(d.resolve() for d in raw if d.exists()))


def find_camera_json(frame_width: int, frame_height: int,
                     search_dir: Path) -> Optional[Path]:
    dirs = _camera_search_dirs(search_dir)
    for w, h in [(frame_width, frame_height),
                 (frame_width, frame_height+1), (frame_width, frame_height-1),
                 (frame_width+1, frame_height), (frame_width-1, frame_height)]:
        for d in dirs:
            hits = sorted(d.glob(f"camera_{w}x{h}_*.json"))
            if hits:
                if (w, h) != (frame_width, frame_height):
                    print(f"  [calibration] codec-padded: "
                          f"video={frame_width}x{frame_height}, using {hits[0].name}")
                return hits[0]
    print(f"  [calibration] no filename match for {frame_width}x{frame_height} (+-1); "
          f"scanning JSON contents ...")
    for d in dirs:
        for candidate in sorted(d.glob("camera_*.json")):
            try:
                ci = json.loads(candidate.read_text(encoding="utf-8")
                                ).get("camera_intrinsics", {})
                jw, jh = int(ci.get("WIDTH", 0)), int(ci.get("HEIGHT", 0))
                if abs(jw - frame_width) <= 1 and abs(jh - frame_height) <= 1:
                    print(f"  [calibration] level-3 match: {candidate.name} "
                          f"(JSON={jw}x{jh})")
                    return candidate
            except Exception:
                pass
    searched  = [str(d) for d in dirs]
    all_jsons = [str(p) for d in dirs for p in sorted(d.glob("camera_*.json"))]
    print(f"  [calibration] no direct match -- dirs: {searched}")
    print(f"  [calibration] available JSONs: {all_jsons or 'none'}")
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
                if jw == 0 or jh == 0: continue
                if abs(jw/jh - ar_video) / ar_video > 0.01: continue
                sx = frame_width / jw; sy = frame_height / jh
                print(f"  [calibration] level-4 scale: {candidate.name} "
                      f"{jw}x{jh}->{frame_width}x{frame_height} "
                      f"(sx={sx:.4f}, sy={sy:.4f})")
                return {
                    "FX": float(ci["FX"])*sx, "FY": float(ci["FY"])*sy,
                    "CX": float(ci["CX"])*sx, "CY": float(ci["CY"])*sy,
                    "W": frame_width, "H": frame_height,
                    "FPS": float(ci.get("FPS", 0)),
                    "K": np.array([[float(ci["FX"])*sx, 0., float(ci["CX"])*sx],
                                   [0., float(ci["FY"])*sy, float(ci["CY"])*sy],
                                   [0., 0., 1.]], dtype=np.float64),
                    "D": np.zeros(5, dtype=np.float64),
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
        print(f"  [{json_path.name}] WARNING: anamorphic sensor (FX != FY).")
    return {
        "FX": float(ci["FX"]), "FY": float(ci["FY"]),
        "CX": float(ci["CX"]), "CY": float(ci["CY"]),
        "W": int(ci["WIDTH"]), "H": int(ci["HEIGHT"]),
        "FPS": float(ci["FPS"]),
        "K": np.array(ci["K"], dtype=np.float64),
        "D": np.array(ci["D"], dtype=np.float64),
        "source": json_path.name, "crop": data.get("crop"),
    }


def fallback_camera_params(frame_width: int, frame_height: int, fps: float) -> dict:
    return {
        "FX": K_FALLBACK[0,0], "FY": K_FALLBACK[1,1],
        "CX": K_FALLBACK[0,2], "CY": K_FALLBACK[1,2],
        "W": frame_width, "H": frame_height, "FPS": fps,
        "K": K_FALLBACK.copy(), "D": np.zeros(5, dtype=np.float64),
        "source": "K_FALLBACK (no calibration JSON found)", "crop": None,
    }


# ── Tracker ───────────────────────────────────────────────────────────────────
class PersonTracker:
    """IoU + centroid-distance multi-person tracker with hit-streak hysteresis."""

    def __init__(self,
                 iou_threshold=TRACKER_IOU_THRESHOLD,
                 max_inactive_frames=TRACKER_MAX_INACTIVE_FRAMES,
                 min_hit_streak=TRACKER_MIN_HIT_STREAK):
        self._tracks:        dict = {}
        self._next_person_id = 0
        self._iou_threshold  = iou_threshold
        self._max_inactive   = max_inactive_frames
        self._min_hit_streak = min_hit_streak

    @staticmethod
    def _bbox_iou(box_a, box_b) -> float:
        ix1 = max(box_a[0], box_b[0]); iy1 = max(box_a[1], box_b[1])
        ix2 = min(box_a[2], box_b[2]); iy2 = min(box_a[3], box_b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        union = ((box_a[2]-box_a[0])*(box_a[3]-box_a[1])
               + (box_b[2]-box_b[0])*(box_b[3]-box_b[1]) - inter)
        return inter / union if union > 0 else 0.

    @staticmethod
    def _centroid_dist(box_a, box_b) -> float:
        return math.hypot((box_a[0]+box_a[2])/2 - (box_b[0]+box_b[2])/2,
                          (box_a[1]+box_a[3])/2 - (box_b[1]+box_b[3])/2)

    def update(self, bounding_boxes: list, confidences: list) -> list:
        for pid in list(self._tracks):
            self._tracks[pid]["inactive_frames"] += 1
            if self._tracks[pid]["inactive_frames"] > self._max_inactive:
                del self._tracks[pid]
        if not bounding_boxes:
            return [(pid, self._tracks[pid]["conf"])
                    for pid in self._tracks
                    if self._tracks[pid]["hit_streak"] >= self._min_hit_streak]
        existing_ids   = list(self._tracks.keys())
        matched_ids, result_ids = [], []
        for bbox, conf in zip(bounding_boxes, confidences):
            best_score, best_pid = 0., None
            for pid in existing_ids:
                if pid in matched_ids: continue
                track = self._tracks[pid]
                iou   = self._bbox_iou(bbox, track["bbox"])
                cdist = self._centroid_dist(bbox, track["bbox"])
                score = (iou + (1 - min(1., cdist / TRACKER_CENTROID_DIST_NORM))
                         * TRACKER_CENTROID_DIST_WEIGHT)
                if (score > best_score
                        and (iou > self._iou_threshold
                             or cdist < TRACKER_CENTROID_DIST_NORM)):
                    best_score, best_pid = score, pid
            if best_pid is not None:
                self._tracks[best_pid].update(
                    {"bbox": bbox, "inactive_frames": 0, "conf": conf})
                self._tracks[best_pid]["hit_streak"] += 1
                matched_ids.append(best_pid)
                result_ids.append(best_pid)
            else:
                self._tracks[self._next_person_id] = {
                    "bbox": bbox, "inactive_frames": 0,
                    "conf": conf, "hit_streak": 1}
                result_ids.append(self._next_person_id)
                self._next_person_id += 1
        return [(pid, self._tracks[pid]["conf"])
                for pid in result_ids
                if pid in self._tracks
                and self._tracks[pid]["hit_streak"] >= self._min_hit_streak]


# ── PersonProfiles ────────────────────────────────────────────────────────────
class PersonProfiles:
    """Per-person temporal smoothers: height, pixel-span, display EMA, velocity."""

    def __init__(self):
        self._height_history:   dict = {}
        self._pixel_height_ema: dict = {}
        self._display_ema:      dict = {}
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
            median = float(np.median(history))
            if abs(height_m - median) > HEIGHT_OUTLIER_MAX_DEVIATION * median:
                return
        history.append(height_m)

    def get_height_reference(self, person_id: int) -> float:
        history = self._height_history.get(person_id, [])
        return (float(np.median(history))
                if len(history) >= HEIGHT_BUFFER_MIN_SAMPLES
                else BODY_HEIGHT_PRIOR_M)

    def smooth_pixel_height(self, person_id: int, pixel_height: float) -> float:
        prev     = self._pixel_height_ema.get(person_id, pixel_height)
        smoothed = EMA_ALPHA_PIXEL_HEIGHT * pixel_height + (1. - EMA_ALPHA_PIXEL_HEIGHT) * prev
        self._pixel_height_ema[person_id] = smoothed
        return smoothed

    def smooth_display_values(self, person_id: int,
                              distance_m: float, elevation_deg: float,
                              azimuth_deg: float, height_m: float) -> tuple:
        if person_id not in self._display_ema:
            self._display_ema[person_id] = (distance_m, elevation_deg,
                                            azimuth_deg, height_m)
            return distance_m, elevation_deg, azimuth_deg, height_m
        pd, pe, pa, ph = self._display_ema[person_id]
        da = azimuth_deg - pa
        if   da >  180.: azimuth_deg -= 360.
        elif da < -180.: azimuth_deg += 360.
        a = EMA_ALPHA_DISPLAY
        s = (a*distance_m + (1-a)*pd, a*elevation_deg + (1-a)*pe,
             a*azimuth_deg + (1-a)*pa, a*height_m     + (1-a)*ph)
        self._display_ema[person_id] = s
        return s

    def compute_velocity(self, person_id: int,
                         centroid_m: np.ndarray,
                         timestamp_s: float) -> ObjectVelocity:
        """OLS regression velocity over VELOCITY_BUFFER_S seconds of torso-centroid history.

        Time-based EMA (τ=VELOCITY_EMA_TAU_S) applied to the OLS result.
        Hard cap at VELOCITY_MAX_SPEED_MS before EMA to block outliers.
        Buffer is flushed on timestamp gaps > VELOCITY_MAX_DT_S (seek/pause).
        """
        buf = self._pos_history.setdefault(
            person_id, deque(maxlen=VELOCITY_BUFFER_SIZE))

        if buf and (timestamp_s - buf[-1][0]) > VELOCITY_MAX_DT_S:
            buf.clear()
            self._vel_ema.pop(person_id, None)

        buf.append((timestamp_s, centroid_m.copy()))

        cutoff  = timestamp_s - VELOCITY_BUFFER_S
        samples = [(tt, pp) for tt, pp in buf if tt >= cutoff]
        prev    = self._vel_ema.get(person_id)

        if (len(samples) < 3
                or samples[-1][0] - samples[0][0] < VELOCITY_MIN_SPAN_S):
            if prev is None:
                return ObjectVelocity()
            return ObjectVelocity(
                vx=float(prev[2]), vy=float(prev[0]), vz=float(-prev[1]))

        T  = np.array([s[0] for s in samples], dtype=np.float64)
        P  = np.array([s[1] for s in samples], dtype=np.float64)
        Tc = T - T.mean(); Pc = P - P.mean(axis=0)
        raw = (Tc @ Pc) / float((Tc**2).sum())

        spd = float(np.linalg.norm(raw))
        if spd > VELOCITY_MAX_SPEED_MS:
            raw = raw * (VELOCITY_MAX_SPEED_MS / spd)

        if prev is not None and len(buf) >= 2:
            dt    = float(buf[-1][0] - buf[-2][0])
            alpha = 1.0 - math.exp(-dt / VELOCITY_EMA_TAU_S) if dt > 1e-4 else 0.1
            ema   = alpha * raw + (1.0 - alpha) * prev
        else:
            ema = raw.copy()
        self._vel_ema[person_id] = ema

        return ObjectVelocity(
            vx=float(ema[2]), vy=float(ema[0]), vz=float(-ema[1]))


# ── 3-D geometry ──────────────────────────────────────────────────────────────
def _pixel_body_height_span(pixel_coords, joint_scores) -> Optional[float]:
    top_y = None
    for grp in ([0], [1, 2], [3, 4]):
        ys = [float(pixel_coords[j, 1])
              for j in grp if joint_scores[j] > MIN_KEYPOINT_CONFIDENCE]
        if ys: top_y = float(np.mean(ys)); break
    bot_y = None
    for grp in ([15, 16], [13, 14], [11, 12]):
        ys = [float(pixel_coords[j, 1])
              for j in grp if joint_scores[j] > MIN_KEYPOINT_CONFIDENCE]
        if ys: bot_y = float(np.mean(ys)); break
    return abs(bot_y - top_y) if top_y is not None and bot_y is not None else None


def compute_3d_pose(pixel_coords, kp3d_normalized, joint_scores,
                    focal_x, focal_y, principal_x, principal_y,
                    height_reference_m=BODY_HEIGHT_PRIOR_M,
                    pixel_height_hint=None):
    """Returns (joints_meters, height_m, centroid_3d, (dist_m, elev_deg, azim_deg))
    or (None, 0., None, None) on failure."""
    sa = joint_scores[_DELEVA_JOINT_A]; sb = joint_scores[_DELEVA_JOINT_B]
    span   = np.abs(kp3d_normalized[_DELEVA_JOINT_A, 2]
                    - kp3d_normalized[_DELEVA_JOINT_B, 2])
    cp     = sa * sb
    valid  = (sa > MIN_KEYPOINT_CONFIDENCE) & (sb > MIN_KEYPOINT_CONFIDENCE) & (span > 1e-3)
    if not valid.any(): return None, 0., None, None
    wt = float(cp[valid].sum())
    if wt < 1e-6:       return None, 0., None, None
    u2m = float((_DELEVA_RATIO[valid]
                 * (BODY_HEIGHT_PRIOR_M / span[valid])
                 * cp[valid]).sum() / wt)

    def vis(idx): return [j for j in idx if joint_scores[j] > MIN_KEYPOINT_CONFIDENCE]
    head   = vis([0,1,2,3,4]); ankles = vis([15,16])
    h_m    = 0.
    if head and ankles:
        h_m = float(np.clip(
            abs(kp3d_normalized[head, 2].mean() - kp3d_normalized[ankles, 2].mean())
            * u2m / RTMW3D_HEAD_ANKLE_Z_RATIO,
            BODY_HEIGHT_MIN_M, BODY_HEIGHT_MAX_M))

    px_h = (pixel_height_hint
            if pixel_height_hint and pixel_height_hint > MIN_PIXEL_HEIGHT_SPAN
            else _pixel_body_height_span(pixel_coords, joint_scores))
    if px_h is None or px_h < MIN_PIXEL_HEIGHT_SPAN:
        return None, 0., None, None
    feet = vis([15,16,13,14,11,12])
    if not head or not feet:
        return None, 0., None, None
    ratio  = (VISIBLE_BODY_RATIO_ANKLES if any(j in feet for j in [15,16]) else
              VISIBLE_BODY_RATIO_HIPS   if any(j in feet for j in [11,12]) else
              VISIBLE_BODY_RATIO_KNEES)
    z_root = focal_y * height_reference_m * ratio / px_h

    depth  = np.clip(z_root + kp3d_normalized[:17, 2] * u2m,
                     JOINT_DEPTH_MIN_M, JOINT_DEPTH_MAX_M).astype(np.float32)
    jm     = np.empty((17, 3), dtype=np.float32)
    jm[:,2] = depth
    jm[:,0] = (pixel_coords[:,0] - principal_x) * depth / focal_x
    jm[:,1] = (pixel_coords[:,1] - principal_y) * depth / focal_y

    torso_vis = [j for j in _TORSO_JOINTS if joint_scores[j] > MIN_KEYPOINT_CONFIDENCE]
    centroid  = (jm[torso_vis].mean(axis=0) if len(torso_vis) >= 2
                 else jm[joint_scores[:17] > MIN_KEYPOINT_CONFIDENCE].mean(axis=0))
    r  = math.sqrt(float(centroid[0])**2 + float(centroid[1])**2 + float(centroid[2])**2)
    hd = math.sqrt(float(centroid[0])**2 + float(centroid[2])**2)
    return (jm, h_m, centroid,
            (r,
             math.degrees(math.atan2(-float(centroid[1]), hd)),
             math.degrees(math.atan2( float(centroid[0]), float(centroid[2])))))


def proj3d2d(point_3d, focal_x, focal_y, principal_x, principal_y) -> Optional[np.ndarray]:
    if point_3d[2] <= 0.: return None
    return np.array([point_3d[0] / point_3d[2] * focal_x + principal_x,
                     point_3d[1] / point_3d[2] * focal_y + principal_y],
                    dtype=np.float32)


def compute_gaze_ray_3d(joints_meters, joint_scores):
    for j in [JOINT_NOSE, JOINT_LEFT_EAR, JOINT_RIGHT_EAR]:
        if joint_scores[j] <= MIN_KEYPOINT_CONFIDENCE:
            return None, None
    ear_mid  = (joints_meters[JOINT_LEFT_EAR] + joints_meters[JOINT_RIGHT_EAR]) * 0.5
    gaze_vec = joints_meters[JOINT_NOSE] - ear_mid
    length   = float(np.linalg.norm(gaze_vec))
    if length < 1e-6: return None, None
    return joints_meters[JOINT_NOSE].copy(), (gaze_vec / length).astype(np.float32)


def gaze_angles_3d(gaze_unit_dir) -> tuple:
    horiz = math.hypot(float(gaze_unit_dir[0]), float(gaze_unit_dir[2]))
    return (math.degrees(math.atan2( float(gaze_unit_dir[0]), float(gaze_unit_dir[2]))),
            math.degrees(math.atan2(-float(gaze_unit_dir[1]), horiz)))


# ── Module state ──────────────────────────────────────────────────────────────
_tracker        = PersonTracker()
_profiles       = PersonProfiles()
_camera_params: Optional[dict] = None

_ground_ref_buf: deque                 = deque(maxlen=GROUND_PLANE_SAMPLE_BUFFER)
_last_ground_t:  float                 = -GROUND_PLANE_SAMPLE_INTERVAL_S
_ground_plane_g: Optional[GroundPlane] = None


def _fit_ground_plane() -> None:
    global _ground_plane_g
    if len(_ground_ref_buf) < GROUND_PLANE_MIN_SAMPLES: return
    pts      = np.array(_ground_ref_buf, dtype=np.float64)
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    normal   = Vt[-1].astype(np.float32)
    if normal[1] > 0: normal = -normal
    _ground_plane_g = GroundPlane(centroid=centroid.astype(np.float32),
                                   normal=normal, n_samples=len(_ground_ref_buf))


def get_ground_plane() -> Optional[GroundPlane]:
    return _ground_plane_g


# ── Detection message ─────────────────────────────────────────────────────────
_BODY_DEPTH_M = 0.25
_SHOULDER_W_M = 0.45
_last_det_msg = None


def _person_to_obj(p: PersonData) -> DetectedObject:
    r, er, ar = p.distance_m, math.radians(p.elevation_deg), math.radians(p.azimuth_deg)
    hd   = r * math.cos(er)
    yaw  = (float(math.atan2(float(p.gaze_direction[0]), float(p.gaze_direction[2])))
            if p.gaze_direction is not None else float(ar))
    jv   = p.joint_visible
    conf = (float(p.joint_scores[jv].mean()) if jv is not None and jv.any()
            else float(p.joint_scores.mean()))
    spd  = (math.sqrt(p.velocity_m_per_s.vx**2 + p.velocity_m_per_s.vy**2)
            if p.velocity_m_per_s is not None else 0.0)
    return DetectedObject(
        id=p.person_id, object_class="PEDESTRIAN", score=conf,
        pose=ObjectPose(x=hd*math.cos(ar), y=hd*math.sin(ar),
                        z=r*math.sin(er), yaw=yaw, pitch=0.0, roll=0.0),
        size=ObjectSize(l=_BODY_DEPTH_M, w=_SHOULDER_W_M, h=p.height_m),
        velocity=p.velocity_m_per_s,
        label=ObjectLabel(text=f"H:{p.height_m*100:.0f}cm spd:{spd*3.6:.1f}km/h",
                          value=round(p.height_m, 3)),
    )


def build_detection_message(persons, timestamp_s: float,
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


def init_camera(frame_width: int, frame_height: int, fps: float) -> dict:
    global _camera_params
    json_path = find_camera_json(frame_width, frame_height, CAMERA_JSON_DIR)
    if json_path:
        _camera_params = load_camera_params(json_path)
    else:
        scaled = _find_scalable_params(frame_width, frame_height, CAMERA_JSON_DIR)
        _camera_params = (scaled if scaled
                          else fallback_camera_params(frame_width, frame_height, fps))
    print(f"Camera  : {_camera_params['source']}")
    if _camera_params.get("crop"):
        crop = _camera_params["crop"]
        print(f"  crop {crop['percent']}% "
              f"({crop['x0_px']},{crop['y0_px']})-({crop['x1_px']},{crop['y1_px']})")
    print(f"  FX={_camera_params['FX']:.1f}  FY={_camera_params['FY']:.1f}"
          f"  CX={_camera_params['CX']:.1f}  CY={_camera_params['CY']:.1f}")
    return _camera_params


def get_camera() -> Optional[dict]:
    return _camera_params


def process_frame(bgr_frame, timestamp_s: float = 0.0) -> List[PersonData]:
    if _camera_params is None:
        raise RuntimeError("call init_camera() before process_frame()")
    try:
        kp3d_all, scores_all, _, kp2d_all = _inference_model(bgr_frame)
    except Exception as e:
        print(f"Inference error: {e}"); return []

    bboxes, confs, det_data = [], [], []
    for i in range(len(kp3d_all)):
        px   = kp2d_all[i, :17, :2].astype(np.float32)
        k3d  = kp3d_all[i, :17].astype(np.float32)
        sc   = scores_all[i, :17].astype(np.float32)
        if (sc > MIN_KEYPOINT_CONFIDENCE).sum() < MIN_VISIBLE_KEYPOINTS: continue
        det_conf = float(sc[sc > MIN_KEYPOINT_CONFIDENCE].mean())
        if det_conf < MIN_DETECTION_MEAN_CONFIDENCE: continue
        vis_px = px[sc > MIN_KEYPOINT_CONFIDENCE]
        x1, y1 = vis_px.min(axis=0); x2, y2 = vis_px.max(axis=0)
        bboxes.append([x1, y1, x2, y2])
        confs.append(det_conf)
        det_data.append((px, k3d, sc))

    results: List[PersonData] = []
    for det_idx, (person_id, _) in enumerate(_tracker.update(bboxes, confs)):
        if det_idx >= len(det_data): break
        px, k3d, sc = det_data[det_idx]

        raw_ph  = _pixel_body_height_span(px, sc)
        sm_ph   = _profiles.smooth_pixel_height(person_id, raw_ph) if raw_ph else None
        result  = compute_3d_pose(
            px, k3d, sc,
            _camera_params["FX"], _camera_params["FY"],
            _camera_params["CX"], _camera_params["CY"],
            _profiles.get_height_reference(person_id),
            pixel_height_hint=sm_ph)

        if result[0] is None: continue
        jm, h_m, centroid_3d, (dist, elev, azim) = result

        _profiles.push_height_sample(
            person_id, h_m,
            head_visible=any(sc[j] > MIN_KEYPOINT_CONFIDENCE for j in [0,1,2,3,4]),
            ankle_visible=any(sc[j] > MIN_KEYPOINT_CONFIDENCE for j in [15,16]))

        sm_dist, sm_elev, sm_azim, sm_h = _profiles.smooth_display_values(
            person_id, dist, elev, azim, _profiles.get_height_reference(person_id))

        gaze_origin, gaze_dir = compute_gaze_ray_3d(jm, sc)
        velocity = _profiles.compute_velocity(person_id, centroid_3d, timestamp_s)

        results.append(PersonData(
            person_id=person_id, pixel_coords=px, joint_scores=sc,
            joints_meters=jm, distance_m=sm_dist, elevation_deg=sm_elev,
            azimuth_deg=sm_azim, height_m=sm_h,
            gaze_origin_m=gaze_origin, gaze_direction=gaze_dir,
            joint_visible=sc > MIN_KEYPOINT_CONFIDENCE,
            velocity_m_per_s=velocity,
        ))

    global _last_ground_t
    if GROUND_PLANE_ENABLED and timestamp_s - _last_ground_t >= GROUND_PLANE_SAMPLE_INTERVAL_S:
        for p in results:
            jv = p.joint_visible
            if jv is None: continue
            ref = ([j for j in [JOINT_LEFT_ANKLE, JOINT_RIGHT_ANKLE] if jv[j]] or
                   [j for j in [JOINT_LEFT_KNEE,  JOINT_RIGHT_KNEE]  if jv[j]] or
                   [j for j in [JOINT_LEFT_HIP,   JOINT_RIGHT_HIP]   if jv[j]])
            for j in ref:
                _ground_ref_buf.append(p.joints_meters[j].copy())
        _last_ground_t = timestamp_s
        _fit_ground_plane()
    return results