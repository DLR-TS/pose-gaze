"""backend.py – inference, tracking, 3-D geometry.

Public API:
    camera_params = init_camera(width, height, fps)
    persons       = process_frame(bgr_frame)  -> List[PersonData]
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
    MIN_KEYPOINT_CONFIDENCE, MIN_VISIBLE_KEYPOINTS, BODY_HEIGHT_PRIOR_M,
    EMA_ALPHA_PIXEL_HEIGHT, EMA_ALPHA_DISPLAY,
    JOINT_NOSE, JOINT_LEFT_EAR, JOINT_RIGHT_EAR,
    TRACKER_IOU_THRESHOLD, TRACKER_MAX_INACTIVE_FRAMES, TRACKER_MIN_HIT_STREAK,
    TRACKER_CENTROID_DIST_NORM, TRACKER_CENTROID_DIST_WEIGHT,
    HEIGHT_BUFFER_WINDOW_FRAMES, HEIGHT_BUFFER_MIN_SAMPLES, HEIGHT_OUTLIER_MAX_DEVIATION,
    BODY_HEIGHT_MIN_M, BODY_HEIGHT_MAX_M,
    RTMW3D_HEAD_ANKLE_Z_RATIO, JOINT_DEPTH_MIN_M, JOINT_DEPTH_MAX_M,
    VISIBLE_BODY_RATIO_ANKLES, VISIBLE_BODY_RATIO_KNEES, VISIBLE_BODY_RATIO_HIPS,
    MIN_PIXEL_HEIGHT_SPAN, DE_LEVA_SEGMENTS,
)

# ── Environment ───────────────────────────────────────────────────────────────
MODEL_DIR.mkdir(parents=True, exist_ok=True)
for _env_key, _env_val in [
        ("RTMLIB_HOME",          str(MODEL_DIR)),
        ("TORCH_HOME",            str(MODEL_DIR)),
        ("HF_HOME",               str(MODEL_DIR / "hub")),
        ("HUGGINGFACE_HUB_CACHE", str(MODEL_DIR / "hub"))]:
    os.environ[_env_key] = _env_val
ort.set_default_logger_severity(4)

try:
    import torch as _torch
    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
print(f"Backend : {DEVICE.upper()}")

from rtmlib import Wholebody3d
_inference_model = Wholebody3d(mode="balanced", backend="onnxruntime", device=DEVICE)

# Precomputed index arrays for vectorised De Leva scale estimation
_DELEVA_JOINT_A  = np.array([seg[0] for seg in DE_LEVA_SEGMENTS], dtype=np.int32)
_DELEVA_JOINT_B  = np.array([seg[1] for seg in DE_LEVA_SEGMENTS], dtype=np.int32)
_DELEVA_RATIO    = np.array([seg[2] for seg in DE_LEVA_SEGMENTS], dtype=np.float32)


# ── Data structure ────────────────────────────────────────────────────────────
@dataclass
class PersonData:
    """Per-person 3-D result for one frame; scalar fields are EMA-smoothed."""
    person_id:    int
    pixel_coords: np.ndarray            # (17,2) f32  2-D keypoints from model
    joint_scores: np.ndarray            # (17,)  f32  per-joint confidence [0,1]
    joints_meters: np.ndarray           # (17,3) f32  camera-3D positions [m]
    distance_m:   float                 # radial distance to centroid [m]
    elevation_deg: float                # elevation angle [°], positive = above horizon
    azimuth_deg:  float                 # azimuth angle [°], positive = right of camera
    height_m:     float                 # estimated body height [m]
    gaze_origin_m: Optional[np.ndarray] # gaze ray origin (nose) in camera-3D [m]
    gaze_direction: Optional[np.ndarray]# unit gaze direction in camera-3D


# ── Camera helpers ────────────────────────────────────────────────────────────
def find_camera_json(frame_width: int, frame_height: int,
                     search_dir: Path) -> Optional[Path]:
    """Return the first matching camera_{W}x{H}_*.json in search_dir, or None."""
    matches = sorted(search_dir.glob(f"camera_{frame_width}x{frame_height}_*.json"))
    if len(matches) > 1:
        print(f"  {len(matches)} calibration JSONs for "
              f"{frame_width}x{frame_height}; using {matches[0].name}")
    return matches[0] if matches else None

def load_camera_params(json_path: Path) -> dict:
    """Parse a calibration JSON and return a camera-params dict."""
    with open(json_path, encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    intrinsics = data["camera_intrinsics"]
    if data.get("warnings", {}).get("anamorphic"):
        print(f"  [{json_path.name}] WARNING: anamorphic sensor (FX != FY).")
    return {
        "FX": float(intrinsics["FX"]), "FY": float(intrinsics["FY"]),
        "CX": float(intrinsics["CX"]), "CY": float(intrinsics["CY"]),
        "W":  int(intrinsics["WIDTH"]), "H":  int(intrinsics["HEIGHT"]),
        "FPS": float(intrinsics["FPS"]),
        "K":  np.array(intrinsics["K"], dtype=np.float64),
        "D":  np.array(intrinsics["D"], dtype=np.float64),
        "source": json_path.name, "crop": data.get("crop"),
    }

def fallback_camera_params(frame_width: int, frame_height: int, fps: float) -> dict:
    """Camera-params dict from K_FALLBACK when no calibration JSON is available."""
    return {
        "FX": K_FALLBACK[0, 0], "FY": K_FALLBACK[1, 1],
        "CX": K_FALLBACK[0, 2], "CY": K_FALLBACK[1, 2],
        "W":  frame_width, "H": frame_height, "FPS": fps,
        "K":  K_FALLBACK.copy(), "D": np.zeros(5, dtype=np.float64),
        "source": "K_FALLBACK (no calibration JSON found)", "crop": None,
    }


# ── Tracker ───────────────────────────────────────────────────────────────────
class PersonTracker:
    """IoU + centroid-distance multi-person tracker with hit-count hysteresis."""

    def __init__(self,
                 iou_threshold=TRACKER_IOU_THRESHOLD,
                 max_inactive_frames=TRACKER_MAX_INACTIVE_FRAMES,
                 min_hit_streak=TRACKER_MIN_HIT_STREAK):
        self._tracks:          dict = {}
        self._next_person_id   = 0
        self._iou_threshold    = iou_threshold
        self._max_inactive     = max_inactive_frames
        self._min_hit_streak   = min_hit_streak

    @staticmethod
    def _bbox_iou(box_a, box_b) -> float:
        inter_x1 = max(box_a[0], box_b[0]);  inter_y1 = max(box_a[1], box_b[1])
        inter_x2 = min(box_a[2], box_b[2]);  inter_y2 = min(box_a[3], box_b[3])
        intersection = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
        area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
        union  = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.

    @staticmethod
    def _centroid_distance(box_a, box_b) -> float:
        return math.hypot((box_a[0]+box_a[2])/2 - (box_b[0]+box_b[2])/2,
                          (box_a[1]+box_a[3])/2 - (box_b[1]+box_b[3])/2)

    def update(self, bounding_boxes: list, confidences: list) -> list:
        """Match detections to tracks; return confirmed (person_id, confidence) pairs."""
        for pid in list(self._tracks):
            self._tracks[pid]["inactive_frames"] += 1
            if self._tracks[pid]["inactive_frames"] > self._max_inactive:
                del self._tracks[pid]

        if not bounding_boxes:
            return [(pid, self._tracks[pid]["conf"])
                    for pid in self._tracks
                    if self._tracks[pid]["hit_streak"] >= self._min_hit_streak]

        existing_ids = list(self._tracks.keys())
        matched_ids, result_ids = [], []

        for bbox, conf in zip(bounding_boxes, confidences):
            best_score, best_pid = 0., None
            for pid in existing_ids:
                if pid in matched_ids:
                    continue
                track    = self._tracks[pid]
                iou      = self._bbox_iou(bbox, track["bbox"])
                centroid_dist = self._centroid_distance(bbox, track["bbox"])
                match_score = (iou
                    + (1 - min(1., centroid_dist / TRACKER_CENTROID_DIST_NORM))
                    * TRACKER_CENTROID_DIST_WEIGHT)
                if (match_score > best_score
                        and (iou > self._iou_threshold
                             or centroid_dist < TRACKER_CENTROID_DIST_NORM)):
                    best_score, best_pid = match_score, pid

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
    """Per-person temporal smoothers: height median, pixel-span EMA, display EMA."""

    def __init__(self):
        self._height_history: dict = {}   # person_id → deque[float]
        self._pixel_height_ema: dict = {}  # person_id → float
        self._display_ema: dict = {}       # person_id → (dist, elev, azim, height)

    def push_height_sample(self, person_id: int, height_m: float,
                           head_visible: bool, ankle_visible: bool) -> None:
        """Append a height sample after outlier rejection; head and ankle must be visible."""
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
        """Median of stored samples, or BODY_HEIGHT_PRIOR_M until enough samples exist."""
        history = self._height_history.get(person_id, [])
        return (float(np.median(history))
                if len(history) >= HEIGHT_BUFFER_MIN_SAMPLES
                else BODY_HEIGHT_PRIOR_M)

    def smooth_pixel_height(self, person_id: int, pixel_height: float) -> float:
        """EMA-smooth the pixel body-height span."""
        previous = self._pixel_height_ema.get(person_id, pixel_height)
        smoothed = (EMA_ALPHA_PIXEL_HEIGHT * pixel_height
                    + (1. - EMA_ALPHA_PIXEL_HEIGHT) * previous)
        self._pixel_height_ema[person_id] = smoothed
        return smoothed

    def smooth_display_values(self, person_id: int,
                              distance_m: float, elevation_deg: float,
                              azimuth_deg: float, height_m: float) -> tuple:
        """EMA-smooth display scalars; wraps azimuth to avoid ±180° discontinuity."""
        if person_id not in self._display_ema:
            self._display_ema[person_id] = (distance_m, elevation_deg,
                                            azimuth_deg, height_m)
            return distance_m, elevation_deg, azimuth_deg, height_m
        prev_dist, prev_elev, prev_azim, prev_height = self._display_ema[person_id]
        azimuth_delta = azimuth_deg - prev_azim
        if   azimuth_delta >  180.: azimuth_deg -= 360.
        elif azimuth_delta < -180.: azimuth_deg += 360.
        alpha   = EMA_ALPHA_DISPLAY
        smoothed = (alpha * distance_m  + (1 - alpha) * prev_dist,
                    alpha * elevation_deg + (1 - alpha) * prev_elev,
                    alpha * azimuth_deg   + (1 - alpha) * prev_azim,
                    alpha * height_m      + (1 - alpha) * prev_height)
        self._display_ema[person_id] = smoothed
        return smoothed


# ── 3-D geometry ──────────────────────────────────────────────────────────────
def _pixel_body_height_span(pixel_coords, joint_scores) -> Optional[float]:
    """Pixel body-height span using priority-ordered joint groups."""
    top_pixel_y = None
    for joint_group in ([0], [1, 2], [3, 4]):
        y_values = [float(pixel_coords[j, 1])
                    for j in joint_group if joint_scores[j] > MIN_KEYPOINT_CONFIDENCE]
        if y_values:
            top_pixel_y = float(np.mean(y_values)); break
    bottom_pixel_y = None
    for joint_group in ([15, 16], [13, 14], [11, 12]):
        y_values = [float(pixel_coords[j, 1])
                    for j in joint_group if joint_scores[j] > MIN_KEYPOINT_CONFIDENCE]
        if y_values:
            bottom_pixel_y = float(np.mean(y_values)); break
    return (abs(bottom_pixel_y - top_pixel_y)
            if top_pixel_y is not None and bottom_pixel_y is not None else None)


def compute_3d_pose(pixel_coords, kp3d_normalized, joint_scores,
                    focal_x, focal_y, principal_x, principal_y,
                    height_reference_m=BODY_HEIGHT_PRIOR_M,
                    pixel_height_hint=None):
    """Convert RTMW3D normalised pose to absolute camera-3D coordinates [m].

    BODY_HEIGHT_PRIOR_M anchors the De Leva scale (units_to_meters) as a fixed
    constant.  height_reference_m (person-specific median) enters only in the
    depth estimate (z_root) to avoid circular height feedback.

    Returns (joints_meters, measured_height_m, (distance_m, elevation_deg, azimuth_deg))
    or (None, 0., None) when pose cannot be estimated.
    """
    # Vectorised De Leva scale: normalised units → metres
    scores_joint_a = joint_scores[_DELEVA_JOINT_A]
    scores_joint_b = joint_scores[_DELEVA_JOINT_B]
    segment_span_units = np.abs(kp3d_normalized[_DELEVA_JOINT_A, 2]
                                - kp3d_normalized[_DELEVA_JOINT_B, 2])
    confidence_product = scores_joint_a * scores_joint_b
    valid_segments = ((scores_joint_a > MIN_KEYPOINT_CONFIDENCE)
                      & (scores_joint_b > MIN_KEYPOINT_CONFIDENCE)
                      & (segment_span_units > 1e-3))
    if not valid_segments.any(): return None, 0., None
    weight_total = float(confidence_product[valid_segments].sum())
    if weight_total < 1e-6:     return None, 0., None
    units_to_meters = float(
        (_DELEVA_RATIO[valid_segments]
         * (BODY_HEIGHT_PRIOR_M / segment_span_units[valid_segments])
         * confidence_product[valid_segments]).sum()
        / weight_total)

    # Measured body height from head-ankle normalised Z span
    def visible(indices):
        return [j for j in indices if joint_scores[j] > MIN_KEYPOINT_CONFIDENCE]
    head_joints  = visible([0, 1, 2, 3, 4])
    ankle_joints = visible([15, 16])
    measured_height_m = 0.
    if head_joints and ankle_joints:
        measured_height_m = float(np.clip(
            abs(kp3d_normalized[head_joints, 2].mean()
                - kp3d_normalized[ankle_joints, 2].mean())
            * units_to_meters / RTMW3D_HEAD_ANKLE_Z_RATIO,
            BODY_HEIGHT_MIN_M, BODY_HEIGHT_MAX_M))

    # Camera depth from pixel body-height span  (z = fy * h_physical / h_px)
    pixel_height = (pixel_height_hint
                    if pixel_height_hint and pixel_height_hint > MIN_PIXEL_HEIGHT_SPAN
                    else _pixel_body_height_span(pixel_coords, joint_scores))
    if pixel_height is None or pixel_height < MIN_PIXEL_HEIGHT_SPAN:
        return None, 0., None
    foot_joints = visible([15, 16, 13, 14, 11, 12])
    if not head_joints or not foot_joints:
        return None, 0., None
    visible_body_ratio = (VISIBLE_BODY_RATIO_ANKLES
                          if any(j in foot_joints for j in [15, 16]) else
                          VISIBLE_BODY_RATIO_HIPS
                          if any(j in foot_joints for j in [11, 12]) else
                          VISIBLE_BODY_RATIO_KNEES)
    z_root = focal_y * height_reference_m * visible_body_ratio / pixel_height

    # Vectorised pinhole back-projection for all 17 joints
    depth_per_joint = np.clip(
        z_root + kp3d_normalized[:17, 2] * units_to_meters,
        JOINT_DEPTH_MIN_M, JOINT_DEPTH_MAX_M).astype(np.float32)
    joints_meters          = np.empty((17, 3), dtype=np.float32)
    joints_meters[:, 2]    = depth_per_joint
    joints_meters[:, 0]    = (pixel_coords[:, 0] - principal_x) * depth_per_joint / focal_x
    joints_meters[:, 1]    = (pixel_coords[:, 1] - principal_y) * depth_per_joint / focal_y

    # Centroid in spherical coordinates
    visible_mask = joint_scores[:17] > MIN_KEYPOINT_CONFIDENCE
    centroid_3d  = joints_meters[visible_mask].mean(axis=0)
    radial_dist  = math.sqrt(float(centroid_3d[0])**2
                             + float(centroid_3d[1])**2
                             + float(centroid_3d[2])**2)
    horizontal_dist = math.sqrt(float(centroid_3d[0])**2 + float(centroid_3d[2])**2)
    return (joints_meters, measured_height_m,
            (radial_dist,
             math.degrees(math.atan2(-float(centroid_3d[1]), horizontal_dist)),
             math.degrees(math.atan2( float(centroid_3d[0]), float(centroid_3d[2])))))


def proj3d2d(point_3d, focal_x, focal_y, principal_x, principal_y) -> Optional[np.ndarray]:
    """Camera-3D point → 2-D pixel coordinates; returns None when point is behind camera."""
    if point_3d[2] <= 0.: return None
    return np.array([point_3d[0] / point_3d[2] * focal_x + principal_x,
                     point_3d[1] / point_3d[2] * focal_y + principal_y],
                    dtype=np.float32)


def compute_gaze_ray_3d(joints_meters, joint_scores):
    """Gaze ray from ear midpoint to nose; returns (origin, unit_dir) or (None, None)."""
    for joint_idx in [JOINT_NOSE, JOINT_LEFT_EAR, JOINT_RIGHT_EAR]:
        if joint_scores[joint_idx] <= MIN_KEYPOINT_CONFIDENCE:
            return None, None
    ear_midpoint = (joints_meters[JOINT_LEFT_EAR] + joints_meters[JOINT_RIGHT_EAR]) * 0.5
    gaze_vec     = joints_meters[JOINT_NOSE] - ear_midpoint
    gaze_length  = float(np.linalg.norm(gaze_vec))
    if gaze_length < 1e-6:
        return None, None
    return (joints_meters[JOINT_NOSE].copy(),
            (gaze_vec / gaze_length).astype(np.float32))


def gaze_angles_3d(gaze_unit_dir) -> tuple:
    """(azimuth_deg, elevation_deg) from a unit gaze direction vector."""
    horizontal = math.hypot(float(gaze_unit_dir[0]), float(gaze_unit_dir[2]))
    return (math.degrees(math.atan2( float(gaze_unit_dir[0]), float(gaze_unit_dir[2]))),
            math.degrees(math.atan2(-float(gaze_unit_dir[1]), horizontal)))


# ── Module state ──────────────────────────────────────────────────────────────
_tracker  = PersonTracker()
_profiles = PersonProfiles()
_camera_params: Optional[dict] = None

def init_camera(frame_width: int, frame_height: int, fps: float) -> dict:
    """Load calibration JSON (or fallback K); must be called once before process_frame."""
    global _camera_params
    json_path = find_camera_json(frame_width, frame_height, CAMERA_JSON_DIR)
    _camera_params = (load_camera_params(json_path)
                      if json_path else
                      fallback_camera_params(frame_width, frame_height, fps))
    print(f"Camera  : {_camera_params['source']}")
    if _camera_params["crop"]:
        crop = _camera_params["crop"]
        print(f"  crop {crop['percent']}% "
              f"({crop['x0_px']},{crop['y0_px']})-"
              f"({crop['x1_px']},{crop['y1_px']})")
    print(f"  FX={_camera_params['FX']:.1f}  FY={_camera_params['FY']:.1f}"
          f"  CX={_camera_params['CX']:.1f}  CY={_camera_params['CY']:.1f}")
    return _camera_params

def get_camera() -> Optional[dict]:
    return _camera_params

def process_frame(bgr_frame) -> List[PersonData]:
    """Inference → tracking → 3-D pose → smoothing for one BGR frame."""
    if _camera_params is None:
        raise RuntimeError("call init_camera() before process_frame()")
    try:
        kp3d_all, scores_all, _, kp2d_all = _inference_model(bgr_frame)
    except Exception as inference_error:
        print(f"Inference error: {inference_error}"); return []

    bounding_boxes, confidences, detection_data = [], [], []
    for det_idx in range(len(kp3d_all)):
        pixel_coords   = kp2d_all[det_idx, :17, :2].astype(np.float32)
        kp3d_norm      = kp3d_all[det_idx, :17   ].astype(np.float32)
        joint_scores   = scores_all[det_idx, :17  ].astype(np.float32)
        if (joint_scores > MIN_KEYPOINT_CONFIDENCE).sum() < MIN_VISIBLE_KEYPOINTS:
            continue
        visible_kpts   = pixel_coords[joint_scores > MIN_KEYPOINT_CONFIDENCE]
        x1, y1 = visible_kpts.min(axis=0)
        x2, y2 = visible_kpts.max(axis=0)
        bounding_boxes.append([x1, y1, x2, y2])
        confidences.append(float(joint_scores[joint_scores > MIN_KEYPOINT_CONFIDENCE].mean()))
        detection_data.append((pixel_coords, kp3d_norm, joint_scores))

    results: List[PersonData] = []
    for det_idx, (person_id, _) in enumerate(_tracker.update(bounding_boxes, confidences)):
        if det_idx >= len(detection_data): break
        pixel_coords, kp3d_norm, joint_scores = detection_data[det_idx]

        raw_pixel_height     = _pixel_body_height_span(pixel_coords, joint_scores)
        smoothed_pixel_height = (_profiles.smooth_pixel_height(person_id, raw_pixel_height)
                                 if raw_pixel_height is not None else None)
        pose_result = compute_3d_pose(
            pixel_coords, kp3d_norm, joint_scores,
            _camera_params["FX"], _camera_params["FY"],
            _camera_params["CX"], _camera_params["CY"],
            _profiles.get_height_reference(person_id),
            pixel_height_hint=smoothed_pixel_height)
        if pose_result[0] is None: continue
        joints_meters, measured_height_m, (dist, elev, azim) = pose_result

        _profiles.push_height_sample(
            person_id, measured_height_m,
            head_visible=any(joint_scores[j] > MIN_KEYPOINT_CONFIDENCE for j in [0,1,2,3,4]),
            ankle_visible=any(joint_scores[j] > MIN_KEYPOINT_CONFIDENCE for j in [15, 16]))

        smoothed_dist, smoothed_elev, smoothed_azim, smoothed_height = (
            _profiles.smooth_display_values(
                person_id, dist, elev, azim,
                _profiles.get_height_reference(person_id)))
        gaze_origin, gaze_direction = compute_gaze_ray_3d(joints_meters, joint_scores)
        results.append(PersonData(
            person_id=person_id,
            pixel_coords=pixel_coords,
            joint_scores=joint_scores,
            joints_meters=joints_meters,
            distance_m=smoothed_dist,
            elevation_deg=smoothed_elev,
            azimuth_deg=smoothed_azim,
            height_m=smoothed_height,
            gaze_origin_m=gaze_origin,
            gaze_direction=gaze_direction,
        ))
    return results
