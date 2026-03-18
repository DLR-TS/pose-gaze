"""basics.py – single source of truth for all constants and shared data types."""
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
import numpy as np
import json as _json

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = Path(__file__).parent.resolve()
MODEL_DIR       = SCRIPT_DIR / "models"
OUTPUT_DIR      = SCRIPT_DIR / "recordings"
CAMERA_JSON_DIR: Path = SCRIPT_DIR / "media"

VIDEO_PATH   = CAMERA_JSON_DIR / "video_1808x1392_mvBlueCOUGAR-X109b_crop205-391-2013-1783.mp4"
FPS_FALLBACK = 13.2

K_FALLBACK = np.array(
    [[4637.7, 0., 2056.], [0., 4637.7, 1088.], [0., 0., 1.]], dtype=np.float64)

# ── Keypoint / detection thresholds ──────────────────────────────────────────
MIN_KEYPOINT_CONFIDENCE       = 0.25   # per-joint score threshold for visibility
MIN_VISIBLE_KEYPOINTS         = 6      # detection rejected below this
MIN_PIXEL_HEIGHT_SPAN         = 5.0    # px; discard implausibly small body spans
DETECTION_SCORE_THRESHOLD     = 0.50
MIN_DETECTION_MEAN_CONFIDENCE = 0.50   # mean score over visible joints

# ── Tracker ───────────────────────────────────────────────────────────────────
TRACKER_IOU_THRESHOLD        = 0.25   # min IoU to consider bboxes overlapping
TRACKER_MAX_INACTIVE_FRAMES  = 20   # ~1.5 s @ 13 fps; increase for slower cameras
TRACKER_MIN_HIT_STREAK       = 2    # suppresses 1-frame phantom detections
TRACKER_CENTROID_DIST_NORM   = 100.0  # px normalization for centroid cost term
TRACKER_CENTROID_DIST_WEIGHT = 0.3   # blend: 0=IoU only, 1=centroid only
TRACKER_MATCH_GATE           = 0.99  # assignment rejected when cost >= this

# ── Body geometry (De Leva 1996 segment ratios) ───────────────────────────────
BODY_HEIGHT_PRIOR_M = 1.75  # assumed standing adult height [m]

DE_LEVA_SEGMENTS: list = [
    ( 0, 15, 1.00), ( 0, 16, 1.00), ( 0, 11, 0.52), ( 0, 12, 0.52),
    ( 0,  5, 0.19), ( 0,  6, 0.19), ( 5, 11, 0.29), ( 6, 12, 0.29),
    (11, 15, 0.48), (12, 16, 0.48), (11, 13, 0.25), (12, 14, 0.25),
    (13, 15, 0.23), (14, 16, 0.23), ( 3,  5, 0.09), ( 4,  6, 0.09),
]

RTMW3D_HEAD_ANKLE_Z_RATIO    = 0.96  # fraction of body height spanned by the model's z-axis between head and ankle joints; empirically calibrated for RTMPose-Wholebody3D
VISIBLE_BODY_RATIO_ANKLES    = 1.00  # full body in frame; pixel span equals full body height
VISIBLE_BODY_RATIO_KNEES     = 0.75  # body cropped below knees; pixel span covers ~75 % of height
VISIBLE_BODY_RATIO_HIPS      = 0.52  # body cropped at hips; pixel span covers ~52 % of height
JOINT_DEPTH_MIN_M            = 0.1   # clamp; avoids division artifacts near camera
JOINT_DEPTH_MAX_M            = 60.0  # clamp; discard implausibly far joints
BODY_HEIGHT_MIN_M            = 0.5
BODY_HEIGHT_MAX_M            = 2.5
HEIGHT_BUFFER_WINDOW_FRAMES  = 60    # rolling window for height median
HEIGHT_BUFFER_MIN_SAMPLES    = 5     # samples before median replaces prior
HEIGHT_OUTLIER_MAX_DEVIATION = 0.30  # relative deviation to reject a height sample
HEIGHT_PUBLISH_INTERVAL_S    = 5.0   # seconds between height reference updates in PersonData

# ── Gaze ──────────────────────────────────────────────────────────────────────
GAZE_RAY_LENGTH_M = 3.0

# ── EMA time constants [s] ────────────────────────────────────────────────────
TAU_PIXEL_HEIGHT_S = 0.3   # EMA time constant for pixel-height smoothing
TAU_Z_ROOT_S       = 0.15  # EMA for depth anchor; damps jumps on visibility change
TAU_DISPLAY_S      = 1.0   # EMA for displayed distance/angles (visual stability)
TAU_CAM_HEIGHT_S   = 5.0   # EMA for camera height; 5s suits static camera

# ── Velocity estimation ───────────────────────────────────────────────────────
VELOCITY_EMA_TAU_S    = 2.0  # smoothing of velocity estimate
VELOCITY_BUFFER_S     = 2.0  # time window for linear regression
VELOCITY_MIN_SPAN_S   = 0.5  # min time span before velocity is reported
VELOCITY_MAX_DT_S     = 3.0  # gap larger than this resets the position buffer
VELOCITY_MAX_SPEED_MS = 8.0  # clamp [m/s]; ~29 km/h
VELOCITY_BUFFER_SIZE  = 60   # max position samples retained

# ── Display ───────────────────────────────────────────────────────────────────
DISPLAY_DS_REFERENCE_PX: Tuple[int, int] = (1808, 1392)  # native sensor resolution; line widths and font sizes are tuned for this resolution and scaled proportionally for other resolutions
DISPLAY_MAX_WIDTH_PX  = 1580
DISPLAY_MAX_HEIGHT_PX = 840
SEEK_STEP_SECONDS     = 5

# ── Runtime device ───────────────────────────────────────────────────────────
try:
    import torch as _torch
    DEVICE: str = "cuda" if _torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE: str = "cpu"

# ── Joint indices ─────────────────────────────────────────────────────────────
JOINT_NOSE           = 0
JOINT_LEFT_EYE       = 1
JOINT_RIGHT_EYE      = 2
JOINT_LEFT_EAR       = 3
JOINT_RIGHT_EAR      = 4
JOINT_LEFT_SHOULDER  = 5
JOINT_RIGHT_SHOULDER = 6
JOINT_LEFT_HIP       = 11
JOINT_RIGHT_HIP      = 12
JOINT_LEFT_KNEE      = 13
JOINT_RIGHT_KNEE     = 14
JOINT_LEFT_ANKLE     = 15
JOINT_RIGHT_ANKLE    = 16

# ── Skeleton ──────────────────────────────────────────────────────────────────
# SKELETON_EDGES follows the original COCO convention (1-based joint numbering).
# All drawing code converts to 0-based with -1 at use.
SKELETON_EDGES: list = [
    (16,14),(14,12),(17,15),(15,13),(12,13),(6,12),(7,13),(6,7),
    (6,8),(7,9),(8,10),(9,11),(2,3),(1,2),(1,3),(2,4),(3,5),(4,6),(5,7),
]
# One color per skeleton edge; length must equal len(SKELETON_EDGES).
SKELETON_EDGE_COLORS: list = [
    (255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),
    (0,255,0),(0,255,85),(0,255,170),(0,255,255),(0,170,255),(0,85,255),
(0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85),
(255,128,0),
]
assert len(SKELETON_EDGES) == len(SKELETON_EDGE_COLORS), (
    f"SKELETON_EDGES ({len(SKELETON_EDGES)}) != "
    f"SKELETON_EDGE_COLORS ({len(SKELETON_EDGE_COLORS)})"
)

# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_KEYPOINT_NOSE     = (0,   0,   255)
COLOR_KEYPOINT_EAR      = (255, 0,   0  )
COLOR_KEYPOINT_OTHER    = (0,   255, 0  )
COLOR_EAR_CONNECTOR     = (150, 150, 150)
COLOR_GAZE_ORIGIN_DOT   = (0,   255, 0  )
COLOR_GAZE_LINE         = (0,   255, 255)
COLOR_GAZE_ENDPOINT     = (0,   0,   255)
COLOR_LABEL_DIST        = (255, 255, 0  )
COLOR_LABEL_ANGLES      = (255, 0,   255)
COLOR_LABEL_GAZE        = (0,   255, 255)
COLOR_HUD_RECORDING     = (0,   0,   255)
COLOR_HUD_PAUSED        = (0,   220, 255)
COLOR_HUD_RUNNING       = (0,   255, 0  )
COLOR_HUD_CAM_INFO      = (200, 200, 200)
COLOR_HUD_HINT          = (255, 255, 255)
COLOR_BEARING_RAY       = (255, 200, 0  )
COLOR_GROUND_LONG_LINES = (80,  80,  80 )
COLOR_GROUND_PLANE_FILL = (30,  80,  30 )

# ── Drawing sizes ─────────────────────────────────────────────────────────────
KEYPOINT_RADIUS          = 2
GAZE_ORIGIN_DOT_RADIUS   = 2
GAZE_ENDPOINT_RADIUS     = 3
SKELETON_LINE_THICKNESS  = 1
GAZE_LINE_THICKNESS      = 2
TEXT_OUTLINE_THICKNESS   = 2
BEARING_RAY_THICKNESS    = 1
BEARING_ENDPOINT_RADIUS  = 4
BEARING_ENDPOINT_COLOR   = (255, 200, 0)

# ── Label layout ──────────────────────────────────────────────────────────────
LABEL_LINE_SPACING_PX  = 26
LABEL_FONT_SCALE_SMALL = 0.55
HUD_FONT_SCALE_STATUS  = 0.7
HUD_FONT_SCALE_CAM     = 0.45

# ── Ground plane ──────────────────────────────────────────────────────────────
GROUND_PLANE_ENABLED              = True
GROUND_PLANE_MAX_DEPTH_M          = 20.0  # far clamp for plane projection
GROUND_PLANE_SAMPLE_INTERVAL_S    = 0.2   # seconds between ground point collections
GROUND_PLANE_FIT_INTERVAL_S       = 5.0   # seconds between plane refits; 0 = every sample
GROUND_PLANE_SAMPLE_BUFFER        = 200   # max 3-D points retained for RANSAC
GROUND_PLANE_MIN_SAMPLES          = 5     # minimum points before RANSAC runs
GROUND_PLANE_LINE_THICKNESS       = 1
GROUND_PLANE_FILL_ALPHA           = 0.25  # overlay transparency (0=invisible, 1=opaque)
GROUND_PLANE_X_STEP_M             = 1.0   # grid line spacing [m]
GROUND_PLANE_X_HALF_M             = 4.0   # grid half-width left/right [m]
GROUND_PLANE_NEAR_FRAC            = 0.75  # image-height fraction for near edge
GROUND_PLANE_FAR_FRAC             = 0.25  # image-height fraction for far edge
GROUND_PLANE_RANSAC_ITERATIONS    = 50    # RANSAC trials per fit
GROUND_PLANE_NORMAL_MIN_Y         = 0.5   # min |normal[1]|; rejects near-vertical planes
CAM_HEIGHT_MIN_M                  = 0.5   # plausible camera height range [m]
CAM_HEIGHT_MAX_M                  = 12.0  # plausible camera height range [m]
GROUND_PLANE_RANSAC_INLIER_DIST   = 0.30  # inlier threshold [m]; accounts for pose depth noise
GROUND_PLANE_TEMPORAL_DECAY_S     = 10.0  # age weight decay for SVD refit

# ── Shared data types ─────────────────────────────────────────────────────────
@dataclass
class GroundPlane:
    centroid:     np.ndarray
    normal:       np.ndarray
    n_samples:    int   = 0
    inlier_ratio: float = 0.0  # fraction of accumulated floor samples that lie within RANSAC inlier distance
    fit_rmse:     float = 0.0  # temporally-weighted RMS distance of inlier points from the fitted plane [m]

class CamHeightEMA:
    """Exponential moving average of camera height derived from ground plane.

    Separates EMA state from draw_ground_plane() so the function has no
    side-effects and is easily testable.
    """
    def __init__(self) -> None:
        self._ema:    Optional[float] = None
        self._last_t: Optional[float] = None

    def update(self, h_raw: float, timestamp_s: float) -> Optional[float]:
        if h_raw < CAM_HEIGHT_MIN_M or h_raw > CAM_HEIGHT_MAX_M:
            return self._ema
        if self._ema is None or self._last_t is None:
            self._ema    = h_raw
            self._last_t = timestamp_s
            return self._ema
        dt           = max(0.0, timestamp_s - self._last_t)
        alpha        = 1.0 - math.exp(-dt / TAU_CAM_HEIGHT_S) if dt > 1e-4 else 0.0
        self._ema    = alpha * h_raw + (1.0 - alpha) * self._ema
        self._last_t = timestamp_s
        return self._ema

    @property
    def value(self) -> Optional[float]:
        return self._ema


@dataclass
class MessageHeader:
    sec: int; nanosec: int; frame_id: str

@dataclass
class ObjectSize:
    l: float; w: float; h: float

@dataclass
class ObjectPose:
    x: float; y: float; yaw: float
    z: float = 0.0; roll: float = 0.0; pitch: float = 0.0

@dataclass
class ObjectVelocity:
    """vx=forward, vy=lateral(right+), vz=up [m/s]."""
    vx: float = 0.0; vy: float = 0.0; vz: float = 0.0
    rx: float = 0.0; ry: float = 0.0; rz: float = 0.0

@dataclass
class ObjectLabel:
    text: str = ""; value: float = 0.0

@dataclass
class DetectedObject:
    id: int; object_class: str; score: float
    pose: ObjectPose; size: ObjectSize
    velocity: Optional[ObjectVelocity] = None
    label:    Optional[ObjectLabel]    = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["class"] = d.pop("object_class")
        return {k: v for k, v in d.items() if v is not None}

@dataclass
class DetectionMessage:
    header:  MessageHeader
    objects: List[DetectedObject]

    def to_dict(self):
        return {"header":  asdict(self.header),
                "objects": [o.to_dict() for o in self.objects]}

    def to_json(self):
        return _json.dumps(self.to_dict(), separators=(",", ":"))


# ── Per-frame result ─────────────────────────────────────────────────────────
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


# ── Projection helpers ───────────────────────────────────────────────────────
def proj3d2d(point_3d, focal_x, focal_y,
             principal_x, principal_y) -> Optional[np.ndarray]:
    if point_3d[2] <= 0.:
        return None
    return np.array([point_3d[0] / point_3d[2] * focal_x + principal_x,
                     point_3d[1] / point_3d[2] * focal_y + principal_y],
                    dtype=np.float32)


def gaze_angles_3d(gaze_unit_dir) -> tuple:
    """Return (azimuth_deg, elevation_deg) for a unit gaze direction vector."""
    h = math.hypot(float(gaze_unit_dir[0]), float(gaze_unit_dir[2]))
    return (math.degrees(math.atan2( float(gaze_unit_dir[0]), float(gaze_unit_dir[2]))),
            math.degrees(math.atan2(-float(gaze_unit_dir[1]), h)))

# ── Detection message ────────────────────────────────────────────────────────
_BODY_DEPTH_M = 0.25  # approximate frontal depth of a standing adult [m]; used for bounding box estimation
_SHOULDER_W_M = 0.45  # approximate shoulder width of a standing adult [m]; used for bounding box estimation


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
                        z=r * math.sin(er), yaw=yaw),
        size=ObjectSize(l=_BODY_DEPTH_M, w=_SHOULDER_W_M, h=p.height_m),
        velocity=p.velocity_m_per_s,
        label=ObjectLabel(text=f"H:{p.height_m*100:.0f}cm spd:{spd*3.6:.1f}km/h",
                          value=round(p.height_m, 3)),
    )


def build_detection_message(persons: List[PersonData], timestamp_s: float,
                             frame_id: str = "CAMERA") -> DetectionMessage:
    sec  = int(timestamp_s)
    nsec = int((timestamp_s - sec) * 1_000_000_000)
    return DetectionMessage(
        header=MessageHeader(sec=sec, nanosec=nsec, frame_id=frame_id),
        objects=[_person_to_obj(p) for p in persons],
    )
