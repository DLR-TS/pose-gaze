"""basics.py – single source of truth for all constants and shared data types."""
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List
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
MIN_KEYPOINT_CONFIDENCE       = 0.25
MIN_VISIBLE_KEYPOINTS         = 6
MIN_PIXEL_HEIGHT_SPAN         = 5.0
DETECTION_SCORE_THRESHOLD     = 0.50
MIN_DETECTION_MEAN_CONFIDENCE = 0.50

# ── Tracker ───────────────────────────────────────────────────────────────────
TRACKER_IOU_THRESHOLD        = 0.25
TRACKER_MAX_INACTIVE_FRAMES  = 90
TRACKER_MIN_HIT_STREAK       = 2
TRACKER_CENTROID_DIST_NORM   = 100.0
TRACKER_CENTROID_DIST_WEIGHT = 0.3

# ── Body geometry (DO NOT CHANGE – anchors De Leva 1996) ─────────────────────
BODY_HEIGHT_PRIOR_M = 1.75

DE_LEVA_SEGMENTS: list = [
    ( 0, 15, 1.00), ( 0, 16, 1.00), ( 0, 11, 0.52), ( 0, 12, 0.52),
    ( 0,  5, 0.19), ( 0,  6, 0.19), ( 5, 11, 0.29), ( 6, 12, 0.29),
    (11, 15, 0.48), (12, 16, 0.48), (11, 13, 0.25), (12, 14, 0.25),
    (13, 15, 0.23), (14, 16, 0.23), ( 3,  5, 0.09), ( 4,  6, 0.09),
]

RTMW3D_HEAD_ANKLE_Z_RATIO = 0.96
VISIBLE_BODY_RATIO_ANKLES = 1.00
VISIBLE_BODY_RATIO_KNEES  = 0.75
VISIBLE_BODY_RATIO_HIPS   = 0.52
JOINT_DEPTH_MIN_M         = 0.1
JOINT_DEPTH_MAX_M         = 60.0
BODY_HEIGHT_MIN_M         = 0.5
BODY_HEIGHT_MAX_M         = 2.5

HEIGHT_BUFFER_WINDOW_FRAMES  = 60
HEIGHT_BUFFER_MIN_SAMPLES    = 5
HEIGHT_OUTLIER_MAX_DEVIATION = 0.30

# ── Gaze ──────────────────────────────────────────────────────────────────────
GAZE_RAY_LENGTH_M = 3.0

# ── EMA smoothing ─────────────────────────────────────────────────────────────
EMA_ALPHA_PIXEL_HEIGHT = 0.30
EMA_ALPHA_DISPLAY      = 0.35

# ── Velocity estimation ───────────────────────────────────────────────────────
VELOCITY_EMA_TAU_S    = 2.0    # time-based EMA time-constant [s], fps-invariant
VELOCITY_BUFFER_S     = 2.0    # OLS regression window [s]
VELOCITY_MIN_SPAN_S   = 0.5    # minimum data span required before computing [s]
VELOCITY_MAX_DT_S     = 3.0    # flush buffer on timestamp gap > this [s]
VELOCITY_MAX_SPEED_MS = 8.0    # hard cap [m/s] ≈ 29 km/h
VELOCITY_BUFFER_SIZE  = 60     # deque upper limit (frames)

# ── Label smoother ────────────────────────────────────────────────────────────
LABEL_SMOOTH_WINDOW_S = 2.0    # trailing-average window for overlay values [s]

# ── Display ───────────────────────────────────────────────────────────────────
DISPLAY_MAX_WIDTH_PX  = 1580
DISPLAY_MAX_HEIGHT_PX = 840
SEEK_STEP_SECONDS     = 5

# ── Joint indices ─────────────────────────────────────────────────────────────
JOINT_NOSE           = 0
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
SKELETON_EDGES: list = [
    (16,14),(14,12),(17,15),(15,13),(12,13),(6,12),(7,13),(6,7),
    (6,8),(7,9),(8,10),(9,11),(2,3),(1,2),(1,3),(2,4),(3,5),(4,6),(5,7),
]
SKELETON_EDGE_COLORS: list = [
    (255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),
    (0,255,0),(0,255,85),(0,255,170),(0,255,255),(0,170,255),(0,85,255),
    (0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85),
]

# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_KEYPOINT_NOSE     = (0,   0,   255)
COLOR_KEYPOINT_EAR      = (255, 0,   0)
COLOR_KEYPOINT_OTHER    = (0,   255, 0)
COLOR_EAR_CONNECTOR     = (150, 150, 150)
COLOR_GAZE_ORIGIN_DOT   = (0,   255, 0)
COLOR_GAZE_LINE         = (0,   255, 255)
COLOR_GAZE_ENDPOINT     = (0,   0,   255)

COLOR_LABEL_DIST        = (255, 255, 0)
COLOR_LABEL_ANGLES      = (255, 0,   255)
COLOR_LABEL_GAZE        = (0,   255, 255)

COLOR_HUD_RECORDING     = (0,   0,   255)
COLOR_HUD_PAUSED        = (0,   220, 255)
COLOR_HUD_RUNNING       = (0,   255, 0)
COLOR_HUD_CAM_INFO      = (200, 200, 200)
COLOR_HUD_HINT          = (255, 255, 255)

COLOR_BEARING_RAY       = (255, 200, 0)
COLOR_GROUND_LONG_LINES = (80,  80,  80)
COLOR_GROUND_PLANE_FILL = (30,  80,  30)

# ── Drawing sizes ─────────────────────────────────────────────────────────────
KEYPOINT_RADIUS         = 2
GAZE_ORIGIN_DOT_RADIUS  = 2
GAZE_ENDPOINT_RADIUS    = 3
SKELETON_LINE_THICKNESS = 1
GAZE_LINE_THICKNESS     = 2
TEXT_OUTLINE_THICKNESS  = 2
BEARING_RAY_THICKNESS   = 1
BEARING_ENDPOINT_RADIUS = 4
BEARING_ENDPOINT_COLOR  = (255, 200, 0)

# ── Label layout ──────────────────────────────────────────────────────────────
LABEL_LINE_SPACING_PX  = 26
LABEL_FONT_SCALE_SMALL = 0.55
HUD_FONT_SCALE_STATUS  = 0.7
HUD_FONT_SCALE_CAM     = 0.45

# ── Ground plane ──────────────────────────────────────────────────────────────
GROUND_PLANE_ENABLED           = True
GROUND_PLANE_MAX_DEPTH_M       = 20.0
GROUND_PLANE_SAMPLE_INTERVAL_S = 0.5
GROUND_PLANE_SAMPLE_BUFFER     = 200
GROUND_PLANE_MIN_SAMPLES       = 20
GROUND_PLANE_LINE_THICKNESS    = 1
GROUND_PLANE_FILL_ALPHA        = 0.25
GROUND_PLANE_X_STEP_M          = 1.0
GROUND_PLANE_X_HALF_M          = 4.0
GROUND_PLANE_NEAR_FRAC         = 0.75
GROUND_PLANE_FAR_FRAC          = 0.25

# ── Shared data types ─────────────────────────────────────────────────────────
@dataclass
class GroundPlane:
    centroid:  np.ndarray
    normal:    np.ndarray
    n_samples: int = 0

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
    covariance: List[float] = field(default_factory=list)

@dataclass
class ObjectVelocity:
    """vx=forward, vy=lateral(right+), vz=up [m/s]."""
    vx: float = 0.0; vy: float = 0.0; vz: float = 0.0
    rx: float = 0.0; ry: float = 0.0; rz: float = 0.0
    covariance: List[float] = field(default_factory=list)

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