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
MIN_KEYPOINT_CONFIDENCE       = 0.25   # joints below this score are treated as not visible
MIN_VISIBLE_KEYPOINTS         = 6      # detections with fewer confident joints than this are discarded
MIN_PIXEL_HEIGHT_SPAN         = 5.0    # pixel height below which a body detection is too small to estimate depth [px]
DETECTION_SCORE_THRESHOLD     = 0.50
MIN_DETECTION_MEAN_CONFIDENCE = 0.50   # detections whose mean visible-joint score falls below this are discarded

# ── Tracker ───────────────────────────────────────────────────────────────────
TRACKER_IOU_THRESHOLD        = 0.25   # min IoU to consider bboxes overlapping
TRACKER_MAX_INACTIVE_FRAMES  = 20   # frames without a matching detection before a track is dropped; ~1.5 s at 13 fps
TRACKER_MIN_HIT_STREAK       = 2    # consecutive matched frames required before a new track is reported
TRACKER_CENTROID_DIST_NORM   = 100.0  # pixel distance at which the centroid cost term equals 1.0; scales centroid vs IoU contribution
TRACKER_CENTROID_DIST_WEIGHT = 0.3   # weight of centroid distance in the assignment cost; 0=IoU only, 1=centroid only
TRACKER_MATCH_GATE           = 0.99  # Hungarian assignments with cost above this threshold are rejected as implausible

# ── Body geometry (De Leva 1996 segment ratios) ───────────────────────────────
BODY_HEIGHT_PRIOR_M = 1.75  # default assumed standing adult height [m]; used until a personal estimate is available

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
JOINT_DEPTH_MIN_M            = 0.1   # minimum allowed joint depth [m]; prevents division artifacts for joints very close to the camera
JOINT_DEPTH_MAX_M            = 60.0  # maximum allowed joint depth [m]; joints beyond this distance are clamped
BODY_HEIGHT_MIN_M            = 0.5
BODY_HEIGHT_MAX_M            = 2.5
HEIGHT_BUFFER_WINDOW_FRAMES  = 60    # number of height samples retained per person for median estimation
HEIGHT_BUFFER_MIN_SAMPLES    = 5     # minimum samples required before the personal height median replaces the prior
HEIGHT_OUTLIER_MAX_DEVIATION = 0.30  # relative deviation to reject a height sample
HEIGHT_PUBLISH_INTERVAL_S    = 5.0   # interval between height updates published in PersonData; the internal depth anchor updates every frame

# ── Gaze ──────────────────────────────────────────────────────────────────────
GAZE_RAY_LENGTH_M = 3.0

# ── EMA time constants [s] ────────────────────────────────────────────────────
TAU_PIXEL_HEIGHT_S = 0.3   # EMA time constant for pixel body-height smoothing [s]; reduces jitter from partial occlusions
TAU_DISPLAY_S      = 1.0   # EMA time constant for displayed distance and bearing angles [s]; higher values reduce label jitter
TAU_CAM_HEIGHT_S   = 5.0   # EMA time constant for camera height estimate [s]; a longer value suits a static camera

# ── Velocity estimation ───────────────────────────────────────────────────────
VELOCITY_EMA_TAU_S    = 2.0  # EMA time constant applied to the OLS velocity estimate [s]
VELOCITY_BUFFER_S     = 2.0  # time window over which position samples are collected for OLS regression [s]
VELOCITY_MIN_SPAN_S   = 0.5  # minimum time span of samples in the regression window before velocity is reported [s]
VELOCITY_MAX_DT_S     = 3.0  # if no detection is received for this long, the position history is cleared [s]
VELOCITY_MAX_SPEED_MS = 8.0  # raw velocity estimates above this value are clamped [m/s]; corresponds to ~29 km/h
VELOCITY_BUFFER_SIZE  = 60   # maximum number of position samples retained in the rolling buffer per person

# ── Display ───────────────────────────────────────────────────────────────────
DISPLAY_DS_REFERENCE_PX: Tuple[int, int] = (1808, 1392)  # reference resolution for which all line widths and font sizes are calibrated; other resolutions are scaled proportionally
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
# Skeleton edges use 1-based COCO joint indices.
# The drawing loop converts to 0-based with index - 1.
SKELETON_EDGES: list = [
    (16,14),(14,12),(17,15),(15,13),(12,13),(6,12),(7,13),(6,7),
    (6,8),(7,9),(8,10),(9,11),(2,3),(1,2),(1,3),(2,4),(3,5),(4,6),(5,7),
]
# One RGB color per skeleton edge; order matches SKELETON_EDGES exactly.
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
GROUND_PLANE_MAX_DEPTH_M          = 20.0  # maximum depth used when projecting the ground plane overlay [m]
GROUND_PLANE_SAMPLE_INTERVAL_S    = 0.2   # interval between floor sample collections from person hip positions [s]
GROUND_PLANE_FIT_INTERVAL_S       = 5.0   # interval between RANSAC plane refits [s]; set to 0 to refit on every sample
GROUND_PLANE_SAMPLE_BUFFER        = 200   # maximum number of 3-D floor points retained for RANSAC fitting
GROUND_PLANE_MIN_SAMPLES          = 5     # minimum number of floor samples required before RANSAC will attempt a fit
GROUND_PLANE_LINE_THICKNESS       = 1
GROUND_PLANE_FILL_ALPHA           = 0.25  # transparency of the ground plane overlay polygon; 0=invisible, 1=fully opaque
GROUND_PLANE_X_STEP_M             = 1.0   # lateral spacing between ground plane grid lines [m]
GROUND_PLANE_X_HALF_M             = 4.0   # half-width of the ground plane grid to the left and right of centre [m]
GROUND_PLANE_NEAR_FRAC            = 0.75  # image-height fraction at which the near edge of the ground plane overlay is drawn
GROUND_PLANE_FAR_FRAC             = 0.25  # image-height fraction at which the far edge of the ground plane overlay is drawn
GROUND_PLANE_RANSAC_ITERATIONS    = 50    # number of random 3-point samples drawn per RANSAC plane fit
GROUND_PLANE_NORMAL_MIN_Y         = 0.5   # minimum absolute Y-component of the fitted plane normal; rejects near-vertical fits that indicate a bad sample set
CAM_HEIGHT_MIN_M                  = 0.5   # lower bound for accepted camera height estimates [m]; measurements below this are discarded
CAM_HEIGHT_MAX_M                  = 12.0  # upper bound for accepted camera height estimates [m]; measurements above this are discarded
GROUND_PLANE_RANSAC_INLIER_DIST   = 0.30  # distance threshold for classifying a point as a RANSAC inlier [m]; should be larger than typical pose depth noise
GROUND_PLANE_TEMPORAL_DECAY_S     = 10.0  # time constant for exponential age-weighting of inlier samples during the SVD refit [s]; older samples contribute less

# Grid x-positions for the ground plane overlay, computed once from the constants above.
# np.linspace guarantees both endpoints are included with exact floating-point values.
_n_grid = round(2 * GROUND_PLANE_X_HALF_M / GROUND_PLANE_X_STEP_M) + 1
GROUND_PLANE_GRID_X = np.linspace(
    -GROUND_PLANE_X_HALF_M, GROUND_PLANE_X_HALF_M, _n_grid, dtype=np.float32)
del _n_grid

# ── Shared data types ─────────────────────────────────────────────────────────
@dataclass
class GroundPlane:
    """Fitted ground plane in camera space.

    normal points upward (negative Y in camera convention where Y+ is down).
    centroid is the weighted mean of inlier floor samples used for the fit.
    inlier_ratio and fit_rmse describe fit quality and can gate downstream use.
    """
    centroid:     np.ndarray
    normal:       np.ndarray
    inlier_ratio: float = 0.0  # fraction of accumulated floor samples within RANSAC inlier distance
    fit_rmse:     float = 0.0  # temporally-weighted RMS distance of inlier points from the fitted plane [m]


# ── EMA helper ────────────────────────────────────────────────────────────────
class _Ema:
    """Time-continuous exponential moving average keyed by integer ID.

    Uses alpha = 1 - exp(-dt / tau) so the smoothing rate is independent of
    frame rate. Supports scalar and numpy-array values transparently.
    Call reset(pid) to clear state when a tracked object disappears.
    """

    def __init__(self, tau: float):
        self._tau = tau
        self._v:  dict = {}
        self._t:  dict = {}

    def __call__(self, pid: int, val, timestamp_s: float):
        prev, last_t = self._v.get(pid), self._t.get(pid)
        if prev is None:
            self._v[pid] = val
            self._t[pid] = timestamp_s
            return val
        dt  = max(0.0, timestamp_s - last_t)
        a   = 1.0 - math.exp(-dt / self._tau) if dt > 1e-4 else 0.0
        out = a * val + (1.0 - a) * prev
        self._v[pid] = out
        self._t[pid] = timestamp_s
        return out

    def reset(self, pid: int) -> None:
        self._v.pop(pid, None)
        self._t.pop(pid, None)

class CamHeightEMA:
    """EMA-smoothed camera height estimate, derived from ground plane geometry.

    Measurements outside [CAM_HEIGHT_MIN_M, CAM_HEIGHT_MAX_M] are discarded
    so that poorly-fitted early planes do not corrupt the estimate.
    TAU_CAM_HEIGHT_S controls how quickly the estimate follows new measurements.
    """
    def __init__(self) -> None:
        self._ema = _Ema(TAU_CAM_HEIGHT_S)

    def update(self, h_raw: float, timestamp_s: float) -> Optional[float]:
        if h_raw < CAM_HEIGHT_MIN_M or h_raw > CAM_HEIGHT_MAX_M:
            return self.value
        self._ema(0, h_raw, timestamp_s)
        return self.value

    @property
    def value(self) -> Optional[float]:
        return self._ema._v.get(0)
@dataclass
class ObjectVelocity:
    """Velocity vector in world-aligned camera space.

    vx = forward [m/s], vy = lateral right [m/s], vz = up [m/s].
    Rotational rates rx/ry/rz are reserved and currently always zero.
    """
    vx: float = 0.0; vy: float = 0.0; vz: float = 0.0
    rx: float = 0.0; ry: float = 0.0; rz: float = 0.0

# ── Per-frame detection result ────────────────────────────────────────────────
@dataclass
class PersonData:
    """All data associated with one detected person for a single frame.

    pixel_coords and joints_meters are (17, 2) and (17, 3) arrays indexed by
    COCO joint index. joint_visible is a boolean mask of the same length.
    Spherical coordinates (distance_m, elevation_deg, azimuth_deg) are relative
    to the camera origin; azimuth is 0 straight ahead, positive to the right.
    """
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
    joint_visible:    Optional[np.ndarray]      = None
    velocity_m_per_s: Optional[ObjectVelocity] = None

# ── Projection helpers ───────────────────────────────────────────────────────
def proj3d2d(point_3d, focal_x, focal_y,
             principal_x, principal_y) -> Optional[np.ndarray]:
    """Project a single 3-D camera-space point to 2-D pixel coordinates.

    Returns None when the point is behind the camera (z <= 0).
    """
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


# ── NDJSON export ────────────────────────────────────────────────────────────
_BODY_DEPTH_M = 0.25  # approximate frontal depth of a standing adult [m]; used for bounding box estimation
_SHOULDER_W_M = 0.45  # approximate shoulder width of a standing adult [m]; used for bounding box estimation


def build_ndjson_line(persons: List[PersonData], timestamp_s: float,
                      frame_id: str = "camera") -> str:
    """Serialise a list of PersonData snapshots as a single NDJSON line.

    Each person is encoded as a flat dict with pose, size, velocity, and a
    human-readable label. The result is ready to write directly to an .ndjson file.
    """
    sec  = int(timestamp_s)
    nsec = int((timestamp_s - sec) * 1e9)
    objects = []
    for p in persons:
        er  = math.radians(p.elevation_deg)
        ar  = math.radians(p.azimuth_deg)
        hd  = p.distance_m * math.cos(er)
        yaw = math.atan2(math.sin(ar), math.cos(ar))
        vx = vy = spd = 0.0
        if p.velocity_m_per_s is not None:
            vx  = p.velocity_m_per_s.vx
            vy  = p.velocity_m_per_s.vy
            spd = math.sqrt(vx**2 + vy**2)
        obj = {
            "id":    p.person_id,
            "class": "person",
            "score": round(float(p.joint_scores[p.joint_visible].mean())
                           if p.joint_visible is not None and p.joint_visible.any()
                           else 0.0, 3),
            "pose":  {"x": round(hd * math.cos(ar), 3),
                      "y": round(hd * math.sin(ar), 3),
                      "z": round(p.distance_m * math.sin(er), 3),
                      "yaw": round(yaw, 4)},
            "size":  {"l": _BODY_DEPTH_M, "w": _SHOULDER_W_M,
                      "h": round(p.height_m, 3)},
            "velocity": {"vx": round(vx, 3), "vy": round(vy, 3)},
            "label": f"H:{p.height_m*100:.0f}cm spd:{spd*3.6:.1f}km/h",
        }
        objects.append(obj)
    msg = {"header": {"sec": sec, "nanosec": nsec, "frame_id": frame_id},
           "objects": objects}
    return _json.dumps(msg, separators=(",", ":"))
