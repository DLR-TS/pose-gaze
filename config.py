"""config.py – single source of truth for all constants. Edit only this file."""
from pathlib import Path
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_DIR  = SCRIPT_DIR / "models"
OUTPUT_DIR = SCRIPT_DIR / "recordings"

# ── Video ─────────────────────────────────────────────────────────────────────
VIDEO_PATH   = "media/video_1808x1392_mvBlueCOUGAR-X109b_crop205-391-2013-1783.mp4"
FPS_FALLBACK = 13.2     # used when the video container reports 0

# ── Camera calibration ────────────────────────────────────────────────────────
CAMERA_JSON_DIR: Path = SCRIPT_DIR / "media"
K_FALLBACK = np.array([[4637.7, 0., 2056.], [0., 4637.7, 1088.], [0., 0., 1.]],
                      dtype=np.float64)

# ── Detection / keypoint ──────────────────────────────────────────────────────
MIN_KEYPOINT_CONFIDENCE = 0.25   # joints below this score are treated as invisible
MIN_VISIBLE_KEYPOINTS   = 6      # minimum visible joints to accept a person detection
MIN_PIXEL_HEIGHT_SPAN   = 5.0    # minimum pixel body-height span to attempt 3-D pose

# ── Tracker ───────────────────────────────────────────────────────────────────
TRACKER_IOU_THRESHOLD       = 0.25    # minimum IoU for a bbox match
TRACKER_MAX_INACTIVE_FRAMES = 90      # frames before an inactive track is dropped
TRACKER_MIN_HIT_STREAK      = 2       # consecutive detections to confirm a new track
TRACKER_CENTROID_DIST_NORM  = 100.0   # centroid-distance normalisation factor [px]
TRACKER_CENTROID_DIST_WEIGHT = 0.3    # centroid-distance contribution to match score

# ── Anthropometrics / depth ───────────────────────────────────────────────────
# DO NOT CHANGE – used as fixed prior for the De Leva scale (u2m).
# Changing this shifts all depth estimates proportionally.
BODY_HEIGHT_PRIOR_M = 1.75    # [m] fixed prior; must NOT be updated per person

# DO NOT CHANGE – De Leva (1996), Table 1: segment ratios relative to standing height.
# Pairs: (COCO-17 joint_a, COCO-17 joint_b, fraction_of_standing_height)
DE_LEVA_SEGMENTS: list = [
    ( 0, 15, 1.00), ( 0, 16, 1.00),   # nose → ankle        (~full height)
    ( 0, 11, 0.52), ( 0, 12, 0.52),   # nose → hip          (De Leva 0.52)
    ( 0,  5, 0.19), ( 0,  6, 0.19),   # nose → shoulder     (De Leva 0.19)
    ( 5, 11, 0.29), ( 6, 12, 0.29),   # shoulder → hip      (De Leva 0.29)
    (11, 15, 0.48), (12, 16, 0.48),   # hip → ankle         (De Leva 0.48)
    (11, 13, 0.25), (12, 14, 0.25),   # hip → knee          (De Leva 0.25)
    (13, 15, 0.23), (14, 16, 0.23),   # knee → ankle        (De Leva 0.23)
    ( 3,  5, 0.09), ( 4,  6, 0.09),   # ear → shoulder      (De Leva 0.09)
]

# DO NOT CHANGE – empirical RTMW3D calibration: head-to-ankle Z span vs. true height.
RTMW3D_HEAD_ANKLE_Z_RATIO = 0.96

# DO NOT CHANGE – De Leva fractions of total height visible from head downward,
# used to infer camera depth when the full body is not in frame.
VISIBLE_BODY_RATIO_ANKLES = 1.00   # ankles visible
VISIBLE_BODY_RATIO_KNEES  = 0.75   # knees visible but not ankles
VISIBLE_BODY_RATIO_HIPS   = 0.52   # only hips visible (matches De Leva nose→hip)

# DO NOT CHANGE – plausibility bounds for joint depth and body height.
JOINT_DEPTH_MIN_M    = 0.1    # minimum plausible joint depth [m]
JOINT_DEPTH_MAX_M    = 60.0   # maximum plausible joint depth [m]
BODY_HEIGHT_MIN_M    = 0.5    # minimum plausible person height [m]
BODY_HEIGHT_MAX_M    = 2.5    # maximum plausible person height [m]

# ── Height estimator ──────────────────────────────────────────────────────────
HEIGHT_BUFFER_WINDOW_FRAMES = 60    # rolling-median window length [frames]
HEIGHT_BUFFER_MIN_SAMPLES   = 5     # samples required before median replaces prior
HEIGHT_OUTLIER_MAX_DEVIATION = 0.30  # reject sample if deviation > 30 % of median

# ── Gaze ray ──────────────────────────────────────────────────────────────────
GAZE_RAY_LENGTH_M = 3.0    # projected gaze line length [m]

# ── Temporal smoothing (EMA) ──────────────────────────────────────────────────
EMA_ALPHA_PIXEL_HEIGHT = 0.20   # pixel body-height span (drives z_root)
EMA_ALPHA_DISPLAY      = 0.25   # displayed scalars: dist / elev / azim / height

# ── Display ───────────────────────────────────────────────────────────────────
DISPLAY_MAX_WIDTH_PX  = 1580
DISPLAY_MAX_HEIGHT_PX = 840
SEEK_STEP_SECONDS     = 5

# ── COCO-17 joint indices ─────────────────────────────────────────────────────
# 0 nose   1 l.eye  2 r.eye  3 l.ear  4 r.ear
# 5 l.sho  6 r.sho  7 l.elb  8 r.elb  9 l.wri 10 r.wri
# 11 l.hip 12 r.hip 13 l.kne 14 r.kne 15 l.ank 16 r.ank
JOINT_NOSE       = 0
JOINT_LEFT_EAR   = 3
JOINT_RIGHT_EAR  = 4
JOINT_LEFT_HIP   = 11
JOINT_RIGHT_HIP  = 12
JOINT_LEFT_KNEE  = 13
JOINT_RIGHT_KNEE = 14
JOINT_LEFT_ANKLE  = 15
JOINT_RIGHT_ANKLE = 16

SKELETON_EDGES: list = [   # 1-based COCO-17 pairs
    (16,14),(14,12),(17,15),(15,13),(12,13),(6,12),(7,13),(6,7),
    (6,8),(7,9),(8,10),(9,11),(2,3),(1,2),(1,3),(2,4),(3,5),(4,6),(5,7),
]
SKELETON_EDGE_COLORS: list = [   # BGR, one colour per edge (rainbow)
    (255,  0,  0),(255, 85,  0),(255,170,  0),(255,255,  0),
    (170,255,  0),( 85,255,  0),(  0,255,  0),(  0,255, 85),
    (  0,255,170),(  0,255,255),(  0,170,255),(  0, 85,255),
    (  0,  0,255),( 85,  0,255),(170,  0,255),(255,  0,255),
    (255,  0,170),(255,  0, 85),
]

# ── Keypoint colours (BGR) ────────────────────────────────────────────────────
COLOR_KEYPOINT_NOSE  = (  0,   0, 255)   # red
COLOR_KEYPOINT_EAR   = (255,   0,   0)   # blue
COLOR_KEYPOINT_OTHER = (  0, 255,   0)   # green

# ── Gaze ray colours (BGR) ────────────────────────────────────────────────────
COLOR_GAZE_ORIGIN_DOT  = (  0, 255,   0)   # green
COLOR_GAZE_LINE        = (  0, 255, 255)   # yellow
COLOR_GAZE_ENDPOINT    = (  0,   0, 255)   # red
COLOR_EAR_CONNECTOR    = (150, 150, 150)   # grey

# ── Label colours (BGR) ──────────────────────────────────────────────────────
COLOR_LABEL_SIZE   = (255, 255,   0)   # cyan  – height and distance
COLOR_LABEL_ANGLES = (255,   0, 255)   # magenta – elevation, azimuth, gaze angles
COLOR_LABEL_ID     = (  0, 255, 255)   # yellow – person id

# ── HUD colours (BGR) ─────────────────────────────────────────────────────────
COLOR_HUD_RECORDING = (  0,   0, 255)   # red
COLOR_HUD_PAUSED    = (  0, 220, 255)   # amber
COLOR_HUD_RUNNING   = (  0, 255,   0)   # green
COLOR_HUD_CAM_INFO  = (200, 200, 200)   # grey
COLOR_HUD_HINT      = (255, 255, 255)   # white

# ── Visualisation sizes ───────────────────────────────────────────────────────
KEYPOINT_RADIUS          = 2
GAZE_ORIGIN_DOT_RADIUS   = 2
GAZE_ENDPOINT_RADIUS     = 3
SKELETON_LINE_THICKNESS  = 1
GAZE_LINE_THICKNESS      = 2
TEXT_OUTLINE_THICKNESS   = 2

# ── Label layout (offsets relative to topmost visible joint) ─────────────────
LABEL_ANCHOR_OFFSET_X   = -30    # horizontal shift [px]
LABEL_ANCHOR_OFFSET_Y   = -120   # vertical shift (upward) [px]
LABEL_LINE_SPACING_PX   = 26     # vertical gap between lines [px]
LABEL_FONT_SCALE_LARGE  = 0.65   # height and distance lines
LABEL_FONT_SCALE_SMALL  = 0.55   # angles, gaze, id
HUD_FONT_SCALE_STATUS   = 0.7
HUD_FONT_SCALE_CAM      = 0.45
