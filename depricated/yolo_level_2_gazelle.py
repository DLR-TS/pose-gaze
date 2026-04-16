"""yolo_gazelle.py — YOLO-Pose + GAZELLE gaze estimation, single-file script.

Reads a video file, runs YOLO11x-pose tracking per frame, and optionally
GAZELLE (DINOv2-ViT-B/14) gaze estimation every GAZELLE_SKIP frames.
Results are rendered at display resolution and shown in an OpenCV window.

Pipeline:
    cap.read() -> frame (iw x ih, BGR)
        |
        +-- YOLO .track() -> keypoints (N,17,2), confs (N,17), track IDs
        |       |
        |       +-- Smoother       per-person EMA on positions + confidences
        |       +-- calc_gaze      geometric gaze vector (ear-mid -> nose dir)
        |       +-- get_head_bbox  normalised head bbox [0,1] for GAZELLE input
        |
        +-- run_gazelle()  [every GAZELLE_SKIP frames, when enabled]
        |       |
        |       +-- _scene_crop    tight crop around all visible keypoints
        |       +-- _letterbox     resize crop to 448x448 with border padding
        |       +-- GAZELLE DNN    -> heatmap (N,64,64), inout score (N,)
        |       +-- _peak_to_frame back-project heatmap peak to original pixels
        |       +-- hm reprojected to full frame (ih x iw, float32)
        |
        +-- render vis (dw x dh)
                +-- Pass 1: heatmap overlay  (drawn under skeleton)
                +-- Pass 2: skeleton, keypoints, gaze vectors, labels

Coordinate spaces:
    original-pixel  (x,y) in [0..iw-1, 0..ih-1]  YOLO/GAZELLE output
    display-pixel   original * scale               drawn onto vis
    normalised      [0,1]^2                        GAZELLE bbox input
    letterbox       [0, GAZELLE_GZ_SIZE]^2         internal to run_gazelle

Keys: [Space] pause  [A/D] seek 5 s  [R] record  [G] GAZELLE  [H] heatmap  [B] bbox  [Q] quit
"""

from ultralytics import YOLO
import cv2
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR   = Path(__file__).parent.parent.resolve()
VIDEO_PATH = ROOT_DIR / "media/video_4112x2176_mvBlueCOUGAR-X109b_crop0-0-4112-2176.mp4"
MODEL_DIR  = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "recordings"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONF_THRESH       = 0.25   # YOLO detection threshold
KPT_THRESH        = 0.25   # keypoint confidence required for skeleton drawing
KPT_THR_GAZE      = 0.05   # lower threshold for gaze/bbox calculations
SKIP_SECONDS      = 5      # seek step in seconds (A/D keys)
GAZE_LEN          = 270    # geometric gaze ray length in original pixels
GAZE_ANGLE        = -5     # rotation correction applied to gaze direction (degrees)
DISP_W, DISP_H    = 1600, 900

GAZELLE_INOUT_THR = 0.20   # minimum inout score to treat gaze as in-frame
GAZELLE_HM_THRESH = 0.35   # heatmap values below this are fully transparent
GAZELLE_HM_ALPHA  = 0.55   # maximum heatmap overlay opacity
GAZELLE_GZ_SIZE   = 448    # model input size; must match the loaded checkpoint
GAZELLE_SKIP      = 3      # run GAZELLE every N frames
SCENE_MARGIN      = 0.25   # padding around keypoint cluster for scene crop

# Smoother hyper-parameters
SM_A_MOV, SM_A_STA, SM_A_SCR = 0.7, 0.4, 0.5
SM_MOV_HIST, SM_SCR_HIST      = 5, 2

# ---------------------------------------------------------------------------
# Device + models
# ---------------------------------------------------------------------------
device     = "cuda" if torch.cuda.is_available() else "cpu"
pose_model = YOLO(str(MODEL_DIR / "yolo11x-pose.pt")).to(device)

_GAZELLE_OK = False
_gaze_model = None
_gaze_tf    = None
_PIL        = None

try:
    from PIL import Image as _PIL
    _hub = MODEL_DIR / "hub"
    _hub.mkdir(exist_ok=True)
    torch.hub.set_dir(str(_hub))
    _gaze_model, _gaze_tf = torch.hub.load(
        "fkryan/gazelle", "gazelle_dinov2_vitb14_inout",
        pretrained=True, trust_repo=True)
    _gaze_model.eval().to(device)
    _GAZELLE_OK = True
    print(f"[GAZELLE] loaded on {device.upper()}")
except Exception as e:
    print(f"[GAZELLE] unavailable: {e}")

# ---------------------------------------------------------------------------
# Smoothers
# ---------------------------------------------------------------------------
class Smoother:
    """Per-track EMA smoother for keypoint positions and confidences.

    Alpha scales with recent motion magnitude: high motion -> alpha near
    SM_A_MOV (responsive), low motion -> alpha near SM_A_STA (stable).
    Confidences are additionally stabilised with a two-frame max-blend.

    Interface:
        smooth(pid, kpts, scrs) -> (smoothed_kpts, smoothed_scrs)
        last(pid, idx)          -> last smoothed value for keypoint idx, or None
        cleanup(ids)            -> drop state for track IDs no longer active
    """

    def __init__(self):
        self.pk = {}   # last smoothed positions
        self.ps = {}   # last smoothed confidences
        self.sh = {}   # short confidence history for max-blend
        self.mh = {}   # motion magnitude history

    def smooth(self, pid, kpts, scrs):
        mov = (1.0 if pid not in self.pk
               else min(1.0, np.linalg.norm(kpts - self.pk[pid]) / 50.0))
        self.mh.setdefault(pid, []).append(mov)
        if len(self.mh[pid]) > SM_MOV_HIST:
            self.mh[pid].pop(0)
        alpha = SM_A_STA + (SM_A_MOV - SM_A_STA) * np.mean(self.mh[pid])
        if pid not in self.pk:
            self.pk[pid] = kpts.copy()
            self.ps[pid] = scrs.copy()
            self.sh[pid] = [scrs.copy()]
            return kpts, scrs
        sk = alpha * kpts + (1 - alpha) * self.pk[pid]
        ss = SM_A_SCR * scrs + (1 - SM_A_SCR) * self.ps[pid]
        self.sh[pid].append(ss.copy())
        if len(self.sh[pid]) > SM_SCR_HIST:
            self.sh[pid].pop(0)
        if len(self.sh[pid]) >= 2:
            ss = 0.7 * ss + 0.3 * np.maximum(self.sh[pid][-2], self.sh[pid][-1])
        self.pk[pid] = sk
        self.ps[pid] = ss
        return sk, ss

    def last(self, pid, idx):
        """Return the last smoothed position for keypoint idx, or None."""
        return self.pk[pid][idx] if pid in self.pk else None

    def cleanup(self, ids):
        active = set(ids)
        for pid in list(self.pk):
            if pid not in active:
                del self.pk[pid], self.ps[pid]
                self.sh.pop(pid, None)
                self.mh.pop(pid, None)


class GazeSmoother:
    """Per-track EMA smoother for the geometric gaze ray (start and end point)."""

    def __init__(self, alpha=0.6):
        self.a = alpha
        self.p = {}

    def smooth(self, pid, s, e):
        if pid not in self.p:
            self.p[pid] = (s, e)
            return s, e
        ps, pe = self.p[pid]
        r = self.a * s + (1 - self.a) * ps, self.a * e + (1 - self.a) * pe
        self.p[pid] = r
        return r

    def cleanup(self, ids):
        active = set(ids)
        for pid in list(self.p):
            if pid not in active:
                del self.p[pid]


smoother      = Smoother()
gaze_smoother = GazeSmoother()

# ---------------------------------------------------------------------------
# Keypoint indices, skeleton, colours  (COCO-17, 0-based)
# ---------------------------------------------------------------------------
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
HEAD_KPS = [NOSE, LEYE, REYE, LEAR, REAR]

SKELETON = [
    (15,13),(13,11),(16,14),(14,12),(11,12),(5,11),(6,12),(5,6),
    (5,7),(6,8),(7,9),(8,10),(1,2),(0,1),(0,2),(1,3),(2,4),(3,5),(4,6),
]

# one colour per skeleton segment, roughly hue-equidistant
SKEL_COLORS = [
    (255,  0,  0),(255, 85,  0),(255,170,  0),(255,255,  0),
    (170,255,  0),( 85,255,  0),(  0,255,  0),(  0,255, 85),
    (  0,255,170),(  0,255,255),(  0,170,255),(  0, 85,255),
    (  0,  0,255),( 85,  0,255),(170,  0,255),(255,  0,255),
    (255,  0,170),(255,  0, 85),
]

# per-person colours (BGR), hue-equidistant, up to 6 simultaneous tracks
_PCOLS = [
    (255, 255,   0),   # cyan      H=90
    (  0,  80, 255),   # orange    H=15
    (  0, 220,   0),   # green     H=120
    (255,   0, 200),   # magenta   H=300
    (  0, 200, 255),   # yellow    H=45
    (255,  50,   0),   # blue      H=210
]


def pcol(tid):
    return _PCOLS[int(tid) % len(_PCOLS)]


def otxt(img, text, pos, scale, col):
    """Draw text with a black outline for readability on any background."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, col,       1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def calc_gaze(kpts, confs, pid=None):
    """Compute a geometric gaze ray from nose and ear-midpoint keypoints.

    Direction: ear-midpoint -> nose, rotated by GAZE_ANGLE degrees.
    Falls back to the smoother's last known ear positions when current
    confidence is below KPT_THR_GAZE.

    Returns (start_px, end_px, valid) in original-pixel coordinates,
    or (None, None, False) if required keypoints are unavailable.
    """
    if confs[NOSE] <= KPT_THR_GAZE:
        return None, None, False
    le = kpts[LEAR] if confs[LEAR] > KPT_THR_GAZE else smoother.last(pid, LEAR)
    re = kpts[REAR] if confs[REAR] > KPT_THR_GAZE else smoother.last(pid, REAR)
    if le is None or re is None:
        return None, None, False
    ear_mid = (le + re) / 2.0
    gv = kpts[NOSE] - ear_mid
    gl = np.linalg.norm(gv)
    if gl < 1e-6:
        return None, None, False
    gn = gv / gl
    a  = -np.radians(GAZE_ANGLE)
    ca, sa = np.cos(a), np.sin(a)
    gd = np.array([ca * gn[0] - sa * gn[1],
                   sa * gn[0] + ca * gn[1]])
    return ear_mid, ear_mid + gd * GAZE_LEN, True


def get_head_bbox_norm(kpts, confs, pid=None):
    """Return a normalised head bounding box [0,1]^2 from head keypoints.

    Uses global iw/ih (available after cap.open()).
    Falls back to the smoother for LEAR/REAR when confidence is low.
    Padding is max(kp_width, kp_height, 1.5% of max(iw,ih)) * 0.30.
    Returns None when fewer than 2 keypoints are usable.
    """
    pts = []
    for j in HEAD_KPS:
        if confs[j] > KPT_THR_GAZE:
            pts.append(kpts[j])
        elif j in (LEAR, REAR) and pid is not None:
            p = smoother.last(pid, j)
            if p is not None:
                pts.append(p)
    if len(pts) < 2:
        return None
    arr = np.array(pts)
    w   = arr[:, 0].max() - arr[:, 0].min()
    h   = arr[:, 1].max() - arr[:, 1].min()
    pad = max(w, h, max(iw, ih) * 0.015) * 0.30
    x1  = max(0.0, (arr[:, 0].min() - pad) / iw)
    y1  = max(0.0, (arr[:, 1].min() - pad) / ih)
    x2  = min(1.0, (arr[:, 0].max() + pad) / iw)
    y2  = min(1.0, (arr[:, 1].max() + pad) / ih)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


# ---------------------------------------------------------------------------
# GAZELLE helpers
# ---------------------------------------------------------------------------
def _scene_crop(all_kpts, all_confs, fw, fh):
    """Return a bounding box (cx1,cy1,cx2,cy2) covering all visible keypoints.

    A single shared crop is used for all persons in the frame.
    Padding is SCENE_MARGIN * the spatial span of the keypoint cluster.
    Falls back to the full frame when no confident keypoints are found.
    """
    pts = [kpts[confs > KPT_THR_GAZE]
           for kpts, confs in zip(all_kpts, all_confs)]
    pts = [p for p in pts if len(p)]
    if not pts:
        return 0, 0, fw, fh
    arr  = np.vstack(pts)
    span = max(arr[:, 0].max() - arr[:, 0].min(),
               arr[:, 1].max() - arr[:, 1].min(), 100.0)
    pad  = span * SCENE_MARGIN
    return (max(0,  int(arr[:, 0].min() - pad)),
            max(0,  int(arr[:, 1].min() - pad)),
            min(fw, int(arr[:, 0].max() + pad)),
            min(fh, int(arr[:, 1].max() + pad)))


def _letterbox(img):
    """Fit img into a GAZELLE_GZ_SIZE square with centred zero-padding.

    Aspect ratio is preserved. Returns (padded, rw, rh, pad_x, pad_y)
    where rw/rh is the size of the scaled content and pad_x/pad_y is
    its top-left offset inside the square output.
    """
    h, w = img.shape[:2]
    s    = min(GAZELLE_GZ_SIZE / w, GAZELLE_GZ_SIZE / h)
    rw, rh = int(w * s), int(h * s)
    px, py = (GAZELLE_GZ_SIZE - rw) // 2, (GAZELLE_GZ_SIZE - rh) // 2
    out = np.zeros((GAZELLE_GZ_SIZE, GAZELLE_GZ_SIZE, 3), dtype=np.uint8)
    out[py:py + rh, px:px + rw] = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    return out, rw, rh, px, py


def _bbox_to_padded(bb, cx1, cy1, cx2, cy2, rw, rh, px, py):
    """Map a normalised bbox from original-frame space into letterbox space.

    Path: normalised original -> scene-crop pixels -> normalised crop ->
          scaled-content pixels (rw x rh) -> normalised letterbox output.

    cw/ch are clamped to 1 to avoid division by zero on degenerate crops.
    """
    cw = max(cx2 - cx1, 1)
    ch = max(cy2 - cy1, 1)
    gs = GAZELLE_GZ_SIZE
    x1 = float(np.clip((bb[0] * iw - cx1) / cw, 0, 1))
    y1 = float(np.clip((bb[1] * ih - cy1) / ch, 0, 1))
    x2 = float(np.clip((bb[2] * iw - cx1) / cw, 0, 1))
    y2 = float(np.clip((bb[3] * ih - cy1) / ch, 0, 1))
    return [(x1 * rw + px) / gs, (y1 * rh + py) / gs,
            (x2 * rw + px) / gs, (y2 * rh + py) / gs]


def _peak_to_frame(col64, row64, cx1, cy1, cx2, cy2, rw, rh, px, py):
    """Back-project a 64x64 heatmap cell to original-frame pixel coordinates.

    Inverse of the _bbox_to_padded transform:
        cell centre -> letterbox pixel -> unpad -> uncrop -> original frame.

    col64 is the x-index (column), row64 is the y-index (row) in the 64x64 grid.
    """
    gs  = GAZELLE_GZ_SIZE
    ppx = (col64 + 0.5) / 64.0 * gs
    ppy = (row64 + 0.5) / 64.0 * gs
    ox  = int(np.clip((ppx - px) / rw * (cx2 - cx1), 0, cx2 - cx1 - 1))
    oy  = int(np.clip((ppy - py) / rh * (cy2 - cy1), 0, cy2 - cy1 - 1))
    return cx1 + ox, cy1 + oy


def _make_empty_gz_entry(kpts, confs, tid):
    """Build a gz_cache entry with head bbox and ear-midpoint but no heatmap data."""
    le = kpts[LEAR] if confs[LEAR] > KPT_THR_GAZE else smoother.last(tid, LEAR)
    re = kpts[REAR] if confs[REAR] > KPT_THR_GAZE else smoother.last(tid, REAR)
    em = (le + re) / 2.0 if (le is not None and re is not None) else None
    return dict(bb_norm=get_head_bbox_norm(kpts, confs, tid),
                hm_frame=None, peak_px=None, inout=0.0,
                in_frame=False, ear_mid_px=em)


def run_gazelle(frame, all_kpts, all_confs, track_ids):
    """Run GAZELLE on all tracked persons in a single shared scene crop.

    Returns Dict[int(tid) -> dict] with keys:
        bb_norm     normalised head bbox, or None
        hm_frame    float32 array (ih x iw), heatmap reprojected to full frame
        peak_px     (x, y) in original pixels, or None when gaze is out-of-frame
        inout       raw inout score from the model
        in_frame    bool, inout >= GAZELLE_INOUT_THR
        ear_mid_px  (x, y) ear midpoint in original pixels, or None
    """
    fh, fw = frame.shape[:2]
    result = {int(tid): _make_empty_gz_entry(kpts, confs, int(tid))
              for kpts, confs, tid in zip(all_kpts, all_confs, track_ids)}

    if not (_GAZELLE_OK and _gaze_model is not None):
        return result

    tid_bb = {tid: result[tid]["bb_norm"]
              for tid in result if result[tid]["bb_norm"] is not None}
    if not tid_bb:
        return result

    cx1, cy1, cx2, cy2 = _scene_crop(all_kpts, all_confs, fw, fh)
    g_frame, rw, rh, pad_x, pad_y = _letterbox(frame[cy1:cy2, cx1:cx2])
    img_pil = _PIL.fromarray(cv2.cvtColor(g_frame, cv2.COLOR_BGR2RGB))
    tids_v  = list(tid_bb.keys())
    bboxes  = [_bbox_to_padded(tid_bb[t], cx1, cy1, cx2, cy2, rw, rh, pad_x, pad_y)
               for t in tids_v]

    with torch.no_grad():
        out = _gaze_model({"images": _gaze_tf(img_pil).unsqueeze(0).to(device),
                           "bboxes": [bboxes]})

    for bi, tid in enumerate(tids_v):
        hm       = out["heatmap"][0][bi].cpu().numpy()
        inout    = float(out["inout"][0][bi]) if out.get("inout") is not None else 1.0
        hm_norm  = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        in_frame = inout >= GAZELLE_INOUT_THR

        # argmax over flat (64*64) index: row = pk//64, col = pk%64
        pk      = int(np.argmax(hm_norm))
        peak_px = (_peak_to_frame(pk % 64, pk // 64, cx1, cy1, cx2, cy2,
                                  rw, rh, pad_x, pad_y)
                   if in_frame else None)

        # upsample heatmap: 64x64 -> letterbox (448x448) -> remove padding
        # -> resize to crop dimensions -> place into full-frame array
        hm_full = cv2.resize(hm_norm, (GAZELLE_GZ_SIZE, GAZELLE_GZ_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        hm_crop = hm_full[pad_y:pad_y + rh, pad_x:pad_x + rw]
        cw, ch  = cx2 - cx1, cy2 - cy1
        hm_frame = np.zeros((fh, fw), dtype=np.float32)
        hm_frame[cy1:cy2, cx1:cx2] = cv2.resize(hm_crop, (cw, ch),
                                                 interpolation=cv2.INTER_LINEAR)
        result[tid].update(hm_frame=hm_frame, peak_px=peak_px,
                           inout=inout, in_frame=in_frame)
    return result


def overlay_heatmap(vis, hm_frame, color_bgr):
    """Alpha-blend a heatmap onto vis in-place using the given BGR colour.

    hm_frame is in original-frame coordinates (ih x iw, float32, [0,1]).
    Values below GAZELLE_HM_THRESH are fully transparent; above that the
    opacity ramps linearly up to GAZELLE_HM_ALPHA.

    np.full is used instead of np.full_like to correctly broadcast the
    three-channel BGR tuple across all columns.
    """
    dh_, dw_ = vis.shape[:2]
    hm    = cv2.resize(hm_frame, (dw_, dh_), interpolation=cv2.INTER_LINEAR)
    above = np.clip((hm - GAZELLE_HM_THRESH) / (1.0 - GAZELLE_HM_THRESH + 1e-8),
                    0.0, 1.0)
    alpha = (above * GAZELLE_HM_ALPHA).astype(np.float32)[:, :, np.newaxis]
    layer = np.full((dh_, dw_, 3), color_bgr, dtype=np.float32)
    vis[:] = np.clip(vis.astype(np.float32) * (1.0 - alpha) + layer * alpha,
                     0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Video init  (iw, ih must be defined before any call to get_head_bbox_norm)
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open: {VIDEO_PATH}")

iw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ih    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps   = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
scale = min(DISP_W / iw, DISP_H / ih, 1.0)
dw, dh = int(iw * scale), int(ih * scale)
LABEL_LINE_H = 16   # vertical spacing between stacked per-person labels

print(f"video: {iw}x{ih}  display: {dw}x{dh}  fps: {fps:.1f}  "
      f"GAZELLE: {'on' if _GAZELLE_OK else 'off'}")
print("keys: [A/D] seek  [Space] pause  [R] record  [G] GAZELLE  [H] heatmap  [B] bbox  [Q] quit")

# ---------------------------------------------------------------------------
# Playback state
# ---------------------------------------------------------------------------
paused     = False
recording  = False
writer     = None
out_file   = None
gazelle_on    = True
show_heatmap  = True
show_headbox  = True
frame_idx     = 0
fps_cnt       = 0
gz_skip_cnt   = 0
t_last        = time.time()
last_vis      = None
gz_cache      = {}
num_persons   = 0
all_kpts = all_confs = track_ids = boxes_conf = None

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
try:
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("end of video")
                paused = True
                continue
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            r = pose_model.track(frame, conf=CONF_THRESH, persist=True,
                                 verbose=False, device=device)[0]

            vis = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
            num_persons = 0

            if r.keypoints is not None and len(r.keypoints) > 0:
                all_kpts   = r.keypoints.xy.cpu().numpy()       # (N,17,2) original pixels
                all_confs  = r.keypoints.conf.cpu().numpy()     # (N,17)
                track_ids  = (r.boxes.id.cpu().numpy().astype(int)
                              if r.boxes.id is not None
                              else np.arange(len(all_kpts)) + 10000)
                boxes_conf = r.boxes.conf.cpu().numpy()
                num_persons = len(all_kpts)

                smoother.cleanup(track_ids)
                gaze_smoother.cleanup(track_ids)

                if gazelle_on and _GAZELLE_OK:
                    gz_skip_cnt += 1
                    if gz_skip_cnt % GAZELLE_SKIP == 0 or not gz_cache:
                        gz_cache = run_gazelle(frame, all_kpts, all_confs, track_ids)
                    # persons that appeared since the last GAZELLE frame get a
                    # geometry-only entry so the rest of the render loop is uniform
                    for kpts, confs, tid in zip(all_kpts, all_confs, track_ids):
                        if int(tid) not in gz_cache:
                            gz_cache[int(tid)] = _make_empty_gz_entry(kpts, confs, int(tid))
                else:
                    gz_cache = {int(tid): _make_empty_gz_entry(kpts, confs, int(tid))
                                for kpts, confs, tid in zip(all_kpts, all_confs, track_ids)}

                # Pass 1: heatmaps drawn before skeleton so skeleton sits on top
                if show_heatmap and gazelle_on:
                    for tid in track_ids:
                        gz = gz_cache.get(int(tid), {})
                        if gz.get("hm_frame") is not None and gz.get("in_frame"):
                            overlay_heatmap(vis, gz["hm_frame"], pcol(int(tid)))

                # Pass 2: skeleton, keypoints, gaze vectors, labels
                for i, (kpts, tid, bconf) in enumerate(zip(all_kpts, track_ids, boxes_conf)):
                    tid    = int(tid)
                    sk, sc = smoother.smooth(tid, kpts, all_confs[i])
                    skd    = sk * scale   # keypoints in display pixels
                    gz     = gz_cache.get(tid, {})
                    col    = pcol(tid)

                    for j, (st, en) in enumerate(SKELETON):
                        if sc[st] > KPT_THRESH and sc[en] > KPT_THRESH:
                            cv2.line(vis,
                                     tuple(map(int, skd[st])),
                                     tuple(map(int, skd[en])),
                                     SKEL_COLORS[j % len(SKEL_COLORS)], 1, cv2.LINE_AA)

                    # geometric gaze: ear-midpoint as origin, direction toward nose
                    s, e, gvalid = calc_gaze(sk, sc, tid)
                    if gvalid:
                        s, e = gaze_smoother.smooth(tid, s, e)
                        sd, ed = s * scale, e * scale
                        if sc[LEAR] > KPT_THRESH and sc[REAR] > KPT_THRESH:
                            cv2.line(vis, tuple(map(int, skd[LEAR])),
                                     tuple(map(int, skd[REAR])),
                                     (130, 130, 130), 1, cv2.LINE_AA)
                        cv2.circle(vis, tuple(map(int, sd)), 2, (0, 255, 0), -1)
                        cv2.line(vis, tuple(map(int, sd)), tuple(map(int, ed)),
                                 (0, 255, 255), 1, cv2.LINE_AA)
                        cv2.circle(vis, tuple(map(int, ed)), 2, (0, 0, 255), -1)

                    # nose=red, ears=blue, other keypoints=green
                    for j, (kx, ky) in enumerate(skd):
                        if sc[j] > KPT_THRESH:
                            kc = ((0,0,255) if j == NOSE
                                  else (255,0,0) if j in (LEAR, REAR)
                                  else (0,255,0))
                            cv2.circle(vis, (int(kx), int(ky)), 2, kc, -1)

                    # label anchor: top edge of head bbox, or highest visible keypoint
                    if gz.get("bb_norm"):
                        bbox_top = int(gz["bb_norm"][1] * dh)
                        label_x  = int(gz["bb_norm"][0] * dw)
                    else:
                        vp = skd[sc > KPT_THRESH]
                        tp = vp[np.argmin(vp[:, 1])] if len(vp) else skd[0]
                        bbox_top = int(tp[1]) - 50
                        label_x  = int(tp[0]) - 20

                    otxt(vis, f"{bconf:.2f}", (label_x, max(12, bbox_top - 4)),
                         0.38, (0, 255, 255))
                    otxt(vis, f"ID:{tid}", (label_x, max(12, bbox_top - 4 - LABEL_LINE_H)),
                         0.45, (0, 255, 255))
                    if gazelle_on and gz.get("inout", 0.0) > 0:
                        otxt(vis, f"io:{gz['inout']:.2f}",
                             (label_x, max(12, bbox_top - 4 - 2 * LABEL_LINE_H)),
                             0.38, col if gz.get("in_frame") else (100, 100, 100))

                    if show_headbox and gz.get("bb_norm"):
                        x1n, y1n, x2n, y2n = gz["bb_norm"]
                        cv2.rectangle(vis,
                                      (int(x1n * dw), int(y1n * dh)),
                                      (int(x2n * dw), int(y2n * dh)),
                                      col, 1, cv2.LINE_AA)

                    # GAZELLE gaze vector: line from ear-midpoint to heatmap peak.
                    # peak_px uses an explicit None check because (0,0) is falsy.
                    if gz.get("peak_px") is not None and gz.get("in_frame"):
                        px_d = int(gz["peak_px"][0] * scale)
                        py_d = int(gz["peak_px"][1] * scale)

                        em = gz.get("ear_mid_px")
                        if em is not None:
                            origin_d = (int(em[0] * scale), int(em[1] * scale))
                        elif sc[NOSE] > KPT_THR_GAZE:
                            origin_d = (int(skd[NOSE][0]), int(skd[NOSE][1]))
                        else:
                            origin_d = None

                        if origin_d is not None:
                            cv2.line(vis, origin_d, (px_d, py_d), col, 1, cv2.LINE_AA)

                        r2 = 10
                        cv2.circle(vis, (px_d, py_d), r2, col, 1, cv2.LINE_AA)
                        cv2.line(vis, (px_d - r2, py_d), (px_d + r2, py_d), col, 1, cv2.LINE_AA)
                        cv2.line(vis, (px_d, py_d - r2), (px_d, py_d + r2), col, 1, cv2.LINE_AA)
                        cv2.circle(vis, (px_d, py_d), 3, col, -1)

        if paused:
            vis = (last_vis.copy() if last_vis is not None
                   else np.zeros((dh, dw, 3), dtype=np.uint8))
            cv2.putText(vis, "PAUSED  [Space]", (dw // 2 - 150, dh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        curr_t = frame_idx / fps if fps > 0 else 0
        gz_lbl = ("GAZELLE:on"      if (gazelle_on and _GAZELLE_OK) else
                  "GAZELLE:off"     if not gazelle_on
                  else "GAZELLE:missing")
        flags  = ("+HM"   if (show_heatmap and gazelle_on and _GAZELLE_OK) else "") + \
                 ("+bbox" if show_headbox else "")

        cv2.putText(vis, f"P:{num_persons}  {curr_t:.1f}s / {total/fps:.1f}s"
                    + ("  [REC]" if recording else ""),
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if recording else (0, 255, 0), 1)
        cv2.putText(vis, gz_lbl + flags,
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 220, 255) if (gazelle_on and _GAZELLE_OK) else (80, 80, 80), 1)
        cv2.putText(vis, "[A/D] seek  [Space] pause  [R] rec  [G] GAZELLE  [H] HM  [B] bbox  [Q] quit",
                    (10, dh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        if recording and writer:
            writer.write(vis)
        if not paused:
            last_vis = vis.copy()

        fps_cnt += 1
        now = time.time()
        if now - t_last >= 1.0:
            print(f"fps:{fps_cnt/(now-t_last):.1f}  frame:{frame_idx}/{total}"
                  f"  t:{curr_t:.1f}s  {gz_lbl}  {device.upper()}")
            t_last  = now
            fps_cnt = 0

        cv2.imshow("YOLO-Pose + GAZELLE", vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("g"):
            gazelle_on  = not gazelle_on
            gz_cache    = {}
            gz_skip_cnt = 0
            gaze_smoother = GazeSmoother()
            print(f"GAZELLE: {'on' if gazelle_on else 'off'}")
        elif key == ord("h"):
            show_heatmap = not show_heatmap
        elif key == ord("b"):
            show_headbox = not show_headbox
        elif key == ord("r"):
            if not recording:
                recording = True
                ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_file = OUTPUT_DIR / f"recording_{ts}.mp4"
                writer   = cv2.VideoWriter(str(out_file),
                                           cv2.VideoWriter_fourcc(*"mp4v"),
                                           fps, (dw, dh))
                print(f"recording: {out_file}")
            else:
                recording = False
                writer.release()
                writer = None
                print(f"saved: {out_file}")
        elif key == ord("a"):
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - SKIP_SECONDS * fps))
            smoother      = Smoother()
            gaze_smoother = GazeSmoother()
            gz_cache      = {}
            gz_skip_cnt   = 0
            paused        = False
        elif key == ord("d"):
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    min(total - 1, cap.get(cv2.CAP_PROP_POS_FRAMES) + SKIP_SECONDS * fps))
            smoother      = Smoother()
            gaze_smoother = GazeSmoother()
            gz_cache      = {}
            gz_skip_cnt   = 0
            paused        = False

finally:
    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    if device == "cuda":
        torch.cuda.empty_cache()
