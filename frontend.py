"""frontend.py – video capture loop, overlay drawing, recording, keyboard control.

Keys:
  R       – toggle recording (MP4 + NDJSON)
  Space   – pause / resume
  A / D   – seek ±SEEK_STEP_SECONDS
  Q / Esc – quit
"""
import math, time
from collections import deque
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from backend import (
    DEVICE,
    init_camera, process_frame,
    build_detection_message,
    PersonData, proj3d2d, gaze_angles_3d,
    get_ground_plane,
)
from basics import (
    VIDEO_PATH, OUTPUT_DIR, FPS_FALLBACK, SEEK_STEP_SECONDS,
    DISPLAY_MAX_WIDTH_PX, DISPLAY_MAX_HEIGHT_PX,
    JOINT_NOSE, JOINT_LEFT_EAR, JOINT_RIGHT_EAR,
    COLOR_HUD_RECORDING, COLOR_HUD_PAUSED, COLOR_HUD_RUNNING,
    COLOR_HUD_CAM_INFO, COLOR_HUD_HINT,
    COLOR_LABEL_DIST, COLOR_LABEL_ANGLES, COLOR_LABEL_GAZE,
    COLOR_KEYPOINT_NOSE, COLOR_KEYPOINT_EAR, COLOR_KEYPOINT_OTHER,
    COLOR_EAR_CONNECTOR, COLOR_GAZE_ORIGIN_DOT, COLOR_GAZE_LINE, COLOR_GAZE_ENDPOINT,
    COLOR_BEARING_RAY, COLOR_GROUND_LONG_LINES,
    SKELETON_EDGES, SKELETON_EDGE_COLORS,
    SKELETON_LINE_THICKNESS, GAZE_LINE_THICKNESS, TEXT_OUTLINE_THICKNESS,
    LABEL_FONT_SCALE_SMALL,
    HUD_FONT_SCALE_STATUS, HUD_FONT_SCALE_CAM,
    BEARING_RAY_THICKNESS, BEARING_ENDPOINT_COLOR, BEARING_ENDPOINT_RADIUS,
    GAZE_RAY_LENGTH_M,
    GAZE_ORIGIN_DOT_RADIUS, GAZE_ENDPOINT_RADIUS, KEYPOINT_RADIUS,
    LABEL_LINE_SPACING_PX, LABEL_SMOOTH_WINDOW_S,
    GROUND_PLANE_ENABLED, GROUND_PLANE_MAX_DEPTH_M, GROUND_PLANE_LINE_THICKNESS,
    GROUND_PLANE_FILL_ALPHA, COLOR_GROUND_PLANE_FILL,
    GROUND_PLANE_X_STEP_M, GROUND_PLANE_X_HALF_M,
    GROUND_PLANE_NEAR_FRAC, GROUND_PLANE_FAR_FRAC,
    GroundPlane,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Label smoother ────────────────────────────────────────────────────────────
class LabelSmoother:
    """Per-person trailing-average smoother for numerical overlay values."""

    def __init__(self):
        self._buf: dict = {}   # (pid, field) -> deque[(t, v)]

    def push(self, person_id: int, field: str,
             value: float, t_s: float) -> float:
        key = (person_id, field)
        buf = self._buf.setdefault(key, deque())
        buf.append((t_s, value))
        cutoff = t_s - LABEL_SMOOTH_WINDOW_S
        while buf and buf[0][0] < cutoff:
            buf.popleft()
        return sum(v for _, v in buf) / len(buf)


_label_smoother  = LabelSmoother()
_h_cam_ema: Optional[float] = None
_H_CAM_EMA_ALPHA = 0.05   # ~60 frames to 95% at 13 fps


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _text(img: np.ndarray, text: str, pos: tuple,
          scale: float, color: tuple) -> None:
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), TEXT_OUTLINE_THICKNESS + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color,   TEXT_OUTLINE_THICKNESS)


def draw_ground_plane(
    img: np.ndarray, gp: Optional[GroundPlane],
    fx: float, fy: float, cx: float, cy: float,
    ds: float = 1.0,
) -> None:
    global _h_cam_ema
    h_img, w_img = img.shape[:2]
    y_near = int(h_img * GROUND_PLANE_NEAR_FRAC)
    y_far  = int(h_img * GROUND_PLANE_FAR_FRAC)

    def _row_to_z(y_px: float) -> float:
        if gp is not None and abs(float(gp.normal[1])) > 1e-3:
            n = gp.normal.astype(np.float64)
            c = gp.centroid.astype(np.float64)
            d = -float(np.dot(n, c))
            ray   = np.array([0., (y_px - cy) / fy, 1.0])
            denom = float(np.dot(n, ray))
            if abs(denom) > 1e-6:
                t = -d / denom
                if t > 0.01:
                    return float(t)
        dy = y_px - cy
        if dy > 2.0:
            return max(0.3, 1.2 * fy / dy)
        return float(GROUND_PLANE_MAX_DEPTH_M)

    z_near = _row_to_z(y_near)
    z_far  = _row_to_z(y_far)
    if z_near <= 0.0 or z_far <= z_near * 1.05:
        z_near, z_far = 1.5, GROUND_PLANE_MAX_DEPTH_M

    x_half = GROUND_PLANE_X_HALF_M

    def _x_px(x_m: float, z: float) -> int:
        return int(x_m / z * fx + cx)

    xl_n = max(0, min(w_img-1, _x_px(-x_half, z_near)))
    xr_n = max(0, min(w_img-1, _x_px( x_half, z_near)))
    xl_f = max(0, min(w_img-1, _x_px(-x_half, z_far)))
    xr_f = max(0, min(w_img-1, _x_px( x_half, z_far)))

    poly = np.array([[xl_n, y_near], [xr_n, y_near],
                     [xr_f, y_far],  [xl_f, y_far]], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], COLOR_GROUND_PLANE_FILL)
    cv2.addWeighted(overlay, GROUND_PLANE_FILL_ALPHA,
                    img,     1.0 - GROUND_PLANE_FILL_ALPHA, 0, img)

    lthick = max(1, round(GROUND_PLANE_LINE_THICKNESS * ds))
    for x_m in np.arange(-x_half, x_half + 1e-9, GROUND_PLANE_X_STEP_M):
        xn = max(0, min(w_img-1, _x_px(x_m, z_near)))
        xf = max(0, min(w_img-1, _x_px(x_m, z_far)))
        cv2.line(img, (xn, y_near), (xf, y_far),
                 COLOR_GROUND_LONG_LINES, lthick, cv2.LINE_AA)

    # ── Camera height above ground plane ──────────────────────────────────────
    if gp is not None and abs(float(gp.normal[1])) > 1e-3:
        h_raw  = abs(float(np.dot(gp.normal.astype(np.float64),
                                  gp.centroid.astype(np.float64))))
        _h_cam_ema = (h_raw if _h_cam_ema is None
                      else _H_CAM_EMA_ALPHA * h_raw
                           + (1.0 - _H_CAM_EMA_ALPHA) * _h_cam_ema)
        label = f"cam:{_h_cam_ema:.2f}m"
        fs = LABEL_FONT_SCALE_SMALL * ds
        (tw, th), bl = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, fs, TEXT_OUTLINE_THICKNESS + 2)
        lx = max(2,      min(w_img - tw - 2, int(cx) + 8))
        ly = max(th + 2, min(h_img - bl - 2, int(cy) - 8))
        _text(img, label, (lx, ly), fs, (0, 255, 255))


def draw_person(
    img: np.ndarray, p: PersonData,
    fx: float, fy: float, cx: float, cy: float,
    ds: float = 1.0,
    timestamp_s: float = 0.0,
) -> None:
    jm     = p.joints_meters
    vis    = p.joint_visible
    safe_d = np.where(vis, jm[:, 2], 1.0)
    px_x   = ((jm[:, 0] / safe_d) * fx + cx).astype(np.int32)
    px_y   = ((jm[:, 1] / safe_d) * fy + cy).astype(np.int32)

    el  = math.radians(p.elevation_deg)
    az  = math.radians(p.azimuth_deg)
    hd  = p.distance_m * math.cos(el)
    torso_3d = np.array([hd * math.sin(az),
                         -p.distance_m * math.sin(el),
                         hd * math.cos(az)], dtype=np.float32)
    torso_px = proj3d2d(torso_3d, fx, fy, cx, cy)
    if torso_px is not None:
        ox, oy = int(cx), int(cy)
        tx, ty = int(torso_px[0]), int(torso_px[1])
        cv2.line(img, (ox, oy), (tx, ty), COLOR_BEARING_RAY,
                 max(1, round(BEARING_RAY_THICKNESS * ds)), cv2.LINE_AA)
        cv2.circle(img, (tx, ty), max(1, round(BEARING_ENDPOINT_RADIUS * ds)),
                   BEARING_ENDPOINT_COLOR, -1)
        mid_px = proj3d2d((torso_3d * 0.5).astype(np.float32), fx, fy, cx, cy)
        if mid_px is not None:
            sm_dist = _label_smoother.push(p.person_id, "dist_m",
                                           p.distance_m, timestamp_s)
            _text(img, f"{sm_dist:.1f}m",
                  (int(mid_px[0]) + 4, int(mid_px[1]) - 4),
                  LABEL_FONT_SCALE_SMALL * ds, COLOR_BEARING_RAY)

    sthick = max(1, round(SKELETON_LINE_THICKNESS * ds))
    for i, (a1, b1) in enumerate(SKELETON_EDGES):
        a, b = a1 - 1, b1 - 1
        if vis[a] and vis[b]:
            cv2.line(img, (px_x[a], px_y[a]), (px_x[b], px_y[b]),
                     SKELETON_EDGE_COLORS[i % len(SKELETON_EDGE_COLORS)],
                     sthick, cv2.LINE_AA)

    gaze_label = "gaze:n/a"
    if p.gaze_origin_m is not None and p.gaze_direction is not None:
        if vis[JOINT_LEFT_EAR] and vis[JOINT_RIGHT_EAR]:
            cv2.line(img,
                     (px_x[JOINT_LEFT_EAR],  px_y[JOINT_LEFT_EAR]),
                     (px_x[JOINT_RIGHT_EAR], px_y[JOINT_RIGHT_EAR]),
                     COLOR_EAR_CONNECTOR, sthick, cv2.LINE_AA)
        ep3 = p.gaze_origin_m + p.gaze_direction * GAZE_RAY_LENGTH_M
        op  = proj3d2d(p.gaze_origin_m, fx, fy, cx, cy)
        ep  = proj3d2d(ep3,             fx, fy, cx, cy)
        if op is not None and ep is not None:
            cv2.circle(img, (int(op[0]), int(op[1])),
                       max(1, round(GAZE_ORIGIN_DOT_RADIUS * ds)),
                       COLOR_GAZE_ORIGIN_DOT, -1)
            cv2.line(img, (int(op[0]), int(op[1])), (int(ep[0]), int(ep[1])),
                     COLOR_GAZE_LINE, max(1, round(GAZE_LINE_THICKNESS * ds)),
                     cv2.LINE_AA)
            cv2.circle(img, (int(ep[0]), int(ep[1])),
                       max(1, round(GAZE_ENDPOINT_RADIUS * ds)),
                       COLOR_GAZE_ENDPOINT, -1)
        g_az, g_el = gaze_angles_3d(p.gaze_direction)
        gaze_label = f"gaze:{g_az:+.0f}/{g_el:+.0f}"

    kr = max(1, round(KEYPOINT_RADIUS * ds))
    for j in range(17):
        if vis[j]:
            col = (COLOR_KEYPOINT_NOSE if j == JOINT_NOSE else
                   COLOR_KEYPOINT_EAR  if j in (JOINT_LEFT_EAR, JOINT_RIGHT_EAR) else
                   COLOR_KEYPOINT_OTHER)
            cv2.circle(img, (px_x[j], px_y[j]), kr, col, -1)

    if not vis.any():
        return

    vi  = np.where(vis)[0]
    top = vi[px_y[vi].argmin()]
    g   = round(LABEL_LINE_SPACING_PX * ds)
    fs  = LABEL_FONT_SCALE_SMALL * ds
    ax  = int(px_x[top]) - round(30 * ds)
    ay  = int(px_y[top]) - round(12 * ds) - g
    conf = (float(p.joint_scores[p.joint_visible].mean())
            if p.joint_visible is not None and p.joint_visible.any() else 0.0)

    _sm     = lambda fld, val: _label_smoother.push(p.person_id, fld, val, timestamp_s)
    spd_raw = (math.sqrt(p.velocity_m_per_s.vx**2 + p.velocity_m_per_s.vy**2) * 3.6
               if p.velocity_m_per_s is not None else 0.0)
    sm_elev = _sm("elev", p.elevation_deg)
    sm_azim = _sm("azim", p.azimuth_deg)
    sm_h    = _sm("h_m",  p.height_m)
    sm_spd  = round(_sm("spd", spd_raw) * 2) / 2

    _text(img, f"#{p.person_id}  {conf:.0%}", (ax, ay),       fs, COLOR_LABEL_GAZE)
    _text(img, gaze_label,                    (ax, ay - g),   fs, COLOR_LABEL_GAZE)
    _text(img, f"ver:{sm_elev:+.1f}",         (ax, ay - 2*g), fs, COLOR_LABEL_ANGLES)
    _text(img, f"hor:{sm_azim:+.1f}",         (ax, ay - 3*g), fs, COLOR_LABEL_ANGLES)
    _text(img, f"spd:{sm_spd:.1f}km/h",       (ax, ay - 4*g), fs, COLOR_LABEL_ANGLES)
    _text(img, f"H:{sm_h*100:.0f}cm",         (ax, ay - 5*g), fs, COLOR_LABEL_DIST)


def draw_hud(img, n, t_s, total_s, rec, paused,
             cam_src, h, ds: float = 1.0) -> None:
    status = f"Persons:{n}  {t_s:.1f}s/{total_s:.1f}s"
    if rec:    status += "  [REC]"
    if paused: status += "  [PAUSED]"
    col  = COLOR_HUD_RECORDING if rec else COLOR_HUD_PAUSED if paused else COLOR_HUD_RUNNING
    row1 = max(20, round(30 * ds))
    row2 = max(36, round(56 * ds))
    _text(img, status,           (10, row1), HUD_FONT_SCALE_STATUS * ds, col)
    _text(img, f"CAM:{cam_src}", (10, row2), HUD_FONT_SCALE_CAM    * ds, COLOR_HUD_CAM_INFO)
    _text(img,
          f"[A]<<{SEEK_STEP_SECONDS}s  [D]>>{SEEK_STEP_SECONDS}s  "
          "[Space]Pause  [R]Rec+NDJSON  [Q]Quit",
          (10, h - max(8, round(12 * ds))),
          LABEL_FONT_SCALE_SMALL * ds, COLOR_HUD_HINT)


def _stop_rec(vw, nw, vp, np_) -> tuple:
    if vw is not None:
        vw.release()
        if vp is not None: print(f"Saved video  : {vp}")
    if nw is not None:
        nw.close()
        if np_ is not None: print(f"Saved NDJSON : {np_}")
    return None, None


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {VIDEO_PATH}")
    fw       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_s  = n_frames / fps
    print(f"Video  : {fw}x{fh}  {fps:.1f}fps  {total_s:.1f}s  ({n_frames} frames)")

    cam            = init_camera(fw, fh, fps)
    fx, fy, cx, cy = cam["FX"], cam["FY"], cam["CX"], cam["CY"]
    _ds_raw = math.sqrt(fw * fh / (1808.0 * 1392.0))
    ds      = 1.0 + (_ds_raw - 1.0) * 0.6
    scale   = min(DISPLAY_MAX_WIDTH_PX / fw, DISPLAY_MAX_HEIGHT_PX / fh, 1.0)
    dw, dh  = int(fw * scale), int(fh * scale)

    paused, rec = False, False
    vw = nw = vp = np_ = None
    cur_frame = fps_cnt = 0
    fps_t0    = time.time()
    last_canvas: Optional[np.ndarray] = None
    persons:     list = []

    cv2.namedWindow("RTMPose 3D", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RTMPose 3D", dw, dh)
    cv2.moveWindow("RTMPose 3D", 0, 0)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("End of video.")
                vw, nw = _stop_rec(vw, nw, vp, np_)
                paused = True; continue
            cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            t_s       = cur_frame / fps
            persons   = process_frame(frame, t_s)
            fps_cnt  += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                print(f"FPS:{fps_cnt/(now-fps_t0):.1f}  frame:{cur_frame}/{n_frames}"
                      f"  t:{t_s:.1f}s  persons:{len(persons)}  {DEVICE.upper()}")
                fps_t0, fps_cnt = now, 0

        if paused and last_canvas is not None:
            canvas = last_canvas.copy()
        elif paused:
            canvas = np.zeros((fh, fw, 3), dtype=np.uint8)
        else:
            canvas = frame.copy()
            if GROUND_PLANE_ENABLED:
                draw_ground_plane(canvas, get_ground_plane(), fx, fy, cx, cy, ds)
            for p in persons:
                draw_person(canvas, p, fx, fy, cx, cy, ds, t_s)
            last_canvas = canvas

        draw_hud(canvas, 0 if paused else len(persons),
                 cur_frame / fps, total_s, rec, paused, cam["source"], fh, ds)

        if rec:
            if vw is not None: vw.write(canvas)
            if nw is not None:
                nw.write(build_detection_message(persons, cur_frame/fps).to_json() + "\n")

        disp = cv2.resize(canvas, (dw, dh),
                          interpolation=cv2.INTER_AREA) if scale < 1.0 else canvas
        cv2.imshow("RTMPose 3D", disp)
        key = cv2.waitKey(100 if paused else 1) & 0xFF

        if key in (ord("q"), 27):
            vw, nw = _stop_rec(vw, nw, vp, np_); break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            if not rec:
                rec = True
                ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
                vp  = OUTPUT_DIR / f"rec_{ts}.mp4"
                np_ = OUTPUT_DIR / f"rec_{ts}.ndjson"
                vw  = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
                nw  = open(np_, "w", encoding="utf-8")
                print(f"Recording  : {vp}"); print(f"Messages   : {np_}")
            else:
                rec = False; vw, nw = _stop_rec(vw, nw, vp, np_)
        elif key == ord("a"):
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - SEEK_STEP_SECONDS * fps))
            paused = False
        elif key == ord("d"):
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    min(n_frames - 1,
                        cap.get(cv2.CAP_PROP_POS_FRAMES) + SEEK_STEP_SECONDS * fps))
            paused = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()