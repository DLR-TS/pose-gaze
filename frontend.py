"""frontend.py – video capture loop, overlay drawing, recording, keyboard control.

Keys:
    R       – toggle recording (MP4 + NDJSON)
    Space   – pause / resume
    A / D   – seek ±SEEK_STEP_SECONDS
    Q / Esc – quit
"""
import math, time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from basics import *  # noqa: F401,F403
import backend

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# grid x-positions for ground plane overlay; constant for given config
_GRID_X_VALUES = np.arange(
    -GROUND_PLANE_X_HALF_M,
     GROUND_PLANE_X_HALF_M + 1e-9,
     GROUND_PLANE_X_STEP_M,
)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _text(img: np.ndarray, text: str, pos: tuple,
          scale: float, color: tuple) -> None:
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), TEXT_OUTLINE_THICKNESS + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, TEXT_OUTLINE_THICKNESS)


def draw_ground_plane(
        img:         np.ndarray,
        gp:          Optional[GroundPlane],
        fx:          float,
        fy:          float,
        cx:          float,
        cy:          float,
        cam_height:  CamHeightEMA,
        ds:          float = 1.0,
        timestamp_s: float = 0.0,
) -> None:
    h_img, w_img = img.shape[:2]
    y_near = int(h_img * GROUND_PLANE_NEAR_FRAC)
    y_far  = int(h_img * GROUND_PLANE_FAR_FRAC)

    # cache plane coefficients once for both near/far row queries
    _gp_valid = gp is not None and abs(float(gp.normal[1])) >= GROUND_PLANE_NORMAL_MIN_Y
    if _gp_valid:
        _n  = gp.normal.astype(np.float64)
        _c  = gp.centroid.astype(np.float64)
        _d  = -float(np.dot(_n, _c))

    def _row_to_z(y_px: float) -> float:
        if _gp_valid:
            ray   = np.array([0., (y_px - cy) / fy, 1.0])
            denom = float(np.dot(_n, ray))
            if abs(denom) > 1e-6:
                t = -_d / denom
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

    poly    = np.array([[xl_n, y_near], [xr_n, y_near],
                        [xr_f, y_far],  [xl_f, y_far]], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], COLOR_GROUND_PLANE_FILL)
    cv2.addWeighted(overlay, GROUND_PLANE_FILL_ALPHA,
                    img, 1.0 - GROUND_PLANE_FILL_ALPHA, 0, img)

    # draw grid lines only when ground plane is fitted and reliable
    _gp_ok = _gp_valid and gp.inlier_ratio > 0.3
    if _gp_ok:
        lthick = max(1, round(GROUND_PLANE_LINE_THICKNESS * ds))
        for x_m in _GRID_X_VALUES:
            xn = max(0, min(w_img-1, _x_px(x_m, z_near)))
            xf = max(0, min(w_img-1, _x_px(x_m, z_far)))
            cv2.line(img, (xn, y_near), (xf, y_far),
                     COLOR_GROUND_LONG_LINES, lthick, cv2.LINE_AA)

    if _gp_valid:
        h_raw = abs(float(np.dot(_n, _c)))
        cam_height.update(h_raw, timestamp_s)
        if cam_height.value is not None:
            label = f"cam:{cam_height.value:.2f}m"
        else:
            label = "cam:---"
        fs    = LABEL_FONT_SCALE_SMALL * ds
        (tw, th), bl = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, fs, TEXT_OUTLINE_THICKNESS + 2)
        lx = max(2, min(w_img - tw - 2, int(cx) + 8))
        ly = max(th + 2, min(h_img - bl - 2, int(cy) - 8))
        _text(img, label, (lx, ly), fs, (0, 255, 255))


def draw_person(
        img: np.ndarray,
        p:   PersonData,
        fx:  float, fy: float, cx: float, cy: float,
        ds:  float = 1.0,
) -> None:
    vis = p.joint_visible
    if vis is None or not vis.any():
        return
    jm     = p.joints_meters
    safe_d = np.where(vis, jm[:, 2], 1.0)
    px_x   = ((jm[:, 0] / safe_d) * fx + cx).astype(np.int32)
    px_y   = ((jm[:, 1] / safe_d) * fy + cy).astype(np.int32)

    el = math.radians(p.elevation_deg)
    az = math.radians(p.azimuth_deg)
    hd = p.distance_m * math.cos(el)
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
            _text(img, f"{p.distance_m:.1f}m",
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

    vi  = np.where(vis)[0]
    top = vi[px_y[vi].argmin()]
    g   = round(LABEL_LINE_SPACING_PX * ds)
    fs  = LABEL_FONT_SCALE_SMALL * ds
    ax  = int(px_x[top]) - round(30 * ds)
    ay  = int(px_y[top]) - round(12 * ds) - g
    conf = (float(p.joint_scores[p.joint_visible].mean())
            if p.joint_visible is not None and p.joint_visible.any() else 0.0)

    spd_kmh = (round(math.sqrt(p.velocity_m_per_s.vx**2
                               + p.velocity_m_per_s.vy**2) * 3.6 * 2) / 2
               if p.velocity_m_per_s is not None else 0.0)

    _text(img, f"#{p.person_id} {conf:.0%}",    (ax, ay),       fs, COLOR_LABEL_GAZE)
    _text(img, gaze_label,                       (ax, ay - g),   fs, COLOR_LABEL_GAZE)
    _text(img, f"ver:{p.elevation_deg:+.1f}",   (ax, ay - 2*g), fs, COLOR_LABEL_ANGLES)
    _text(img, f"hor:{p.azimuth_deg:+.1f}",     (ax, ay - 3*g), fs, COLOR_LABEL_ANGLES)
    _text(img, f"spd:{spd_kmh:.1f}km/h",        (ax, ay - 4*g), fs, COLOR_LABEL_ANGLES)
    _text(img, f"H:{p.height_m*100:.0f}cm",     (ax, ay - 5*g), fs, COLOR_LABEL_DIST)


def draw_hud(img, n, t_s, total_s, rec, paused,
             cam_src, h, ds: float = 1.0) -> None:
    status = f"Persons:{n} {t_s:.1f}s/{total_s:.1f}s"
    if rec:    status += " [REC]"
    if paused: status += " [PAUSED]"
    col  = COLOR_HUD_RECORDING if rec else COLOR_HUD_PAUSED if paused else COLOR_HUD_RUNNING
    row1 = max(20, round(30 * ds))
    row2 = max(36, round(56 * ds))
    _text(img, status,           (10, row1), HUD_FONT_SCALE_STATUS * ds, col)
    _text(img, f"CAM:{cam_src}", (10, row2), HUD_FONT_SCALE_CAM    * ds, COLOR_HUD_CAM_INFO)
    _text(img,
          f"[A]<<{SEEK_STEP_SECONDS}s [D]>>{SEEK_STEP_SECONDS}s "
          "[Space]Pause [R]Rec+NDJSON [Q]Quit",
          (10, h - max(8, round(12 * ds))),
          LABEL_FONT_SCALE_SMALL * ds, COLOR_HUD_HINT)


def _stop_rec(vw, nw, vp, np_) -> tuple:
    if vw is not None: vw.release()
    if vp is not None: print(f"Saved video  : {vp}")
    if nw is not None:
        nw.close()
        if np_ is not None: print(f"Saved NDJSON : {np_}")
    return None, None, None
def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {VIDEO_PATH}")
    fw       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_s  = n_frames / fps
    print(f"Video : {fw}x{fh} {fps:.1f}fps {total_s:.1f}s ({n_frames} frames)")

    cam             = backend.init_camera(fw, fh, fps)
    fx, fy, cx, cy  = cam["FX"], cam["FY"], cam["CX"], cam["CY"]
    _ds_raw         = math.sqrt(fw * fh / (DISPLAY_DS_REFERENCE_PX[0] * DISPLAY_DS_REFERENCE_PX[1]))
    ds              = 1.0 + (_ds_raw - 1.0) * 0.6
    scale           = min(DISPLAY_MAX_WIDTH_PX / fw, DISPLAY_MAX_HEIGHT_PX / fh, 1.0)
    dw, dh          = int(fw * scale), int(fh * scale)

    cam_height      = CamHeightEMA()
    _frame_ms       = max(1, int(1000 / fps))
    paused, rec     = False, False
    vw = nw = vp = np_ = None
    cur_frame = fps_cnt = 0
    fps_t0              = time.time()
    last_canvas: Optional[np.ndarray] = None
    persons: list       = []

    cv2.namedWindow("RTMPose 3D", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RTMPose 3D", dw, dh)
    cv2.moveWindow("RTMPose 3D", 0, 0)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("End of video.")
                if rec:
                    rec = False
                    vw, nw, vp = _stop_rec(vw, nw, vp, np_)
                    np_ = None
                paused = True; continue
            cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            t_s       = cur_frame / fps
            _t0          = time.time()
            persons      = backend.process_frame(frame, t_s)
            inference_ms = max(1, int((time.time() - _t0) * 1000))
            fps_cnt     += 1
            now       = time.time()
            if now - fps_t0 >= 5.0:
                print(f"FPS:{fps_cnt/(now-fps_t0):.1f} frame:{cur_frame}/{n_frames}"
                      f" t:{t_s:.1f}s persons:{len(persons)} {DEVICE.upper()}")
                fps_t0, fps_cnt = now, 0

        if paused:
            # paused: display last rendered frame unchanged
            canvas = (last_canvas.copy()
                      if last_canvas is not None
                      else np.zeros((fh, fw, 3), dtype=np.uint8))
        else:
            canvas = frame.copy()
            if GROUND_PLANE_ENABLED:
                draw_ground_plane(canvas, backend.get_ground_plane(),
                                  fx, fy, cx, cy, cam_height, ds, t_s)
            for p in persons:
                draw_person(canvas, p, fx, fy, cx, cy, ds)
            last_canvas = canvas

        draw_hud(canvas, len(persons),
                 cur_frame / fps, total_s, rec, paused, cam["source"], fh, ds)

        if rec:
            if vw is not None: vw.write(canvas)
            if nw is not None:
                nw.write(build_detection_message(persons, cur_frame/fps).to_json() + "\n")

        disp = cv2.resize(canvas, (dw, dh),
                          interpolation=cv2.INTER_AREA) if scale < 1.0 else canvas
        cv2.imshow("RTMPose 3D", disp)
        _wait = max(1, _frame_ms - inference_ms) if not paused else 100
        key = cv2.waitKey(_wait) & 0xFF

        if key in (ord("q"), 27):
            vw, nw, vp = _stop_rec(vw, nw, vp, np_)
            np_ = None; break
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
                print(f"Recording  : {vp}")
                print(f"Messages   : {np_}")
            else:
                rec = False
                vw, nw, vp = _stop_rec(vw, nw, vp, np_)
                np_ = None
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
