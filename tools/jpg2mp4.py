#!/usr/bin/env python3
"""
jpg2mp4_calibrated.py
=====================
Converts a JPEG image sequence to a cropped/scaled MP4 and writes a matching
camera-parameter JSON file consumed by rtmpose_level_3d.py.

Output JSON naming convention:
  camera_{W}x{H}_{model_id}_crop{x0}-{y0}-{x1}-{y1}.json

rtmpose_level_3d.py discovers the correct JSON automatically by matching
the measured video resolution against JSON files in CAMERA_JSON_DIR.
"""

import cv2, json, numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

SCRIPT_DIR = Path(__file__).parent.resolve()  # = xxx/tools
PROJ_DIR   = SCRIPT_DIR.parent                 # = xxx/

# ── Configuration ────────────────────────────────────────────────────────────
IMAGE_FOLDER  = r"D:\\videos\\images_ros"
FILENAME_TMPL = "left{:04d}.jpg"
FRAME_START   = 0
FRAME_END     = 469

#: Path to the original (uncropped) camera-calibration JSON.
ORIGINAL_CAMERA_JSON = PROJ_DIR / "media" / "camera_4112x2176_mvBlueCOUGAR-X109b.json"

#: Output directory for the video, JSON, and preview.
OUTPUT_DIR = PROJ_DIR / "media"  # generated files (video, JSON, preview) go here

#: Output video FPS.  None = inherit from original JSON.
VIDEO_FPS_OVERRIDE: Optional[float] = 8.0  # override original FPS (e.g. to slow down fast-motion footage for better visualization)

#: Apply a sharpening kernel to each frame.
SHARPEN = True

#: Crop region as (x_start%, y_start%, x_end%, y_end%) in [0, 100].
CROP_PERCENT: tuple = (5, 18, 49, 82)
#CROP_PERCENT: tuple = (0, 0, 100, 100)

#: Output resolution (width, height) or None for exact crop dimensions.
TARGET_SIZE: Optional[tuple] = None

#: Undistort frames before cropping (requires non-zero D in JSON).
UNDISTORT_BEFORE_CROP = False


# ── Camera JSON helpers ───────────────────────────────────────────────────────
def load_original_json(path: Path) -> dict:
    """Load the original camera-calibration JSON.

    Returns a flat dict: K, D, FX, FY, CX, CY, W, H, FPS,
    pixel_um, lens_mm, meta, model_id.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Original camera JSON not found: {path}. "
            "Create it with generate_original_camera_json.py or place manually."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    ci, meta = data["camera_intrinsics"], data["camera_meta"]
    return {
        "K":        np.array(ci["K"], dtype=np.float64),
        "D":        np.array(ci["D"], dtype=np.float64),
        "FX":       float(ci["FX"]),  "FY": float(ci["FY"]),
        "CX":       float(ci["CX"]),  "CY": float(ci["CY"]),
        "W":        int(ci["WIDTH"]), "H":  int(ci["HEIGHT"]),
        "FPS":      float(ci["FPS"]),
        "pixel_um": float(meta.get("pixel_size_um", 0.0)),
        "lens_mm":  float(meta.get("lens_mm", 0.0)),
        "meta":     meta,
        "model_id": meta["model_id"],
    }


def _build_crop_meta(orig: dict, info: dict, fps: float) -> dict:
    """Derive physical camera parameters for the cropped sensor area.

    Physical focal length is invariant under cropping and scaling:
    lens_mm = FX * pix_um / 1000 = FX_prime * (pix_um/sx) / 1000 = const.
    """
    pix = orig["pixel_um"];  FX_o = orig["FX"]
    cw, ch = info["crop_w"], info["crop_h"]
    sx, sy = info["sx"],     info["sy"]
    m = dict(orig["meta"])
    m["sensor_size_mm"]       = [round(cw*pix/1000, 4), round(ch*pix/1000, 4)]
    m["sensor_size_px"]       = [cw, ch]
    m["output_size_px"]       = [info["tw"], info["th"]]
    m["lens_mm"]              = orig["lens_mm"]
    m["pixel_size_um"]        = pix
    m["output_pixel_size_um"] = ([round(pix/sx,4), round(pix/sy,4)]
                                  if abs(sx-1.)>1e-4 or abs(sy-1.)>1e-4
                                  else [pix, pix])
    m["fov_h_deg"] = round(2*np.degrees(np.arctan(cw/2/FX_o)), 3)
    m["fov_v_deg"] = round(2*np.degrees(np.arctan(ch/2/FX_o)), 3)
    m["native_fps"] = fps
    return m


# ── Crop / K-prime ───────────────────────────────────────────────────────────
def compute_k_prime(K, orig_w, orig_h, crop_pct, target_size) -> tuple:
    """Compute K_prime after a percentage crop and optional scaling.

    Transformation: cx_prime = (cx - x0)*sx,  fx_prime = fx*sx.
    cx_prime may lie outside [0, tw]; this is geometrically correct.

    Returns:
        (K_prime np.ndarray(3,3), info dict)
    """
    cp = tuple(float(np.clip(v, 0., 100.)) for v in crop_pct)
    if cp[0] >= cp[2] or cp[1] >= cp[3]:
        raise ValueError(f"Invalid crop_pct {crop_pct}.")
    x0 = int(cp[0]/100*orig_w);  y0 = int(cp[1]/100*orig_h)
    x1 = int(cp[2]/100*orig_w);  y1 = int(cp[3]/100*orig_h)
    cwr = x1-x0;  chr_ = y1-y0
    crop_w = (cwr//2)*2;  crop_h = (chr_//2)*2
    if crop_w < 2: raise ValueError(f"crop_w={crop_w}px < 2.")
    if crop_h < 2: raise ValueError(f"crop_h={crop_h}px < 2.")
    x1 = x0+crop_w;  y1 = y0+crop_h
    if target_size is not None:
        tw = (int(target_size[0])//2)*2;  th = (int(target_size[1])//2)*2
        if tw < 2 or th < 2: raise ValueError(f"TARGET_SIZE {target_size} < 2.")
    else:
        tw, th = crop_w, crop_h
    sx = tw/crop_w;  sy = th/crop_h
    cx_n = (K[0,2]-x0)*sx;  cy_n = (K[1,2]-y0)*sy
    Kp = np.array([[K[0,0]*sx, 0., cx_n],[0., K[1,1]*sy, cy_n],[0.,0.,1.]])
    return Kp, dict(x0=x0, y0=y0, x1=x1, y1=y1,
                    crop_w=crop_w, crop_h=crop_h, tw=tw, th=th,
                    sx=sx, sy=sy,
                    anamorphic_warning=abs(sx-sy)>1e-4,
                    odd_clip_x=cwr%2!=0, odd_clip_y=chr_%2!=0)


def build_json(Kp, D, info, fps, crop_pct, orig, orig_path) -> dict:
    """Assemble the self-contained output JSON for the cropped camera."""
    return {
        "_generated":     datetime.now().isoformat(timespec="seconds"),
        "_tool":          "jpg2mp4_calibrated.py",
        "_original_json": str(orig_path),
        "is_original":    False,
        "camera_meta":    _build_crop_meta(orig, info, fps),
        "crop": {
            "percent": list(crop_pct),
            "x0_px": info["x0"], "y0_px": info["y0"],
            "x1_px": info["x1"], "y1_px": info["y1"],
            "crop_w": info["crop_w"], "crop_h": info["crop_h"],
        },
        "output_video": {
            "width": info["tw"], "height": info["th"], "fps": fps,
            "scale_x": info["sx"], "scale_y": info["sy"],
            "anamorphic_warning": info["anamorphic_warning"],
        },
        "warnings": {
            "anamorphic": info["anamorphic_warning"],
            "odd_clip_x": info["odd_clip_x"],
            "odd_clip_y": info["odd_clip_y"],
        },
        "camera_intrinsics": {
            "K": Kp.tolist(), "D": D.tolist(),
            "FX": float(Kp[0,0]), "FY": float(Kp[1,1]),
            "CX": float(Kp[0,2]), "CY": float(Kp[1,2]),
            "WIDTH": info["tw"], "HEIGHT": info["th"], "FPS": fps,
        },
    }


# ── Preview rendering ─────────────────────────────────────────────────────────
def _t(img, txt, pos, sc, col, th=2):
    """Draw text with a black outline for readability on any background."""
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, (0,0,0), th+2)
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th)


def _kmat(img, K, header, col, bx, by, sc=1.3, lh=60):
    """Render a 3x3 camera matrix as formatted text at position (bx, by)."""
    _t(img, header, (bx, by), sc, col, th=3)
    for r in range(3):
        row = "[" + "  ".join(f"{K[r,c]:9.2f}" for c in range(3)) + " ]"
        _t(img, row, (bx, by + lh*(r+1)), sc*0.95, col, th=2)


def _col(img, rows, bx, by, col, sc=1.3, lh=56):
    """Render a vertical list of strings at (bx, by) in given color."""
    for k, txt in enumerate(rows):
        _t(img, txt, (bx, by + lh*k), sc, col, th=2)


def render_preview(image, K_ORIG, Kp, info, orig, tw, th, fps):
    """Return an annotated preview image showing crop geometry and K matrices.

    - Orange: original camera (K_orig, arrows from image origin, left-side matrix)
    - Green:  cropped camera  (K_crop, arrows from crop origin, right-side matrix)
    - Both K matrices are placed left/right of the optical centre, offset downward.
    """
    prev = image.copy()
    ih, iw = prev.shape[:2]
    C_O = (0, 140, 255)    # orange
    C_C = (0, 210, 60)     # green
    C_W = (230, 230, 230)  # white

    cx_o, cy_o = int(K_ORIG[0,2]), int(K_ORIG[1,2])
    cx_p, cy_p = Kp[0,2], Kp[1,2]

    # Crop rectangle
    cv2.rectangle(prev, (info["x0"],info["y0"]), (info["x1"],info["y1"]), C_C, 5)
    sz = (f"{info['crop_w']}x{info['crop_h']}" if tw==info["crop_w"] and th==info["crop_h"]
          else f"{info['crop_w']}x{info['crop_h']}  ->  {tw}x{th}")
    _t(prev, f"Crop {list(CROP_PERCENT)}%  |  {sz}",
       (info["x0"]+14, info["y0"]+60), 1.6, C_C)

    # Crop-origin white cross
    ref = 22
    cv2.line(prev,(info["x0"]-ref,info["y0"]),(info["x0"]+ref,info["y0"]),C_W,2)
    cv2.line(prev,(info["x0"],info["y0"]-ref),(info["x0"],info["y0"]+ref),C_W,2)
    _t(prev, f"({info['x0']},{info['y0']})",(info["x0"]+14,info["y0"]-10),1.1,C_W,th=1)

    # Arrow: image origin -> optical centre (orange)
    cv2.arrowedLine(prev,(0,0),(cx_o,cy_o),C_O,3,cv2.LINE_AA,tipLength=0.03)
    _t(prev, f"cx={cx_o}  cy={cy_o}", (cx_o//4, cy_o//3), 1.4, C_O)

    # Arrow: crop origin -> optical centre (green)
    cv2.arrowedLine(prev,(info["x0"],info["y0"]),(cx_o,cy_o),C_C,3,cv2.LINE_AA,tipLength=0.04)
    mx=(info["x0"]+cx_o)//2;  my=(info["y0"]+cy_o)//2
    _t(prev, f"cx'={cx_p:.1f}  cy'={cy_p:.1f}", (mx+14,my-14), 1.4, C_C)

    # Optical-centre crosshair (orange, single physical point)
    arm=50
    cv2.line(prev,(cx_o-arm,cy_o),(cx_o+arm,cy_o),C_O,4)
    cv2.line(prev,(cx_o,cy_o-arm),(cx_o,cy_o+arm),C_O,4)
    cv2.circle(prev,(cx_o,cy_o),16,C_O,-1)

    # K_orig: left of optical centre, offset down
    off = 90
    _kmat(prev, K_ORIG, "K_orig:", C_O, max(20, cx_o-860), cy_o+off)
    # K_crop: right of optical centre, same offset
    _kmat(prev, Kp,     "K_crop:", C_C, cx_o+80,            cy_o+off)

    # Parameter columns top-right
    pix_o  = orig.get("pixel_um", 0.); lens_o = orig.get("lens_mm", 0.)
    sm_o   = orig["meta"].get("sensor_size_mm",[0,0])
    cm     = _build_crop_meta(orig, info, fps)
    cx1 = iw-1620;  cx2 = iw-800;  cyt = 70;  lh = 58
    _t(prev,"--- ORIGINAL ---",(cx1,cyt),1.5,C_O)
    _t(prev,"--- CROP ---",    (cx2,cyt),1.5,C_C)
    _col(prev,[
        f"lens:  {lens_o:.4g} mm",
        f"FX/FY: {K_ORIG[0,0]:.1f} / {K_ORIG[1,1]:.1f}",
        f"CX/CY: {K_ORIG[0,2]:.1f} / {K_ORIG[1,2]:.1f}",
        f"WxH:   {orig['W']}x{orig['H']} px",
        f"pix:   {pix_o:.2f} um",
        f"sens:  {sm_o[0]:.3f}x{sm_o[1]:.3f} mm",
        f"FOV:   {orig['meta'].get('fov_h_deg',0):.1f}x{orig['meta'].get('fov_v_deg',0):.1f} deg",
        f"FPS:   {orig['FPS']:.4g}",
    ], cx1, cyt+lh, C_O, sc=1.3, lh=lh)
    _col(prev,[
        f"lens:  {lens_o:.4g} mm  (same)",
        f"FX/FY: {Kp[0,0]:.1f} / {Kp[1,1]:.1f}",
        f"CX/CY: {Kp[0,2]:.1f} / {Kp[1,2]:.1f}",
        f"WxH:   {tw}x{th} px",
        f"pix:   {cm['output_pixel_size_um'][0]:.2f} um",
        f"sens:  {cm['sensor_size_mm'][0]:.3f}x{cm['sensor_size_mm'][1]:.3f} mm",
        f"FOV:   {cm['fov_h_deg']:.1f}x{cm['fov_v_deg']:.1f} deg",
        f"FPS:   {fps:.4g}",
    ], cx2, cyt+lh, C_C, sc=1.3, lh=lh)
    return prev


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """Run the full pipeline: load frames, crop, encode video, write JSON."""
    img_dir = Path(IMAGE_FOLDER)
    orig    = load_original_json(ORIGINAL_CAMERA_JSON)
    K_ORIG  = orig["K"];  D_ORIG = orig["D"]
    fps     = VIDEO_FPS_OVERRIDE if VIDEO_FPS_OVERRIDE is not None else orig["FPS"]
    print(f"Camera      : {orig['meta']['model']}")
    print(f"FPS         : {fps:.4g}  ('override' if VIDEO_FPS_OVERRIDE else 'from JSON')")

    first = cv2.imread(str(img_dir / FILENAME_TMPL.format(FRAME_START)))
    if first is None:
        raise FileNotFoundError(f"Frame not found: {FILENAME_TMPL.format(FRAME_START)}")
    orig_h, orig_w = first.shape[:2]
    print(f"Images      : {orig_w}x{orig_h}  [{FRAME_START}..{FRAME_END}]")

    Kp, info = compute_k_prime(K_ORIG, orig_w, orig_h, CROP_PERCENT, TARGET_SIZE)
    tw, th   = info["tw"], info["th"]
    print(f"Crop        : {CROP_PERCENT}  ->  ({info['x0']},{info['y0']})-({info['x1']},{info['y1']})")
    print(f"Output      : {tw}x{th}  sx={info['sx']:.4f}  sy={info['sy']:.4f}")
    if info["anamorphic_warning"]: print("WARNING: anamorphic scale (sx != sy).")
    if info["odd_clip_x"]:         print("INFO: 1px clipped right (H.264 rounding).")
    if info["odd_clip_y"]:         print("INFO: 1px clipped bottom (H.264 rounding).")

    stem      = (f"camera_{tw}x{th}_{orig['model_id']}"
                 f"_crop{info['x0']}-{info['y0']}-{info['x1']}-{info['y1']}")
    out_json  = OUTPUT_DIR / f"{stem}.json"
    out_video = OUTPUT_DIR / f"video_{tw}x{th}_{orig['model_id']}_crop{info['x0']}-{info['y0']}-{info['x1']}-{info['y1']}.mp4"
    out_prev  = OUTPUT_DIR / f"preview_{stem}.jpg"

    cv2.imwrite(str(out_prev), render_preview(first, K_ORIG, Kp, info, orig, tw, th, fps))
    print(f"Preview     : {out_prev.name}")

    undist_map1 = undist_map2 = None
    if UNDISTORT_BEFORE_CROP and not np.allclose(D_ORIG, 0):
        undist_map1, undist_map2 = cv2.initUndistortRectifyMap(
            K_ORIG, D_ORIG, None, K_ORIG, (orig_w, orig_h), cv2.CV_32FC1)

    sharpen_k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    video = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (tw,th))
    if not video.isOpened(): raise RuntimeError(f"VideoWriter failed: {out_video}")

    x0,y0,x1,y1 = info["x0"],info["y0"],info["x1"],info["y1"]
    n = FRAME_END - FRAME_START + 1;  nw = 0
    for i in range(FRAME_START, FRAME_END+1):
        frame = cv2.imread(str(img_dir / FILENAME_TMPL.format(i)))
        if frame is None: print(f"  WARNING: {FILENAME_TMPL.format(i)} not found."); continue
        if undist_map1 is not None:
            frame = cv2.remap(frame, undist_map1, undist_map2, cv2.INTER_LINEAR)
        crop = frame[y0:y1, x0:x1]
        if crop.shape[1]!=tw or crop.shape[0]!=th:
            crop = cv2.resize(crop,(tw,th), interpolation=cv2.INTER_AREA if tw<info["crop_w"] else cv2.INTER_CUBIC)
        video.write(cv2.filter2D(crop,-1,sharpen_k) if SHARPEN else crop)
        nw+=1
        if i%50==0: print(f"  {i-FRAME_START+1}/{n} frames written.")
    video.release()
    print(f"Video       : {out_video.name}  ({nw} frames @ {fps}fps)")

    with open(out_json,"w",encoding="utf-8") as f:
        json.dump(build_json(Kp,D_ORIG,info,fps,CROP_PERCENT,orig,ORIGINAL_CAMERA_JSON),f,indent=2,ensure_ascii=False)
    print(f"Camera JSON : {out_json.name}")
    print(f"rtmpose auto-selects this JSON for any {tw}x{th} video.")


if __name__ == "__main__":
    main()
