"""frontend.py – video capture, drawing, recording, keyboard control.

    [Space] Pause/resume   [R] Record   [A]/[D] Seek   [Q]/Esc Quit
"""
import time, cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import (
    VIDEO_PATH, OUTPUT_DIR, FPS_FALLBACK, SEEK_STEP_SECONDS,
    DISPLAY_MAX_WIDTH_PX, DISPLAY_MAX_HEIGHT_PX, GAZE_RAY_LENGTH_M,
    JOINT_NOSE, JOINT_LEFT_EAR, JOINT_RIGHT_EAR, MIN_KEYPOINT_CONFIDENCE,
    SKELETON_EDGES, SKELETON_EDGE_COLORS,
    COLOR_KEYPOINT_NOSE, COLOR_KEYPOINT_EAR, COLOR_KEYPOINT_OTHER, COLOR_EAR_CONNECTOR,
    COLOR_GAZE_ORIGIN_DOT, COLOR_GAZE_LINE, COLOR_GAZE_ENDPOINT,
    COLOR_LABEL_SIZE, COLOR_LABEL_ANGLES, COLOR_LABEL_ID,
    COLOR_HUD_RECORDING, COLOR_HUD_PAUSED, COLOR_HUD_RUNNING,
    COLOR_HUD_CAM_INFO, COLOR_HUD_HINT,
    KEYPOINT_RADIUS, GAZE_ORIGIN_DOT_RADIUS, GAZE_ENDPOINT_RADIUS,
    SKELETON_LINE_THICKNESS, GAZE_LINE_THICKNESS, TEXT_OUTLINE_THICKNESS,
    LABEL_ANCHOR_OFFSET_X, LABEL_ANCHOR_OFFSET_Y, LABEL_LINE_SPACING_PX,
    LABEL_FONT_SCALE_LARGE, LABEL_FONT_SCALE_SMALL,
    HUD_FONT_SCALE_STATUS, HUD_FONT_SCALE_CAM,
)
from backend import (
    DEVICE, init_camera, process_frame,
    PersonData, proj3d2d, gaze_angles_3d,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _draw_text(image, text: str, position: tuple,
               font_scale: float, color: tuple,
               thickness: int = TEXT_OUTLINE_THICKNESS) -> None:
    """Text with dark outline for readability on any background."""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)


def draw_person(image, person: PersonData,
                focal_x: float, focal_y: float,
                principal_x: float, principal_y: float) -> None:
    """Skeleton, keypoints, gaze ray, and metric labels from person.joints_meters."""
    joint_positions_m = person.joints_meters
    joint_scores      = person.joint_scores

    # Batch-project all 17 joints; invisible or behind-camera joints are masked
    depth_per_joint = joint_positions_m[:, 2]
    joint_visible   = (joint_scores > MIN_KEYPOINT_CONFIDENCE) & (depth_per_joint > 0)
    safe_depth      = np.where(joint_visible, depth_per_joint, 1.)
    pixel_x = ((joint_positions_m[:, 0] / safe_depth) * focal_x + principal_x).astype(np.int32)
    pixel_y = ((joint_positions_m[:, 1] / safe_depth) * focal_y + principal_y).astype(np.int32)

    for edge_idx, (joint_a_1based, joint_b_1based) in enumerate(SKELETON_EDGES):
        idx_a, idx_b = joint_a_1based - 1, joint_b_1based - 1
        if joint_visible[idx_a] and joint_visible[idx_b]:
            cv2.line(image,
                     (pixel_x[idx_a], pixel_y[idx_a]),
                     (pixel_x[idx_b], pixel_y[idx_b]),
                     SKELETON_EDGE_COLORS[edge_idx % len(SKELETON_EDGE_COLORS)],
                     SKELETON_LINE_THICKNESS, cv2.LINE_AA)

    # Gaze drawn before keypoint circles so the red nose dot renders on top
    gaze_angle_label = "gaze:n/a"
    if person.gaze_origin_m is not None:
        if joint_visible[JOINT_LEFT_EAR] and joint_visible[JOINT_RIGHT_EAR]:
            cv2.line(image,
                     (pixel_x[JOINT_LEFT_EAR],  pixel_y[JOINT_LEFT_EAR]),
                     (pixel_x[JOINT_RIGHT_EAR], pixel_y[JOINT_RIGHT_EAR]),
                     COLOR_EAR_CONNECTOR, SKELETON_LINE_THICKNESS, cv2.LINE_AA)
        gaze_endpoint_3d = person.gaze_origin_m + person.gaze_direction * GAZE_RAY_LENGTH_M
        origin_pixel     = proj3d2d(person.gaze_origin_m,
                                    focal_x, focal_y, principal_x, principal_y)
        endpoint_pixel   = proj3d2d(gaze_endpoint_3d,
                                    focal_x, focal_y, principal_x, principal_y)
        if origin_pixel is not None and endpoint_pixel is not None:
            cv2.circle(image,
                       (int(origin_pixel[0]), int(origin_pixel[1])),
                       GAZE_ORIGIN_DOT_RADIUS, COLOR_GAZE_ORIGIN_DOT, -1)
            cv2.line(image,
                     (int(origin_pixel[0]),   int(origin_pixel[1])),
                     (int(endpoint_pixel[0]), int(endpoint_pixel[1])),
                     COLOR_GAZE_LINE, GAZE_LINE_THICKNESS, cv2.LINE_AA)
            cv2.circle(image,
                       (int(endpoint_pixel[0]), int(endpoint_pixel[1])),
                       GAZE_ENDPOINT_RADIUS, COLOR_GAZE_ENDPOINT, -1)
            gaze_azimuth, gaze_elevation = gaze_angles_3d(person.gaze_direction)
            gaze_angle_label = f"g:{gaze_azimuth:+.0f}/{gaze_elevation:+.0f}"

    # Keypoint circles drawn last so nose (red) appears on top of gaze origin dot
    for joint_idx in range(17):
        if joint_visible[joint_idx]:
            color = (COLOR_KEYPOINT_NOSE if joint_idx == JOINT_NOSE else
                     COLOR_KEYPOINT_EAR  if joint_idx in (JOINT_LEFT_EAR, JOINT_RIGHT_EAR) else
                     COLOR_KEYPOINT_OTHER)
            cv2.circle(image,
                       (pixel_x[joint_idx], pixel_y[joint_idx]),
                       KEYPOINT_RADIUS, color, -1)

    if not joint_visible.any(): return
    visible_indices = np.where(joint_visible)[0]
    topmost_idx     = visible_indices[pixel_y[visible_indices].argmin()]
    anchor_x = int(pixel_x[topmost_idx]) + LABEL_ANCHOR_OFFSET_X
    anchor_y = int(pixel_y[topmost_idx]) + LABEL_ANCHOR_OFFSET_Y
    gap      = LABEL_LINE_SPACING_PX
    _draw_text(image, f"H:{person.height_m*100:.0f}cm",
               (anchor_x, anchor_y),           LABEL_FONT_SCALE_LARGE, COLOR_LABEL_SIZE)
    _draw_text(image, f"r:{person.distance_m:.1f}m",
               (anchor_x, anchor_y + gap),      LABEL_FONT_SCALE_LARGE, COLOR_LABEL_SIZE)
    _draw_text(image, f"v:{person.elevation_deg:+.1f} az:{person.azimuth_deg:+.1f}",
               (anchor_x - 20, anchor_y + 2*gap), LABEL_FONT_SCALE_SMALL, COLOR_LABEL_ANGLES)
    _draw_text(image, gaze_angle_label,
               (anchor_x - 20, anchor_y + 3*gap), LABEL_FONT_SCALE_SMALL, COLOR_LABEL_ANGLES)
    _draw_text(image, f"#{person.person_id}",
               (anchor_x, anchor_y + 4*gap),   LABEL_FONT_SCALE_SMALL, COLOR_LABEL_ID)


def draw_hud(image, num_persons: int, current_time_s: float, total_time_s: float,
             is_recording: bool, is_paused: bool, cam_source: str, frame_height: int) -> None:
    """Status bar and keyboard hint overlay."""
    status_text = f"Persons:{num_persons}   {current_time_s:.1f}s / {total_time_s:.1f}s"
    if is_recording: status_text += "   [REC]"
    if is_paused:    status_text += "   [PAUSED]"
    status_color = (COLOR_HUD_RECORDING if is_recording else
                    COLOR_HUD_PAUSED    if is_paused    else
                    COLOR_HUD_RUNNING)
    _draw_text(image, status_text,         (10, 30), HUD_FONT_SCALE_STATUS, status_color)
    _draw_text(image, f"CAM:{cam_source}", (10, 56), HUD_FONT_SCALE_CAM,    COLOR_HUD_CAM_INFO)
    _draw_text(image,
               f"[A]<<{SEEK_STEP_SECONDS}s  [D]>>{SEEK_STEP_SECONDS}s"
               "  [Space]Pause  [R]Rec  [Q]Quit",
               (10, frame_height - 12), LABEL_FONT_SCALE_SMALL, COLOR_HUD_HINT)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main() -> None:
    video_capture = cv2.VideoCapture(VIDEO_PATH)
    if not video_capture.isOpened():
        raise RuntimeError(f"Cannot open: {VIDEO_PATH}")

    frame_width   = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height  = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps     = video_capture.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    total_frames  = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = total_frames / video_fps
    print(f"Video   : {frame_width}x{frame_height}  {video_fps:.1f}fps"
          f"  {total_seconds:.1f}s  ({total_frames} fr)")

    camera_params = init_camera(frame_width, frame_height, video_fps)
    focal_x     = camera_params["FX"]
    focal_y     = camera_params["FY"]
    principal_x = camera_params["CX"]
    principal_y = camera_params["CY"]

    display_scale  = min(DISPLAY_MAX_WIDTH_PX / frame_width,
                         DISPLAY_MAX_HEIGHT_PX / frame_height, 1.)
    display_width  = int(frame_width  * display_scale)
    display_height = int(frame_height * display_scale)

    is_paused        = False
    is_recording     = False
    video_writer:    Optional[cv2.VideoWriter] = None
    recording_path:  Optional[Path]            = None
    current_frame    = 0
    fps_frame_count  = 0
    fps_timer_start  = time.time()
    last_canvas:     Optional[np.ndarray]      = None
    detected_persons = []

    cv2.namedWindow("RTMPose 3D", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RTMPose 3D", display_width, display_height)
    cv2.moveWindow("RTMPose 3D", 0, 0)

    while True:
        if not is_paused:
            frame_ok, bgr_frame = video_capture.read()
            if not frame_ok:
                print("End of video.")
                if video_writer: video_writer.release(); print(f"Saved: {recording_path}")
                is_paused = True; continue
            current_frame    = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            detected_persons = process_frame(bgr_frame)
            fps_frame_count += 1
            now = time.time()
            if now - fps_timer_start >= 1.:
                print(f"FPS:{fps_frame_count/(now-fps_timer_start):.1f}  "
                      f"frame:{current_frame}/{total_frames}  "
                      f"t:{current_frame/video_fps:.1f}s  "
                      f"persons:{len(detected_persons)}  {DEVICE.upper()}")
                fps_timer_start, fps_frame_count = now, 0

        if   is_paused and last_canvas is not None: canvas = last_canvas.copy()
        elif is_paused:                             canvas = np.zeros((frame_height, frame_width, 3), np.uint8)
        else:
            canvas = bgr_frame.copy()
            for person in detected_persons:
                draw_person(canvas, person, focal_x, focal_y, principal_x, principal_y)
            last_canvas = canvas

        draw_hud(canvas, len(detected_persons) if not is_paused else 0,
                 current_frame / video_fps, total_seconds,
                 is_recording, is_paused, camera_params["source"], frame_height)
        if is_recording and video_writer:
            video_writer.write(canvas)

        display_frame = (cv2.resize(canvas, (display_width, display_height),
                                    interpolation=cv2.INTER_AREA)
                         if display_scale < 1. else canvas)
        cv2.imshow("RTMPose 3D", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            if video_writer: video_writer.release(); print(f"Saved: {recording_path}")
            break
        elif key == ord(" "): is_paused = not is_paused
        elif key == ord("r"):
            if not is_recording:
                is_recording   = True
                timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
                recording_path = OUTPUT_DIR / f"rec_{timestamp}.mp4"
                video_writer   = cv2.VideoWriter(
                    str(recording_path), cv2.VideoWriter_fourcc(*"mp4v"),
                    video_fps, (frame_width, frame_height))
                print(f"Recording: {recording_path}")
            else:
                is_recording = False
                if video_writer: video_writer.release(); print(f"Saved: {recording_path}")
                video_writer = None
        elif key == ord("a"):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES,
                max(0, video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                    - SEEK_STEP_SECONDS * video_fps))
            is_paused = False
        elif key == ord("d"):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES,
                min(total_frames - 1,
                    video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                    + SEEK_STEP_SECONDS * video_fps))
            is_paused = False

    if video_writer: video_writer.release()
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
