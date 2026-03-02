from ultralytics import YOLO
import cv2
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Config
VIDEO_PATH = "media/test0_cut.mp4"
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("recordings")
SKIP_SECONDS = 5
KPT_THRESH = 0.25
KPT_THRESH_GAZE = 0.05  # Sehr niedriger Threshold nur für Gaze (nutzt YOLO's Schätzung)
CONF_THRESH = 0.25

# Smoothing / Temporal Filter
SMOOTH_ALPHA_POS_MOVING = 0.7
SMOOTH_ALPHA_POS_STATIC = 0.4
SMOOTH_ALPHA_SCORE = 0.5
SMOOTH_MOVEMENT_HISTORY = 5
SMOOTH_SCORE_HISTORY = 2

# Gaze Ray
GAZE_LENGTH = 270
GAZE_ANGLE_CORRECTION = -5

# Display Settings
DISPLAY_MAX_WIDTH = 1600
DISPLAY_MAX_HEIGHT = 900
DISPLAY_KEEP_ASPECT = True

# Model setup
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "yolo11x-pose.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = YOLO(str(MODEL_PATH))
model.to(device)
print(f"Models: {MODEL_PATH}")

# Smoother with memory for missing points
class Smoother:
    def __init__(self):
        self.prev_kpts = {}
        self.prev_scrs = {}
        self.scr_hist = {}
        self.mov_hist = {}
    
    def smooth(self, pid, kpts, scrs):
        if pid not in self.prev_kpts:
            mov = 1.0
        else:
            diff = np.linalg.norm(kpts - self.prev_kpts[pid])
            mov = min(1.0, diff / 50.0)
        
        if pid not in self.mov_hist:
            self.mov_hist[pid] = [mov]
        else:
            self.mov_hist[pid].append(mov)
            if len(self.mov_hist[pid]) > SMOOTH_MOVEMENT_HISTORY:
                self.mov_hist[pid].pop(0)
        
        avg_mov = np.mean(self.mov_hist[pid])
        alpha_pos = SMOOTH_ALPHA_POS_STATIC + (SMOOTH_ALPHA_POS_MOVING - SMOOTH_ALPHA_POS_STATIC) * avg_mov
        
        if pid not in self.prev_kpts:
            self.prev_kpts[pid] = kpts.copy()
            self.prev_scrs[pid] = scrs.copy()
            self.scr_hist[pid] = [scrs.copy()]
            return kpts, scrs
        
        smooth_kpts = alpha_pos * kpts + (1 - alpha_pos) * self.prev_kpts[pid]
        smooth_scrs = SMOOTH_ALPHA_SCORE * scrs + (1 - SMOOTH_ALPHA_SCORE) * self.prev_scrs[pid]
        
        self.scr_hist[pid].append(smooth_scrs.copy())
        if len(self.scr_hist[pid]) > SMOOTH_SCORE_HISTORY:
            self.scr_hist[pid].pop(0)
        
        if len(self.scr_hist[pid]) >= 2:
            stable = np.maximum(self.scr_hist[pid][-2], self.scr_hist[pid][-1])
            smooth_scrs = 0.7 * smooth_scrs + 0.3 * stable
        
        self.prev_kpts[pid] = smooth_kpts
        self.prev_scrs[pid] = smooth_scrs
        return smooth_kpts, smooth_scrs
    
    def get_last_valid(self, pid, idx):
        """Get last valid keypoint for specific index."""
        if pid in self.prev_kpts:
            return self.prev_kpts[pid][idx]
        return None
    
    def cleanup(self, active_ids):
        for pid in list(self.prev_kpts.keys()):
            if pid not in active_ids:
                del self.prev_kpts[pid]
                del self.prev_scrs[pid]
                if pid in self.scr_hist: del self.scr_hist[pid]
                if pid in self.mov_hist: del self.mov_hist[pid]

smoother = Smoother()

# Gaze smoother - separate for gaze stability
class GazeSmoother:
    def __init__(self, alpha=0.8):
        self.alpha = alpha
        self.prev_gaze = {}
    
    def smooth(self, pid, start, end):
        if pid not in self.prev_gaze:
            self.prev_gaze[pid] = (start, end)
            return start, end
        
        prev_start, prev_end = self.prev_gaze[pid]
        smooth_start = self.alpha * start + (1 - self.alpha) * prev_start
        smooth_end = self.alpha * end + (1 - self.alpha) * prev_end
        
        self.prev_gaze[pid] = (smooth_start, smooth_end)
        return smooth_start, smooth_end
    
    def cleanup(self, active_ids):
        for pid in list(self.prev_gaze.keys()):
            if pid not in active_ids:
                del self.prev_gaze[pid]

gaze_smoother = GazeSmoother(alpha=0.6)

# Skeleton & Keypoint indices (YOLO format, 0-based)
NOSE_IDX = 0
LEFT_EYE_IDX = 1
RIGHT_EYE_IDX = 2
LEFT_EAR_IDX = 3
RIGHT_EAR_IDX = 4

SKELETON = [(15,13),(13,11),(16,14),(14,12),(11,12),(5,11),(6,12),(5,6),(5,7),(6,8),(7,9),
            (8,10),(1,2),(0,1),(0,2),(1,3),(2,4),(3,5),(4,6)]

COLORS = [(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),(0,255,0),(0,255,85),
          (0,255,170),(0,255,255),(0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),(255,0,255),
          (255,0,170),(255,0,85)]

# Gaze calculation using YOLO's low-confidence predictions
def calc_gaze(kpts, confs, pid=None):
    """Calculate gaze using YOLO's predictions even at low confidence."""
    nose = kpts[NOSE_IDX]
    left_ear = kpts[LEFT_EAR_IDX]
    right_ear = kpts[RIGHT_EAR_IDX]
    nose_conf = confs[NOSE_IDX]
    left_ear_conf = confs[LEFT_EAR_IDX]
    right_ear_conf = confs[RIGHT_EAR_IDX]
    
    # Use YOLO's predictions if ANY confidence (very low threshold)
    has_nose = nose_conf > KPT_THRESH_GAZE
    has_left_ear = left_ear_conf > KPT_THRESH_GAZE
    has_right_ear = right_ear_conf > KPT_THRESH_GAZE
    
    # Need nose + at least one ear
    if not has_nose or (not has_left_ear and not has_right_ear):
        return None, None, False
    
    # If one ear missing, use last known position from smoother
    if not has_left_ear and pid is not None:
        last_left = smoother.get_last_valid(pid, LEFT_EAR_IDX)
        if last_left is not None:
            left_ear = last_left
            has_left_ear = True
    
    if not has_right_ear and pid is not None:
        last_right = smoother.get_last_valid(pid, RIGHT_EAR_IDX)
        if last_right is not None:
            right_ear = last_right
            has_right_ear = True
    
    # Still need both ears
    if not (has_left_ear and has_right_ear):
        return None, None, False
    
    # Calculate gaze direction
    ear_mid = (left_ear + right_ear) / 2.0
    gaze_dir = nose - ear_mid
    gaze_len = np.linalg.norm(gaze_dir)
    
    if gaze_len < 1e-6:
        return None, None, False
    
    gaze_dir_norm = gaze_dir / gaze_len
    angle_rad = -np.radians(GAZE_ANGLE_CORRECTION)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x, y = gaze_dir_norm[0], gaze_dir_norm[1]
    x_rot = cos_a * x - sin_a * y
    y_rot = sin_a * x + cos_a * y
    gaze_dir_corrected = np.array([x_rot, y_rot])
    
    start_pt = ear_mid
    end_pt = start_pt + gaze_dir_corrected * GAZE_LENGTH
    return start_pt, end_pt, True

# Video capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO_PATH}")

w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate display dimensions
if DISPLAY_KEEP_ASPECT:
    scale = min(DISPLAY_MAX_WIDTH / w, DISPLAY_MAX_HEIGHT / h, 1.0)
    display_w, display_h = int(w * scale), int(h * scale)
else:
    display_w, display_h = min(w, DISPLAY_MAX_WIDTH), min(h, DISPLAY_MAX_HEIGHT)
    scale = display_w / w

print(f"Resolution: {w}x{h}")
print(f"FPS: {fps:.1f}")
print(f"Duration: {total/fps:.1f}s ({total} frames)")
print(f"Display size: {display_w}x{display_h}")
print(f"\nControls: [A]<<{SKIP_SECONDS}s [D]>>{SKIP_SECONDS}s [Space]Pause [R]Record [Q]Quit")

# Main loop
paused, recording, writer, out_file = False, False, None, None
frame_idx, fps_cnt, t_last, last_frame = 0, 0, time.time(), None

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Video end")
            if writer:
                writer.release()
                print(f"Recording saved: {out_file}")
            paused = True
            continue
        
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # YOLO inference with tracking
        results = model.track(frame, conf=CONF_THRESH, persist=True, verbose=False, device=device)
        r = results[0]
        
        vis = frame.copy()
        num_persons = 0
        
        # Process detections
        if r.keypoints is not None and len(r.keypoints) > 0:
            all_kpts = r.keypoints.xy.cpu().numpy()
            kpts_conf = r.keypoints.conf.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else list(range(len(all_kpts)))
            boxes_conf = r.boxes.conf.cpu().numpy()
            num_persons = len(all_kpts)
            
            # Cleanup
            smoother.cleanup(track_ids)
            gaze_smoother.cleanup(track_ids)
            
            # Draw poses
            for idx, (kpts, tid, bconf) in enumerate(zip(all_kpts, track_ids, boxes_conf)):
                # Apply smoothing to all keypoints
                smooth_kpts, smooth_conf = smoother.smooth(tid, kpts, kpts_conf[idx])
                
                # Draw skeleton with colored lines
                for i, (st, en) in enumerate(SKELETON):
                    if smooth_conf[st] > KPT_THRESH and smooth_conf[en] > KPT_THRESH:
                        cv2.line(vis, tuple(map(int, smooth_kpts[st])), tuple(map(int, smooth_kpts[en])),
                                COLORS[i % len(COLORS)], 1, cv2.LINE_AA)
                
                # Calculate gaze using low threshold (YOLO's internal estimation)
                start, end, valid = calc_gaze(smooth_kpts, smooth_conf, tid)
                
                if valid:
                    # Apply additional smoothing to gaze
                    start, end = gaze_smoother.smooth(tid, start, end)
                    
                    left_ear = smooth_kpts[LEFT_EAR_IDX]
                    right_ear = smooth_kpts[RIGHT_EAR_IDX]
                    
                    # Draw ear connection (only if both visible with normal threshold)
                    if smooth_conf[LEFT_EAR_IDX] > KPT_THRESH and smooth_conf[RIGHT_EAR_IDX] > KPT_THRESH:
                        cv2.line(vis, tuple(map(int, left_ear)), tuple(map(int, right_ear)),
                                (150,150,150), 1, cv2.LINE_AA)
                    
                    # Draw ear midpoint
                    cv2.circle(vis, tuple(map(int, start)), 2, (0,255,0), -1)
                    
                    # Draw gaze ray
                    cv2.line(vis, tuple(map(int, start)), tuple(map(int, end)),
                            (0,255,255), 1, cv2.LINE_AA)
                    
                    # Draw gaze endpoint
                    cv2.circle(vis, tuple(map(int, end)), 3, (0,0,255), -1)
                
                # Draw keypoints (red for nose, blue for ears, green for others)
                for i, (x, y) in enumerate(smooth_kpts):
                    if smooth_conf[i] > KPT_THRESH:
                        if i == NOSE_IDX:
                            color = (0,0,255)
                        elif i in [LEFT_EAR_IDX, RIGHT_EAR_IDX]:
                            color = (255,0,0)
                        else:
                            color = (0,255,0)
                        cv2.circle(vis, (int(x), int(y)), 2, color, -1)
                
                # Draw labels
                visible = smooth_kpts[smooth_conf > KPT_THRESH]
                x, y = (int(visible[np.argmin(visible[:,1])][0])-30, int(visible[np.argmin(visible[:,1])][1])-30) \
                       if len(visible)>0 else (int(smooth_kpts[0,0])-30, int(smooth_kpts[0,1])-30)
                cv2.putText(vis, f"ID:{tid}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.putText(vis, f"{bconf:.2f}", (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 2)
        
        # Calculate video position
        curr_t = frame_idx / fps if fps > 0 else 0
        total_t = total / fps if fps > 0 else 0
        
        # Info overlay
        info = f"Persons: {num_persons} | {curr_t:.1f}s / {total_t:.1f}s"
        if recording:
            info += " | [REC]"
        cv2.putText(vis, info, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if recording else (0,255,0), 2)
        cv2.putText(vis, f"[A]<<{SKIP_SECONDS}s [D]>>{SKIP_SECONDS}s [Space]Pause [R]Record [Q]Quit",
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # Recording
        if recording and writer:
            writer.write(vis)
        
        last_frame = vis
        fps_cnt += 1
        
        # FPS counter
        now = time.time()
        if now - t_last >= 1.0:
            print(f"FPS: {fps_cnt/(now-t_last):.1f} | Frame: {frame_idx}/{total} | Time: {curr_t:.1f}s | {device.upper()}")
            t_last, fps_cnt = now, 0
    
    else:
        # Show paused frame
        if last_frame is not None:
            vis = last_frame.copy()
            cv2.putText(vis, "PAUSED - [Space] Continue", (w//2-200, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        else:
            vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Resize for display
    if scale < 1.0:
        vis_display = cv2.resize(vis, (display_w, display_h), interpolation=cv2.INTER_AREA)
    else:
        vis_display = vis
    
    cv2.imshow("YOLOv11 Pose Video", vis_display)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == 27:
        if writer:
            writer.release()
            print(f"Recording saved: {out_file}")
        break
    
    elif key == ord(' '):
        paused = not paused
        print("PAUSED" if paused else "PLAYING")
    
    elif key == ord('r'):
        if not recording:
            recording = True
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = OUTPUT_DIR / f"recording_{timestamp}.mp4"
            writer = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f"Recording started: {out_file}")
        else:
            recording = False
            if writer:
                writer.release()
                print(f"Recording saved: {out_file}")
            writer = None
    
    elif key == ord('a'):
        pos = max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - SKIP_SECONDS * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        frame_idx = int(pos)
        print(f"← Skip to frame {frame_idx} ({frame_idx/fps:.1f}s)")
        paused = False
    
    elif key == ord('d'):
        pos = min(total-1, cap.get(cv2.CAP_PROP_POS_FRAMES) + SKIP_SECONDS * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        frame_idx = int(pos)
        print(f"→ Skip to frame {frame_idx} ({frame_idx/fps:.1f}s)")
        paused = False

# Cleanup
if writer:
    writer.release()
cap.release()
cv2.destroyAllWindows()

if device == 'cuda':
    torch.cuda.empty_cache()