import cv2
import time
import numpy as np
import os
import shutil
from rtmlib import Body
from pathlib import Path
from datetime import datetime

# Config
VIDEO_PATH = "media/test0_cut.mp4"
CONF_THRESH = 0.5
KPT_THRESH = 0.25
MIN_KEYPOINTS = 3
SKIP_SECONDS = 5

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
DISPLAY_MAX_WIDTH = 1920
DISPLAY_MAX_HEIGHT = 1080
DISPLAY_KEEP_ASPECT = True

# Model setup
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = SCRIPT_DIR / "models"
OUTPUT_DIR = SCRIPT_DIR / "recordings"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

os.environ['RTMLIB_HOME'] = str(MODEL_DIR)
os.environ['TORCH_HOME'] = str(MODEL_DIR)

import rtmlib.tools.solution.body
if hasattr(rtmlib.tools.solution.body, 'CACHE_DIR'):
    rtmlib.tools.solution.body.CACHE_DIR = MODEL_DIR

device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
print(f"Device: {device}")

body = Body(mode='performance', backend='onnxruntime', device=device, to_openpose=False)

cache_check = Path.home() / '.cache' / 'rtmlib' / 'hub' / 'checkpoints'
if cache_check.exists():
    for f in cache_check.glob("*.onnx"):
        if not (MODEL_DIR / f.name).exists():
            shutil.copy2(f, MODEL_DIR / f.name)

print(f"Models: {MODEL_DIR}")

# Smoother
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
    
    def cleanup(self, active_ids):
        for pid in list(self.prev_kpts.keys()):
            if pid not in active_ids:
                del self.prev_kpts[pid]
                del self.prev_scrs[pid]
                if pid in self.scr_hist: del self.scr_hist[pid]
                if pid in self.mov_hist: del self.mov_hist[pid]

smoother = Smoother()

# Gaze Smoother
class GazeSmoother:
    def __init__(self, alpha=0.6):
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

gaze_smoother = GazeSmoother()

# Tracker
class Tracker:
    def __init__(self, iou_thresh=0.25, max_age=90, min_hits=2):
        self.persons = {}
        self.next_id = 0
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
    
    def _iou(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0
    
    def _center_dist(self, b1, b2):
        c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
        c2 = np.array([(b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2])
        return np.linalg.norm(c1 - c2)
    
    def update(self, bboxes, confs):
        if not bboxes:
            for pid in list(self.persons.keys()):
                self.persons[pid]["age"] += 1
                if self.persons[pid]["age"] > self.max_age:
                    del self.persons[pid]
            confirmed = [pid for pid, d in self.persons.items() if d["hits"] >= self.min_hits]
            return confirmed, [self.persons[pid]["conf"] for pid in confirmed]
        
        if not self.persons:
            for bbox, conf in zip(bboxes, confs):
                self.persons[self.next_id] = {"bbox": bbox, "age": 0, "conf": conf, "hits": 1}
                self.next_id += 1
            return [], []
        
        ids, matched, matched_confs = [], set(), []
        for bbox, conf in zip(bboxes, confs):
            best_score, best_id = 0, None
            for pid, data in self.persons.items():
                if pid in matched: continue
                iou = self._iou(bbox, data["bbox"])
                center_dist = self._center_dist(bbox, data["bbox"])
                score = iou + (1.0 - min(1.0, center_dist / 100)) * 0.3
                if score > best_score and (iou > self.iou_thresh or center_dist < 100):
                    best_score, best_id = score, pid
            
            if best_id:
                ids.append(best_id)
                matched.add(best_id)
                self.persons[best_id]["bbox"] = bbox
                self.persons[best_id]["age"] = 0
                self.persons[best_id]["conf"] = conf
                self.persons[best_id]["hits"] += 1
                matched_confs.append(conf)
            else:
                self.persons[self.next_id] = {"bbox": bbox, "age": 0, "conf": conf, "hits": 1}
                ids.append(self.next_id)
                matched_confs.append(conf)
                self.next_id += 1
        
        for pid in list(self.persons.keys()):
            if pid not in matched:
                self.persons[pid]["age"] += 1
                if self.persons[pid]["age"] > self.max_age:
                    del self.persons[pid]
        
        confirmed = [pid for pid in ids if self.persons[pid]["hits"] >= self.min_hits]
        confirmed_confs = [self.persons[pid]["conf"] for pid in confirmed]
        return confirmed, confirmed_confs

tracker = Tracker()

# Skeleton & Keypoint indices
NOSE_IDX = 0
LEFT_EAR_IDX = 3
RIGHT_EAR_IDX = 4

SKELETON = [(16,14),(14,12),(17,15),(15,13),(12,13),(6,12),(7,13),(6,7),(6,8),(7,9),(8,10),(9,11),
            (2,3),(1,2),(1,3),(2,4),(3,5),(4,6),(5,7)]

COLORS = [(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),(0,255,0),(0,255,85),
          (0,255,170),(0,255,255),(0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),(255,0,255),
          (255,0,170),(255,0,85)]

# Gaze calculation
def calc_gaze(nose, left_ear, right_ear):
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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

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
        kpts, scrs = body(frame)
        
        # Filter detections and create bboxes
        bboxes, valid_data, confs = [], [], []
        for k, s in zip(kpts, scrs):
            avg = np.mean(s[s > 0])
            if avg > CONF_THRESH and (s > KPT_THRESH).sum() >= MIN_KEYPOINTS:
                vis_kpts = k[s > KPT_THRESH]
                x1, y1 = vis_kpts.min(axis=0)
                x2, y2 = vis_kpts.max(axis=0)
                # Expand bbox by 15%
                w_box, h_box = x2 - x1, y2 - y1
                x1 = max(0, x1 - w_box * 0.15)
                y1 = max(0, y1 - h_box * 0.15)
                x2 = min(frame.shape[1], x2 + w_box * 0.15)
                y2 = min(frame.shape[0], y2 + h_box * 0.15)
                bboxes.append([x1, y1, x2, y2])
                valid_data.append((k, s))
                confs.append(float(avg))
        
        # Update tracker and smoother
        ids, tracked_confs = tracker.update(bboxes, confs)
        smoother.cleanup(ids)
        gaze_smoother.cleanup(ids)
        
        vis = frame.copy()
        
        # Draw poses
        for (k, s), pid, conf in zip(valid_data, ids, tracked_confs):
            smooth_k, smooth_s = smoother.smooth(pid, k, s)
            
            # Draw skeleton
            for i, (st, en) in enumerate(SKELETON):
                if smooth_s[st-1] > KPT_THRESH and smooth_s[en-1] > KPT_THRESH:
                    cv2.line(vis, tuple(map(int, smooth_k[st-1])), tuple(map(int, smooth_k[en-1])),
                            COLORS[i % len(COLORS)], 1, cv2.LINE_AA)
            
            # Draw gaze ray
            if (smooth_s[NOSE_IDX] > KPT_THRESH and 
                smooth_s[LEFT_EAR_IDX] > KPT_THRESH and 
                smooth_s[RIGHT_EAR_IDX] > KPT_THRESH):
                nose = smooth_k[NOSE_IDX]
                left_ear = smooth_k[LEFT_EAR_IDX]
                right_ear = smooth_k[RIGHT_EAR_IDX]
                start, end, valid = calc_gaze(nose, left_ear, right_ear)
                
                if valid:
                    # Apply gaze smoothing
                    start, end = gaze_smoother.smooth(pid, start, end)
                    
                    # Draw ear connection
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
            for i, (x, y) in enumerate(smooth_k):
                if smooth_s[i] > KPT_THRESH:
                    if i == NOSE_IDX:
                        color = (0,0,255)
                    elif i in [LEFT_EAR_IDX, RIGHT_EAR_IDX]:
                        color = (255,0,0)
                    else:
                        color = (0,255,0)
                    cv2.circle(vis, (int(x), int(y)), 2, color, -1)
            
            # Draw labels
            visible = smooth_k[smooth_s > KPT_THRESH]
            x, y = (int(visible[np.argmin(visible[:,1])][0])-30, int(visible[np.argmin(visible[:,1])][1])-30) \
                   if len(visible)>0 else (int(smooth_k[0,0])-30, int(smooth_k[0,1])-30)
            cv2.putText(vis, f"ID:{pid}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(vis, f"{conf:.2f}", (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 2)
        
        # Calculate video position
        curr_t = frame_idx / fps if fps > 0 else 0
        total_t = total / fps if fps > 0 else 0
        
        # Info overlay
        info = f"Persons: {len(ids)} | {curr_t:.1f}s / {total_t:.1f}s"
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
    
    cv2.imshow("RTMPose", vis_display)
    
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
        print(f"<< Skip to frame {frame_idx} ({frame_idx/fps:.1f}s)")
        paused = False
    
    elif key == ord('d'):
        pos = min(total-1, cap.get(cv2.CAP_PROP_POS_FRAMES) + SKIP_SECONDS * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        frame_idx = int(pos)
        print(f">> Skip to frame {frame_idx} ({frame_idx/fps:.1f}s)")
        paused = False

# Cleanup
if writer:
    writer.release()
cap.release()
cv2.destroyAllWindows()