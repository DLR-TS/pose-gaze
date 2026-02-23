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

# Tracker
class Tracker:
    def __init__(self, max_dist=100, max_age=30):
        self.persons, self.next_id = {}, 0
        self.max_dist, self.max_age = max_dist, max_age
    
    def update(self, kpts_list, scrs_list):
        if not kpts_list:
            for pid in list(self.persons.keys()):
                self.persons[pid]["age"] += 1
                if self.persons[pid]["age"] > self.max_age: del self.persons[pid]
            return []
        
        centers = [kpts[scrs > KPT_THRESH].mean(axis=0) if (scrs > KPT_THRESH).sum() > 0 
                   else kpts.mean(axis=0) for kpts, scrs in zip(kpts_list, scrs_list)]
        
        if not self.persons:
            ids = []
            for c in centers:
                self.persons[self.next_id] = {"center": c, "age": 0}
                ids.append(self.next_id)
                self.next_id += 1
            return ids
        
        ids, matched = [], set()
        for c in centers:
            best_id, best_dist = None, float('inf')
            for pid, data in self.persons.items():
                if pid in matched: continue
                dist = np.linalg.norm(c - data["center"])
                if dist < best_dist and dist < self.max_dist:
                    best_dist, best_id = dist, pid
            
            if best_id:
                ids.append(best_id)
                matched.add(best_id)
                self.persons[best_id] = {"center": c, "age": 0}
            else:
                self.persons[self.next_id] = {"center": c, "age": 0}
                ids.append(self.next_id)
                self.next_id += 1
        
        for pid in list(self.persons.keys()):
            if pid not in matched:
                self.persons[pid]["age"] += 1
                if self.persons[pid]["age"] > self.max_age: del self.persons[pid]
        return ids

tracker = Tracker()

# Skeleton definition (RTMPose format, 1-based)
SKELETON = [(16,14),(14,12),(17,15),(15,13),(12,13),(6,12),(7,13),(6,7),(6,8),(7,9),(8,10),(9,11),
            (2,3),(1,2),(1,3),(2,4),(3,5),(4,6),(5,7)]

COLORS = [(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),(0,255,0),(0,255,85),
          (0,255,170),(0,255,255),(0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),(255,0,255),
          (255,0,170),(255,0,85)]

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
        
        # Filter detections
        valid_k, valid_s, valid_c = [], [], []
        for k, s in zip(kpts, scrs):
            avg = np.mean(s[s > 0])
            if avg > CONF_THRESH and (s > KPT_THRESH).sum() >= MIN_KEYPOINTS:
                valid_k.append(k)
                valid_s.append(s)
                valid_c.append(float(avg))
        
        ids = tracker.update(valid_k, valid_s)
        vis = frame.copy()
        
        # Draw poses
        for k, s, pid, conf in zip(valid_k, valid_s, ids, valid_c):
            # Draw skeleton with colored lines
            for i, (st, en) in enumerate(SKELETON):
                if s[st-1] > KPT_THRESH and s[en-1] > KPT_THRESH:
                    cv2.line(vis, tuple(map(int, k[st-1])), tuple(map(int, k[en-1])),
                            COLORS[i % len(COLORS)], 1, cv2.LINE_AA)
            
            # Draw keypoints (red for nose, green for others)
            for i, (x, y) in enumerate(k):
                if s[i] > KPT_THRESH:
                    cv2.circle(vis, (int(x), int(y)), 3, (0,0,255) if i==0 else (0,255,0), -1)
            
            # Draw labels (yellow text above highest visible keypoint)
            visible = k[s > KPT_THRESH]
            x, y = (int(visible[np.argmin(visible[:,1])][0])-30, int(visible[np.argmin(visible[:,1])][1])-30) \
                   if len(visible)>0 else (int(k[0,0])-30, int(k[0,1])-30)
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
        
        # Recording writes original resolution
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
    
    # Resize for display if needed
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
