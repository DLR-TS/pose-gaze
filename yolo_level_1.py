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
CONF_THRESH = 0.25

# Display Settings
DISPLAY_MAX_WIDTH = 1600
DISPLAY_MAX_HEIGHT = 900
DISPLAY_KEEP_ASPECT = True

# Model setup
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "yolo11x-pose.pt"  # X = best model (largest, most accurate)

# GPU/CUDA Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = YOLO(str(MODEL_PATH))
model.to(device)
print(f"Models: {MODEL_PATH}")

# Skeleton definition (YOLO format, 0-based)
SKELETON = [(15,13),(13,11),(16,14),(14,12),(11,12),(5,11),(6,12),(5,6),(5,7),(6,8),(7,9),
            (8,10),(1,2),(0,1),(0,2),(1,3),(2,4),(3,5),(4,6)]

COLORS = [(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),(0,255,0),(0,255,85),(0,255,170),
          (0,255,255),(0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85)]

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
            
            # Draw poses
            for idx, (kpts, tid, bconf) in enumerate(zip(all_kpts, track_ids, boxes_conf)):
                # Draw skeleton with colored lines
                for i, (st, en) in enumerate(SKELETON):
                    if kpts_conf[idx][st] > KPT_THRESH and kpts_conf[idx][en] > KPT_THRESH:
                        cv2.line(vis, tuple(map(int, kpts[st])), tuple(map(int, kpts[en])),
                                COLORS[i % len(COLORS)], 1, cv2.LINE_AA)
                
                # Draw keypoints (red for nose, green for others)
                for i, (x, y) in enumerate(kpts):
                    if kpts_conf[idx][i] > KPT_THRESH:
                        cv2.circle(vis, (int(x), int(y)), 3, (0,0,255) if i==0 else (0,255,0), -1)
                
                # Draw labels (yellow text above highest visible keypoint)
                visible = kpts[kpts_conf[idx] > KPT_THRESH]
                x, y = (int(visible[np.argmin(visible[:,1])][0])-30, int(visible[np.argmin(visible[:,1])][1])-30) \
                       if len(visible)>0 else (int(kpts[0,0])-30, int(kpts[0,1])-30)
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

# GPU Memory cleanup
if device == 'cuda':
    torch.cuda.empty_cache()
