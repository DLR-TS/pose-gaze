from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# Config
INPUT_DIR = Path("media")
INPUT_FILENAME = "test3.jpg"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "yolo11x-pose.pt"  # X = best model
OUTPUT_FILENAME = INPUT_FILENAME.rsplit('.', 1)[0] + "_yolo." + INPUT_FILENAME.rsplit('.', 1)[1]

# Model setup
MODEL_DIR.mkdir(exist_ok=True)
model = YOLO(str(MODEL_PATH))

# Skeleton definition
SKELETON = [(15,13),(13,11),(16,14),(14,12),(11,12),(5,11),(6,12),(5,6),(5,7),(6,8),
            (7,9),(8,10),(1,2),(0,1),(0,2),(1,3),(2,4),(3,5),(4,6)]
COLORS = [(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),(0,255,0),(0,255,85),(0,255,170),
          (0,255,255),(0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85)]
KPT_THRESH = 0.25

# Load and process image
img = cv2.imread(str(INPUT_DIR / INPUT_FILENAME))
if img is None:
    print(f"ERROR: Image '{INPUT_FILENAME}' not found!"); exit(1)

results = model.track(img, conf=0.25, persist=True)
r = results[0]
img_vis = img.copy()

if r.keypoints is not None and len(r.keypoints) > 0:
    all_kpts = r.keypoints.xy.cpu().numpy()
    kpts_conf = r.keypoints.conf.cpu().numpy()
    track_ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else list(range(len(all_kpts)))
    boxes_conf = r.boxes.conf.cpu().numpy()
    
    print(f"OK: {len(all_kpts)} person(s) detected!")
    
    for idx, (kpts, tid, bconf) in enumerate(zip(all_kpts, track_ids, boxes_conf)):
        # Draw skeleton with colored lines
        for i, (s, e) in enumerate(SKELETON):
            if kpts_conf[idx][s] > KPT_THRESH and kpts_conf[idx][e] > KPT_THRESH:
                cv2.line(img_vis, tuple(map(int, kpts[s])), tuple(map(int, kpts[e])), 
                        COLORS[i % len(COLORS)], 1, cv2.LINE_AA)
        
        # Draw keypoints (green, nose red)
        for i, (x, y) in enumerate(kpts):
            if kpts_conf[idx][i] > KPT_THRESH:
                cv2.circle(img_vis, (int(x), int(y)), 3, (0,0,255) if i==0 else (0,255,0), -1)
        
        # Draw labels (yellow text above highest keypoint)
        visible = kpts[kpts_conf[idx] > KPT_THRESH]
        x, y = (int(visible[np.argmin(visible[:,1])][0])-30, int(visible[np.argmin(visible[:,1])][1])-30) \
            if len(visible)>0 else (int(kpts[0,0])-30, int(kpts[0,1])-30)
        cv2.putText(img_vis, f"ID:{tid}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(img_vis, f"{bconf:.2f}", (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 2)
        print(f"  Person ID {tid}: Confidence {bconf:.2f}")
else:
    print("WARNING: No persons detected!")

cv2.imwrite(str(INPUT_DIR / OUTPUT_FILENAME), img_vis)
print(f"OK: Image saved: {OUTPUT_FILENAME}")