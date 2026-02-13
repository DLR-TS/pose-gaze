import cv2
import numpy as np
from pathlib import Path
from rtmlib import Body

# Config
INPUT_DIR = Path("media")
INPUT_FILENAME = "test3.jpg"
MODEL_DIR = Path("models")
OUTPUT_FILENAME = INPUT_FILENAME.rsplit('.', 1)[0] + "_rtmpose." + INPUT_FILENAME.rsplit('.', 1)[1]

# Model paths
MODEL_DIR.mkdir(exist_ok=True)
DET_MODEL = str(MODEL_DIR / "yolox_x_8xb8-300e_humanart-a39d44ed.onnx")
POSE_MODEL = str(MODEL_DIR / "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.onnx")

# Model setup (use local models if available, otherwise download to cache)
try:
    body = Body(det=DET_MODEL, pose=POSE_MODEL, backend='onnxruntime', device='cpu')
    print(f"Using models from: {MODEL_DIR}")
except:
    body = Body(mode='performance', backend='onnxruntime', device='cpu')
    print(f"Downloading models to cache...")

# Skeleton definition
SKELETON = [(15,13),(13,11),(16,14),(14,12),(11,12),(5,11),(6,12),(5,6),(5,7),(6,8),(7,9),
            (8,10),(1,2),(0,1),(0,2),(1,3),(2,4),(3,5),(4,6)]
COLORS = [(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),(0,255,0),(0,255,85),(0,255,170),
          (0,255,255),(0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85)]
KPT_THRESH = 0.25

# Load and process image
img = cv2.imread(str(INPUT_DIR / INPUT_FILENAME))
if img is None:
    print(f"ERROR: Image not found!"); exit(1)

keypoints, scores = body(img)
img_vis = img.copy()

if len(keypoints) > 0:
    print(f"OK: {len(keypoints)} person(s) detected!")
    
    for idx, (kpts, scrs) in enumerate(zip(keypoints, scores)):
        avg_conf = np.mean(scrs[scrs > KPT_THRESH])
        
        # Draw skeleton with colored lines
        for i, (s, e) in enumerate(SKELETON):
            if scrs[s] > KPT_THRESH and scrs[e] > KPT_THRESH:
                cv2.line(img_vis, tuple(map(int, kpts[s])), tuple(map(int, kpts[e])), 
                        COLORS[i % len(COLORS)], 1, cv2.LINE_AA)
        
        # Draw keypoints (green, nose red)
        for i, (x, y) in enumerate(kpts):
            if scrs[i] > KPT_THRESH:
                cv2.circle(img_vis, (int(x), int(y)), 3, (0,0,255) if i==0 else (0,255,0), -1)
        
        # Draw labels (yellow text above highest keypoint)
        visible = kpts[scrs > KPT_THRESH]
        x, y = (int(visible[np.argmin(visible[:,1])][0])-30, int(visible[np.argmin(visible[:,1])][1])-30) \
            if len(visible)>0 else (int(kpts[0,0])-30, int(kpts[0,1])-30)
        cv2.putText(img_vis, f"ID:{idx}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(img_vis, f"{avg_conf:.2f}", (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 2)
        print(f"  Person ID {idx}: Confidence {avg_conf:.2f}")
else:
    print("WARNING: No persons detected!")

cv2.imwrite(str(INPUT_DIR / OUTPUT_FILENAME), img_vis)
print(f"OK: Image saved: {OUTPUT_FILENAME}")
