import cv2
import numpy as np
import os
import shutil
from pathlib import Path
from rtmlib import Body

# Config
INPUT_DIR = Path("media")
INPUT_FILENAME = "test1.jpg"
OUTPUT_FILENAME = INPUT_FILENAME.rsplit('.', 1)[0] + "_rtmpose." + INPUT_FILENAME.rsplit('.', 1)[1]

# Model setup - like rtmpose_level_1.py
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = SCRIPT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Force rtmlib to use local models directory
os.environ['RTMLIB_HOME'] = str(MODEL_DIR)
os.environ['TORCH_HOME'] = str(MODEL_DIR)

# Set CACHE_DIR if available
import rtmlib.tools.solution.body
if hasattr(rtmlib.tools.solution.body, 'CACHE_DIR'):
    rtmlib.tools.solution.body.CACHE_DIR = MODEL_DIR

# Device selection
device = "cpu"
print(f"Device: {device}")

# Load body model)
body = Body(mode='performance', backend='onnxruntime', device=device, to_openpose=False)

# Copy models from cache to local directory if they exist
cache_check = Path.home() / '.cache' / 'rtmlib' / 'hub' / 'checkpoints'
if cache_check.exists():
    for f in cache_check.glob("*.onnx"):
        if not (MODEL_DIR / f.name).exists():
            shutil.copy2(f, MODEL_DIR / f.name)

print(f"Models: {MODEL_DIR}")

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