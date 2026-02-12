from ultralytics import YOLO
import cv2
from pathlib import Path

# Definiere Model-Verzeichnis
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Model-Pfad
MODEL_PATH = MODEL_DIR / "yolo11n-pose.pt"

# Modell laden (wird automatisch nach MODEL_DIR heruntergeladen wenn nicht vorhanden)
model = YOLO(str(MODEL_PATH))

# Bild laden (BGR)
img = cv2.imread("media/test1.jpg")

# Pose-Inferenz
results = model(img)

# Erste erkannte Person
r = results[0]
keypoints = r.keypoints.xy[0].cpu().numpy()  # Shape: (K, 2) mit x,y

# YOLO visualisiert automatisch Pose-Skeleton
pose_img = r.plot()

cv2.imwrite("media/test1_yolo.jpg", pose_img)