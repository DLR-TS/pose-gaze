import cv2
import numpy as np
import os

# Setze den richtigen Pfad
image_folder = r'D:\videos\images_ros'
os.chdir(image_folder)

# Erste Datei lesen für Dimensionen
first_frame = cv2.imread('left0000.jpg')
if first_frame is None:
    print("FEHLER: Datei nicht gefunden!")
    exit()

orig_height, orig_width = first_frame.shape[:2]
print(f"Original: {orig_width}x{orig_height}")

# === CROP-PARAMETER IN PROZENT ===
# Format: (x_start%, y_start%, x_end%, y_end%)
# (0, 0, 100, 100) = komplettes Bild
# (0, 0, 50, 50) = linke obere Ecke, 50% Breite/Höhe
# (25, 25, 75, 75) = zentrierter 50% Ausschnitt

# Wähle eine Crop-Option:
#crop_percent = (0, 0, 100, 100)  # Komplettes Bild

# crop_percent = (0, 0, 50, 50)    # Linke obere Ecke (50%)
# crop_percent = (50, 0, 100, 50)  # Rechte obere Ecke (50%)
# crop_percent = (0, 50, 50, 100)  # Linke untere Ecke (50%)
# crop_percent = (50, 50, 100, 100) # Rechte untere Ecke (50%)
# crop_percent = (25, 25, 75, 75)  # Zentriert (50%)
# crop_percent = (10, 10, 90, 90)  # Zentriert (80%)
crop_percent = (5, 18, 49, 82)  # Custom

# Berechne Pixel-Koordinaten aus Prozent
x_start_px = int((crop_percent[0] / 100.0) * orig_width)
y_start_px = int((crop_percent[1] / 100.0) * orig_height)
x_end_px = int((crop_percent[2] / 100.0) * orig_width)
y_end_px = int((crop_percent[3] / 100.0) * orig_height)

crop_width = x_end_px - x_start_px
crop_height = y_end_px - y_start_px

# Runde auf Vielfache von 2 für MP4-Kompatibilität (H.264 Anforderung)
crop_width = (crop_width // 2) * 2
crop_height = (crop_height // 2) * 2

# Korrigiere End-Koordinaten nach Rundung
x_end_px = x_start_px + crop_width
y_end_px = y_start_px + crop_height

print(f"Crop-Bereich: ({x_start_px}, {y_start_px}) bis ({x_end_px}, {y_end_px})")
print(f"Crop-Größe: {crop_width}x{crop_height} (gerade Zahlen für MP4)")

# Crop-Vorschau erstellen
preview = first_frame.copy()
cv2.rectangle(preview, (x_start_px, y_start_px), (x_end_px, y_end_px), (0, 255, 0), 5)
# Zusätzliche Info im Bild
cv2.putText(preview, f"Crop: {crop_width}x{crop_height}", 
            (x_start_px + 10, y_start_px + 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imwrite('crop_preview.jpg', preview)
print("Crop-Vorschau gespeichert als crop_preview.jpg")

# Zielauflösung für Video (auf Vielfache von 2 gerundet)
target_width = (crop_width // 2) * 2
target_height = (crop_height // 2) * 2
print(f"Video-Auflösung: {target_width}x{target_height}")
print(f"Videolänge bei 10 fps: {470/10} Sekunden")

# Mildere Schärfung
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

# Video erstellen mit 10 fps
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = os.path.join(image_folder, 'output_cropped.mp4')
video = cv2.VideoWriter(output_path, fourcc, 10, (target_width, target_height))

if not video.isOpened():
    print("FEHLER: VideoWriter konnte nicht geöffnet werden!")
    exit()

# Alle 470 Frames verarbeiten
for i in range(470):
    frame = cv2.imread(f'left{i:04d}.jpg')
    if frame is not None:
        # Crop-Bereich ausschneiden
        cropped = frame[y_start_px:y_end_px, x_start_px:x_end_px]
        
        # Skalieren falls nötig (nur wenn target != crop size)
        if cropped.shape[1] != target_width or cropped.shape[0] != target_height:
            if cropped.shape[1] > target_width:
                resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)
            else:
                resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        else:
            resized = cropped
        
        # Nachschärfen (mild)
        sharpened = cv2.filter2D(resized, -1, sharpen_kernel)
        
        video.write(sharpened)
        
        if i % 50 == 0:
            print(f"Verarbeitet: {i}/470")
    else:
        print(f"Warnung: left{i:04d}.jpg nicht gefunden")

video.release()
print(f"Video erstellt: {output_path}")
