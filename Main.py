from ultralytics import YOLO
import cv2
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone

# --- Firebase Init ---
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load YOLO model
model = YOLO("yolov8n.pt")

# Green color range (HSV)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Black/soil range (HSV)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

# --- Reference for pixel to cm conversion ---
reference_height_cm = 25.4   # 10 inches = 25.4 cm
reference_height_px = 330    # measured pixel height
scale_factor = reference_height_cm / reference_height_px

# --- Load test image ---
image_path = "test_image/test_1.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print("Failed to load image:", image_path)
    exit()

# Create output directories
os.makedirs("detected_output", exist_ok=True)
os.makedirs("debug_output", exist_ok=True)

# Copy for drawing
output_frame = frame.copy()

# Define 4 equal-width regions for plants
height, width, _ = frame.shape
plant_regions = np.linspace(0, width, 5, dtype=int)  # 5 points = 4 regions

# Debug: draw dividers
for x in plant_regions:
    cv2.line(output_frame, (x, 0), (x, height), (0, 0, 255), 2)

# Run YOLO detection
results = model(frame, verbose=False)

# Track plant data
plants_detected = {f"plant{i+1}": {"detected": False, "height_cm": None} for i in range(4)}

# Process YOLO detections
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        if cls_name != "potted plant":
            continue
        cls_name = "plant"

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # Determine which plant region this falls into
        box_mid_x = (x1 + x2) // 2
        plant_idx = np.searchsorted(plant_regions, box_mid_x) - 1
        if plant_idx < 0 or plant_idx >= 4:
            continue

        plant_name = f"plant{plant_idx+1}"

        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Soil mask
        mask_black = cv2.inRange(hsv_roi, lower_black, upper_black)
        soil_y = y2
        black_pixels = np.where(mask_black > 0)
        if black_pixels[0].size > 0:
            soil_y = y1 + np.max(black_pixels[0])

        # Plant mask
        mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
        green_pixels = np.where(mask_green > 0)
        if green_pixels[0].size == 0:
            continue

        top_leaf_y = y1 + np.min(green_pixels[0])

        # Height calculation
        plant_height_px = soil_y - top_leaf_y
        plant_height_cm = plant_height_px * scale_factor

        # Update plant detection
        plants_detected[plant_name] = {
            "detected": True,
            "height_cm": round(plant_height_cm, 2)
        }

        # Draw debug rectangle
        cv2.rectangle(output_frame, (x1, top_leaf_y), (x2, soil_y), (0, 255, 0), 2)
        cv2.putText(output_frame, f"{plant_name} {plant_height_cm:.2f} cm",
                    (x1, top_leaf_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

# Save output
out_path = "detected_output/plants_detected.jpg"
cv2.imwrite(out_path, output_frame)

# --- Store in Firestore ---
timestamp = datetime.now(timezone.utc).isoformat()

for plant_name, data in plants_detected.items():
    doc_ref = db.collection("plants_growth").document(plant_name)
    measurements_ref = doc_ref.collection("measurements").document(timestamp)

    measurements_ref.set({
        "detected": data["detected"],
        "height_cm": data["height_cm"],
        "timestamp": timestamp
    })

# Print results
for plant_name, data in plants_detected.items():
    if data["detected"]:
        print(f"{plant_name} detected, Height = {data['height_cm']} cm")
    else:
        print(f"{plant_name} NOT detected")

print(f"🖼Processed image saved at: {out_path}")
