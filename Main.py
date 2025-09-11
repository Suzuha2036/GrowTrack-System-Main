from ultralytics import YOLO
import cv2
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone
import time


#main file does not saved on github

# --- Firebase Init ---
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load YOLO model
model = YOLO("best.pt")

# Green color range (HSV)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Black/soil range (HSV)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

# Fixed scale factor based on reference measurement
scale_factor = 25 / 168  # cm per pixel

# --- Output directories ---
os.makedirs("detected_output", exist_ok=True)

# --- Capture interval ---
capture_interval = 15  # seconds

# --- Initialize camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Warm-up time
time.sleep(5)

try:
    while True:
        # Capture multiple frames to stabilize exposure
        for _ in range(30):
            ret, frame = cap.read()

        if not ret:
            print("Failed to capture image")
            continue

        # Brighten slightly
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)

        # Copy for drawing
        output_frame = frame.copy()

        # Define 4 equal-width regions for plants
        height, width, _ = frame.shape
        plant_regions = np.linspace(0, width, 5, dtype=int)

        # Draw dividers
        for x in plant_regions:
            cv2.line(output_frame, (x, 0), (x, height), (0, 0, 255), 2)

        # Run YOLO detection
        results = model(frame, verbose=False)

        # Track plant data
        plants_detected = {f"plant{i+1}": {"detected": False, "height_cm": None, "height_px": None} for i in range(4)}

        # Process YOLO detections
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                if cls_name != "eggplant":
                    continue
                cls_name = "eggplant"

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

                # Height calculation using fixed scale factor
                plant_height_px = soil_y - top_leaf_y
                plant_height_cm = plant_height_px * scale_factor

                # Update plant detection
                plants_detected[plant_name] = {
                    "detected": True,
                    "height_cm": round(plant_height_cm, 2),
                    "height_px": plant_height_px
                }

                # Draw rectangle and label with smaller font
                cv2.rectangle(output_frame, (x1, top_leaf_y), (x2, soil_y), (0, 255, 0), 2)
                cv2.putText(output_frame, f"{plant_name} {plant_height_cm:.2f} cm",
                            (x1, top_leaf_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw text for each section with smaller font
        for i in range(4):
            plant_name = f"plant{i+1}"
            section_x_start = plant_regions[i]
            text_y = 30  # vertical position

            if plants_detected[plant_name]["detected"]:
                text = f"Detected: {plants_detected[plant_name]['height_px']} px"
            else:
                text = "No plant"

            text_x = section_x_start + 10
            cv2.putText(output_frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # --- Show live preview ---
        cv2.imshow("Plant Detection", output_frame)

        # Save output image
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = f"detected_output/plants_{timestamp}.jpg"
        cv2.imwrite(out_path, output_frame)
        print(f"🖼 Image saved at: {out_path}")

        # Store in Firestore
        ts_iso = datetime.now(timezone.utc).isoformat()
        for plant_name, data in plants_detected.items():
            doc_ref = db.collection("plants_growth").document(plant_name)
            measurements_ref = doc_ref.collection("measurements").document(ts_iso)
            measurements_ref.set({
                "detected": data["detected"],
                "height_cm": data["height_cm"],
                "timestamp": ts_iso
            })

        # Print results
        for plant_name, data in plants_detected.items():
            if data["detected"]:
                print(f"{plant_name} detected, Height = {data['height_cm']} cm, Pixels = {data['height_px']}")
            else:
                print(f"{plant_name} NOT detected")

        # Wait for 15 seconds or until 'q' key is pressed
        if cv2.waitKey(capture_interval * 1000) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stop automatic capture.")

finally:
    cap.release()
    cv2.destroyAllWindows()
