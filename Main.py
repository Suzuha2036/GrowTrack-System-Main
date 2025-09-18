from ultralytics import YOLO
import cv2
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone
import time

# --- Firebase Init ---
cred = credentials.Certificate("firebase_key.json")  # ensure same project as the Android app
firebase_admin.initialize_app(cred)
db = firestore.client()


# Load YOLO model
model = YOLO("yolov8n.pt")

# Color ranges (HSV)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

# Scale factor (cm per pixel) - adjust for your setup
scale_factor = 25 / 168

# Output folder
os.makedirs("detected_output", exist_ok=True)

# Interval (seconds)
capture_interval = 15

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera")
    exit()

time.sleep(5)

try:
    while True:
        # Stabilize exposure
        for _ in range(30):
            ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            continue

        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
        output_frame = frame.copy()

        h, w, _ = frame.shape
        plant_regions = np.linspace(0, w, 3, dtype=int)

        for x in plant_regions:
            cv2.line(output_frame, (x, 0), (x, h), (0, 0, 255), 2)

        results = model(frame, verbose=False)

        plants_detected = {
            f"plant{i+1}": {"detected": False, "height_cm": None, "height_px": None}
            for i in range(2)
        }

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                if cls_name != "potted plant":
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                mid_x = (x1 + x2) // 2
                plant_idx = np.searchsorted(plant_regions, mid_x) - 1
                if plant_idx < 0 or plant_idx >= 4:
                    continue
                plant_name = f"plant{plant_idx+1}"

                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                mask_black = cv2.inRange(hsv_roi, lower_black, upper_black)
                soil_y = y2
                black_pixels = np.where(mask_black > 0)
                if black_pixels[0].size > 0:
                    soil_y = y1 + int(np.max(black_pixels[0]))

                mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
                green_pixels = np.where(mask_green > 0)
                if green_pixels[0].size == 0:
                    continue

                top_leaf_y = y1 + int(np.min(green_pixels[0]))
                height_px = max(0, soil_y - top_leaf_y)
                height_cm = height_px * scale_factor

                plants_detected[plant_name] = {
                    "detected": True,
                    "height_cm": round(float(height_cm), 2),
                    "height_px": int(height_px),
                }

                cv2.rectangle(output_frame, (x1, top_leaf_y), (x2, soil_y), (0, 255, 0), 2)
                cv2.putText(output_frame, f"{plant_name} {height_cm:.2f} cm",
                            (x1, max(0, top_leaf_y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        for i in range(2):
            plant_name = f"plant{i+1}"
            section_x_start = plant_regions[i]
            text = f"Detected: {plants_detected[plant_name]['height_px']} px" if plants_detected[plant_name]["detected"] else "No plant"
            cv2.putText(output_frame, text, (section_x_start + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Plant Detection", output_frame)

        ts_file = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = f"detected_output/plants_{ts_file}.jpg"
        cv2.imwrite(out_path, output_frame)
        print(f"Saved: {out_path}")

        # --- WRITE TO FIRESTORE ---
        now = datetime.now(timezone.utc)
        ts_iso = now.isoformat()

        batch = db.batch()
        for plant_name, data in plants_detected.items():
            doc_ref = db.collection("plant_growth").document(plant_name)

            payload = {
                "detected": bool(data["detected"]),
                "height_cm": float(data["height_cm"]) if data["height_cm"] is not None else None,
                "timestamp": ts_iso,
            }

            # 1) Append to history
            time_ref = doc_ref.collection("time").document(ts_iso)
            batch.set(time_ref, payload)

            # 2) Update latest
            current_ref = doc_ref.collection("current").document("latest")
            batch.set(current_ref, payload, merge=True)

        batch.commit()

        #system active
        status_ref = db.collection("plant_growth").document("system_status")

        now = datetime.now(timezone.utc).isoformat()
        status_ref.set({
            "timestamp": now,
            "type": "service"
        }, merge=True)

        for plant_name, data in plants_detected.items():
            if data["detected"]:
                print(f"{plant_name}: {data['height_cm']} cm ({data['height_px']} px)")
            else:
                print(f"{plant_name}: NOT detected")

        if cv2.waitKey(capture_interval * 1000) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping capture.")


finally:
    cap.release()
    cv2.destroyAllWindows()

