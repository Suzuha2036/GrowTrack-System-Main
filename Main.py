from ultralytics import YOLO
import cv2
import numpy as np
import time
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Function to calculate mask area
def calculate_mask_areas(masks):
    areas = []
    for mask in masks.data:
        binary_mask = mask.cpu().numpy().astype(np.uint8)
        area = np.sum(binary_mask)
        areas.append(area)
    return areas

# Load YOLOv8 Segmentation model
model = YOLO('yolov8s-seg.pt')

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture image from webcam.")
            break

        # Run YOLOv8 segmentation
        results = model(frame)
        annotated_image = results[0].plot()

        # Process segmentation results
        if results[0].masks is not None:
            areas = [int(area) for area in calculate_mask_areas(results[0].masks)]

            data = {
                "timestamp": firestore.SERVER_TIMESTAMP,
                "detected": True,
                "mask_areas": areas
            }

            db.collection("yolo_detections").add(data)
            print(f"✅ Sent to Firestore: {data}")
        else:
            print("⚠️ No masks detected.")

        # Optional: Display the result
        cv2.imshow("YOLOv8 Segmentation", annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Wait 20 seconds before next capture
        time.sleep(20)

except KeyboardInterrupt:
    print("⏹️ Stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
