from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

# Create output folder
os.makedirs("output", exist_ok=True)

# Load YOLOv8 segmentation model
model = YOLO("yolov8s-seg.pt")

# Function to get available camera indices
def find_cameras(max_test=5):
    indices = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.read()[0]:
            indices.append(i)
        cap.release()
    return indices

# Function to calculate height from mask
def get_mask_height(mask):
    mask = mask.astype(np.uint8)
    ys, _ = np.where(mask == 1)
    if len(ys) > 0:
        return int(np.max(ys) - np.min(ys))
    else:
        return 0

# Find all working cameras
camera_indices = find_cameras()
if not camera_indices:
    print("❌ No available cameras found.")
    exit()

print(f"📷 Found cameras: {camera_indices}")

# Loop over cameras in sequence
current_index = 0
while True:
    cam_id = camera_indices[current_index]
    print(f"🎥 Using camera {cam_id}")
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    time.sleep(2)  # Let camera warm up

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"⚠️ Failed to capture from camera {cam_id}")
    else:
        results = model(frame)

        # Get height of "person" class only
        if results[0].masks is not None and results[0].boxes is not None:
            names = model.names  # class index to label mapping
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            found_person = False
            for i, class_id in enumerate(classes):
                label = names[class_id]
                if label == "person":
                    found_person = True
                    mask = results[0].masks.data[i].cpu().numpy()
                    height_px = get_mask_height(mask)
                    print(f"🧍 Plant height from cam {cam_id}: {height_px} px")

                    # Save person mask image with timestamp
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    mask_filename = f"output/mask_person_cam{cam_id}_{timestamp}.png"
                    cv2.imwrite(mask_filename, mask * 255)
                    break

            if not found_person:
                print("⚠️ No 'person' class detected.")
        else:
            print("⚠️ No masks or boxes detected.")

    # Move to next camera after 10 seconds
    current_index = (current_index + 1) % len(camera_indices)
    time.sleep(10)
