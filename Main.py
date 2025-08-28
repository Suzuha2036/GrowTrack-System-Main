from ultralytics import YOLO
import cv2
import numpy as np
import os

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
image_path = "test_image/test_1.jpg"   # change to your file
frame = cv2.imread(image_path)

if frame is None:
    print("Failed to load image:", image_path)
    exit()

# Copy for drawing
output_frame = frame.copy()

# Run YOLO detection (suppress console logs)
results = model(frame, verbose=False)

for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])  # class ID
        cls_name = model.names[cls_id]  # class name

        if cls_name != "potted plant":
            continue
        cls_name = "plant"

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # --- Soil detection (black) ---
        mask_black = cv2.inRange(hsv_roi, lower_black, upper_black)
        soil_y = y2
        black_pixels = np.where(mask_black > 0)
        if black_pixels[0].size > 0:
            soil_y = y1 + np.max(black_pixels[0])

        # --- Plant detection (green) ---
        mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
        green_pixels = np.where(mask_green > 0)
        if green_pixels[0].size == 0:
            continue

        top_leaf_y = y1 + np.min(green_pixels[0])

        # --- Height calculation ---
        plant_height_px = soil_y - top_leaf_y
        plant_height_cm = plant_height_px * scale_factor

        # --- Draw on original image ---
        cv2.rectangle(output_frame, (x1, top_leaf_y), (x2, soil_y), (0, 255, 0), 2)
        mid_x = (x1 + x2) // 2
        cv2.line(output_frame, (mid_x, top_leaf_y), (mid_x, soil_y), (255, 0, 0), 2)
        cv2.putText(output_frame, f"Plant Height: {plant_height_cm:.2f} cm",
                    (x1, top_leaf_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Save output
os.makedirs("detected_output", exist_ok=True)
out_path = "detected_output/plants_detected.jpg"
cv2.imwrite(out_path, output_frame)

print(f"✅ Processed image saved at: {out_path}")

# Show preview
screen_res = 1280, 720   # change if your screen is bigger
scale_width = screen_res[0] / output_frame.shape[1]
scale_height = screen_res[1] / output_frame.shape[0]
scale = min(scale_width, scale_height)

# Resize to fit screen
window_width = int(output_frame.shape[1] * scale)
window_height = int(output_frame.shape[0] * scale)
resized = cv2.resize(output_frame, (window_width, window_height))

cv2.imshow("Plant Height Detection", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
