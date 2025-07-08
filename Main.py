from ultralytics import YOLO
import cv2
import numpy as np

# 🔢 Function to calculate mask area in pixels
def calculate_mask_areas(masks):
    """
    Calculates the area (in pixels) of each binary segmentation mask.
    """
    areas = []
    for mask in masks.data:
        binary_mask = mask.cpu().numpy().astype(np.uint8)  # Convert to NumPy uint8
        area = np.sum(binary_mask)  # Count all non-zero pixels
        areas.append(area)
    return areas

# 1. Load the YOLOv8 Segmentation model
model = YOLO('yolov8s-seg.pt')  # Ensure this is a segmentation model

# 2. Initialize webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    # 3. Save the captured image (optional)
    cv2.imwrite("captured.jpg", frame)

    # 4. Run YOLOv8 segmentation on the captured frame
    results = model(frame)

    # 5. Plot the results with masks
    annotated_image = results[0].plot()

    # ✅ 6. Calculate and print the segmented area(s)
    if results[0].masks is not None:
        areas = calculate_mask_areas(results[0].masks)
        for i, area in enumerate(areas):
            print(f"Mask {i+1} Area: {area} pixels")
    else:
        print("⚠️ No masks detected.")

    # 7. Display the result
    cv2.imshow("YOLOv8 Segmentation", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ Failed to capture image from webcam.")

cap.release()

