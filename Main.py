from ultralytics import YOLO
import cv2

# 1. Load the YOLOv8 Segmentation model
model = YOLO('yolov8s-seg.pt')  # Use a segmentation model

# 2. Initialize webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    # 3. Save the captured image (optional)
    cv2.imwrite("captured.jpg", frame)

    # 4. Run YOLOv8 segmentation on the captured frame
    results = model(frame)

    # 5. Plot the results with masks
    annotated_image = results[0].plot()  # Includes masks if model is segmentation

    # 6. Display the result
    cv2.imshow("YOLOv8 Segmentation", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ Failed to capture image from webcam.")

cap.release()
