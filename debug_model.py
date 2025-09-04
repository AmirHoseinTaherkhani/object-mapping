from ultralytics import YOLO
import cv2

# Load your custom retrained model
model_path = "models/weights/best.onnx"
model = YOLO(model_path)

# Check model info
print("Custom model class names:", model.names)
print("Number of classes:", len(model.names))

# Test on a frame
video_path = "/Users/at1293/Desktop/ObjectMapping-from-github/experiments/video_test/v01.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if ret:
    results = model(frame, conf=0.7)
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                print(f"Detected: Class {int(box.cls.item())}, Confidence: {float(box.conf.item()):.3f}")
cap.release()
