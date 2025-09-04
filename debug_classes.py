from ultralytics import YOLO
import cv2

# Load your custom model
model = YOLO("models/weights/best.onnx", task="detect")

# Test on your video
video_path = "/Users/at1293/Desktop/ObjectMapping-from-github/experiments/video_test/v01.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if ret:
    results = model(frame, conf=0.5)
    for r in results:
        if r.boxes is not None:
            print("Detections found:")
            for box in r.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                print(f"  Class ID: {class_id}, Confidence: {confidence:.3f}")
                
                # Show bounding box info
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"    BBox: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
cap.release()
