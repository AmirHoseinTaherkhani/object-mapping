from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load your model
model = YOLO('weights/yolov8n.pt', task='detect')

# Run inference
results = model('weights/test.jpeg')

# Method 1: Save using the plot method
for i, result in enumerate(results):
    # Plot the result and save it
    plotted_img = result.plot()  # This returns the image with bounding boxes drawn
    cv2.imwrite(f'output_{i}.jpg', plotted_img)
    print(f"Saved output_{i}.jpg")

# Method 2: Display the image with matplotlib
for result in results:
    plotted_img = result.plot()
    # Convert BGR to RGB for matplotlib
    plotted_img_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(plotted_img_rgb)
    plt.axis('off')
    plt.title('Detection Results')
    plt.show()

# Method 3: Extract detection data manually
for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            # Get box coordinates (x, y, width, height)
            x, y, w, h = box.xywh[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            
            print(f"Detected {class_name} with confidence {confidence:.2f}")
            print(f"Location: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")