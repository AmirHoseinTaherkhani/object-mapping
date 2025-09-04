import cv2
import os
import pandas as pd
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def resize_frame(frame, max_width=1280):
    """Resize frame to fit screen while preserving aspect ratio."""
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return frame

def get_bottom_side_coordinates(boxes, ids):
    """Extract bottom side coordinates with track IDs."""
    bottom_coordinates = []
    if ids is None:  # If no IDs (e.g., detection mode), skip IDs
        ids = [None] * len(boxes)
    for box, track_id in zip(boxes, ids):
        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
        bottom_left = (int(x_min), int(y_max))
        bottom_right = (int(x_max), int(y_max))
        bottom_coordinates.append({
            "track_id": int(track_id) if track_id is not None else None,
            "bottom_left": bottom_left,
            "bottom_right": bottom_right,
            "bottom_left_x": bottom_left[0],
            "bottom_left_y": bottom_left[1],
            "bottom_right_x": bottom_right[0],
            "bottom_right_y": bottom_right[1],
            "class": box.cls.item(),
            "confidence": box.conf.item()
        })
    return bottom_coordinates

def main():
    # Load the model
    model_path = "C:/AMIR/CosMos_Code/car_person/CODE/train/weights/best.onnx"
    model = YOLO(model_path, task="detect")

    # Video source
    source = "C:/AMIR/CosMos_Code/car_person/CODE/Data/V1.mp4" 
    save_dir = "V3"
    os.makedirs(save_dir, exist_ok=True)

    # Initialize CSV data
    csv_data = []

    # Process video with tracking
    results = model.track(
        source=source,
        stream=True,
        show=False,
        tracker="bytetrack.yaml",
        save=True,
        save_dir=save_dir,
    )

    # Set up display window
    cv2.namedWindow("Tracking with Bottom Coordinates", cv2.WINDOW_NORMAL)

    # Real-time processing
    frame_count = 0
    for result in results:
        frame_count += 1
        boxes = result.boxes
        ids = boxes.id  # Track IDs

        # Get bottom coordinates
        bottom_coords = get_bottom_side_coordinates(boxes, ids)

        # Collect data for CSV
        for coord in bottom_coords:
            track_id = coord["track_id"] if coord["track_id"] is not None else "N/A"
            cls = "person" if coord["class"] == 0 else "car"  # Adjust based on your classes
            csv_data.append({
                "Frame": frame_count,
                "Track_ID": track_id,
                "Class": cls,
                "Confidence": coord["confidence"],
                "Bottom_Left_X": coord["bottom_left_x"],
                "Bottom_Left_Y": coord["bottom_left_y"],
                "Bottom_Right_X": coord["bottom_right_x"],
                "Bottom_Right_Y": coord["bottom_right_y"]
            })

        # Visualize with track IDs and coordinates
        frame = result.plot()  # Draw bounding boxes and IDs
        for coord in bottom_coords:
            if coord["track_id"] is not None:
                # Draw bottom line
                cv2.line(
                    frame,
                    coord["bottom_left"],
                    coord["bottom_right"],
                    (0, 255, 0),
                    2,
                )
                # Draw track ID
                cv2.putText(
                    frame,
                    f"ID {coord['track_id']}",
                    (coord["bottom_left"][0], coord["bottom_left"][1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                # Draw coordinates
                coord_text = f"BL: ({coord['bottom_left_x']}, {coord['bottom_left_y']})"
                cv2.putText(
                    frame,
                    coord_text,
                    (coord["bottom_left"][0], coord["bottom_left"][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Resize frame for display
        frame = resize_frame(frame, max_width=1280)

        # Display frame
        cv2.imshow("Tracking with Bottom Coordinates", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
            break

    # Save coordinates to CSV
    csv_file = os.path.join(save_dir, "bottom_coordinates.csv")
    pd.DataFrame(csv_data).to_csv(csv_file, index=False)

    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()