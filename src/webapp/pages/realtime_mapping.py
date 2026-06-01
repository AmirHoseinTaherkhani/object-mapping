"""
Real-time Mapping Page with Embedded Visualization
Shows video and map streams directly in the webapp.

Tracker: uses YOLO model.track() with ByteTrack (persist=True) instead of the
custom SimpleTracker. ByteTrack uses a Kalman filter + IoU re-identification,
so tracks survive short detection gaps without getting new IDs.
"""

import streamlit as st
import tempfile
import cv2
import numpy as np
import time
from pathlib import Path
import json
import sys
import os
import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from ultralytics import YOLO
from object_detection.mapping.homography import HomographyCalculator
from object_detection.visualization.map_canvas import MapCanvas


def render_realtime_mapping_page():
    st.header("Real-time Video-to-Map Tracking")
    st.markdown("Process videos with live object tracking and coordinate mapping")

    config_col, control_col = st.columns([1, 1])

    with config_col:
        st.subheader("Input Configuration")

        video_option = st.radio("Video Source", ["Upload File", "Camera"])

        if video_option == "Upload File":
            uploaded_video = st.file_uploader(
                "Upload video file",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload video for real-time processing"
            )
            video_source = None
            if uploaded_video:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video.getvalue())
                    video_source = tmp_file.name
                st.success("Video uploaded successfully")
        else:
            camera_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0)
            video_source = camera_index

        gt_files = list(Path("outputs/ground_truth").glob("*.json")) if Path("outputs/ground_truth").exists() else []

        if gt_files:
            gt_file_names = [f.name for f in gt_files]
            selected_gt = st.selectbox("Ground Truth File", gt_file_names)
            gt_path = f"outputs/ground_truth/{selected_gt}"

            if st.checkbox("Preview Ground Truth"):
                with open(gt_path, 'r') as f:
                    gt_data = json.load(f)
                st.json(gt_data)
        else:
            st.warning("No ground truth files found. Please create ground truth points first.")
            gt_path = None

    with control_col:
        st.subheader("Processing Parameters")

        model_files = list(Path("models/weights").glob("*.pt"))
        if model_files:
            model_names = [f.name for f in model_files]
            # Default to best_v3.pt if available, else best.pt
            default_idx = next((i for i, n in enumerate(model_names) if n == "best_v3.pt"), 0)
            selected_model = st.selectbox("Model", model_names, index=default_idx)
            model_path = f"models/weights/{selected_model}"
        else:
            st.error("No model files found")
            model_path = None

        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
        device = st.selectbox("Device", ["cpu", "mps", "cuda"], index=0)
        max_frames = st.number_input("Max Frames to Process", min_value=10, max_value=5000, value=300)

        tracker_type = st.selectbox(
            "Tracker",
            ["bytetrack.yaml", "botsort.yaml"],
            help="ByteTrack: faster, better in crowded scenes. BoTSORT: uses re-ID, better after occlusion."
        )

    can_process = all([
        video_source is not None,
        gt_path is not None,
        model_path is not None,
        Path(model_path).exists() if model_path else False
    ])

    if not can_process:
        missing = []
        if video_source is None:
            missing.append("Video source")
        if gt_path is None:
            missing.append("Ground truth file")
        if model_path is None or not Path(model_path).exists():
            missing.append("Model file")
        st.warning(f"Missing requirements: {', '.join(missing)}")
        return

    if st.button("Start Processing", disabled=not can_process):
        process_video_embedded(video_source, gt_path, model_path, confidence, device, max_frames, tracker_type)


def process_video_embedded(video_source, gt_path, model_path, confidence, device, max_frames, tracker_type):
    """Process video using YOLO's built-in ByteTrack for robust track persistence."""

    model = YOLO(model_path)

    homography_calc = HomographyCalculator(gt_path)
    homography_calc.calculate_homography()

    map_canvas = MapCanvas(
        width=400, height=300,
        ground_truth_file=gt_path,
        buffer_meters=10.0
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Video Tracking")
        video_placeholder = st.empty()
    with col2:
        st.subheader("2D Map")
        map_placeholder = st.empty()

    progress_bar = st.progress(0)
    status_text = st.empty()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Could not open video source")
        return

    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = 0
    tracking_data_export = []

    try:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO tracking with persist=True: the tracker maintains Kalman-filtered
            # state across frames, so IDs survive short occlusions/detection gaps.
            results = model.track(
                frame,
                persist=True,
                conf=confidence,
                device=device,
                tracker=tracker_type,
                verbose=False,
            )

            frame_export = {
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'objects': []
            }

            annotated = frame.copy()

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                ids = boxes.id  # None if no track assigned yet

                for i, box in enumerate(boxes):
                    track_id = int(ids[i].item()) if ids is not None else -1
                    conf_val = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x = (x1 + x2) / 2
                    bottom_y = y2  # bottom-center = ground contact point

                    world_x, world_y = homography_calc.transform_point(center_x, bottom_y)

                    if track_id >= 0:
                        map_canvas.update_object(track_id, world_x, world_y, class_name, conf_val)
                        color = map_canvas.get_track_color(track_id)
                    else:
                        color = (128, 128, 128)

                    # Draw bounding box + label on frame
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"ID:{track_id} {class_name} {conf_val:.2f}"
                    cv2.putText(annotated, label, (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

                    frame_export['objects'].append({
                        'track_id': track_id,
                        'class': class_name,
                        'confidence': round(conf_val, 4),
                        'bbox': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        'pixel_x': round(center_x, 1),
                        'pixel_y': round(bottom_y, 1),
                        'world_x': round(world_x, 3),
                        'world_y': round(world_y, 3),
                    })

            tracking_data_export.append(frame_export)

            map_image = map_canvas.render(show_trails=True, show_grid=True)

            video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                    channels="RGB", use_column_width=True)
            map_placeholder.image(cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB),
                                  channels="RGB", use_column_width=True)

            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
            n_obj = len(frame_export['objects'])
            status_text.text(f"Frame {frame_count}/{total_frames} — tracked objects: {n_obj}")

            time.sleep(0.02)

    except Exception as e:
        st.error(f"Processing error: {e}")

    finally:
        cap.release()

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"tracking_data_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(tracking_data_export, f, indent=2)

        total_detections = sum(len(f['objects']) for f in tracking_data_export)
        st.success(f"Processing completed. Saved to: {output_file}")
        st.info(f"Frames: {len(tracking_data_export)} | Total detections: {total_detections}")
