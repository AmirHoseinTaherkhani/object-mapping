"""
Ground Truth Annotation Page
Integrates the existing point selector functionality
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import tempfile
from streamlit_drawable_canvas import st_canvas
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

def render_ground_truth_page():
    st.header("Ground Truth Point Selector")
    st.markdown("Select reference points with known real-world coordinates for perspective transformation")
    
    # Initialize session state
    if 'selected_points' not in st.session_state:
        st.session_state.selected_points = []
    if 'real_world_coords' not in st.session_state:
        st.session_state.real_world_coords = []
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = None
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Video Upload")
        uploaded_video = st.file_uploader(
            "Upload surveillance video",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload video file for point annotation"
        )
        
        if uploaded_video is not None:
            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.getvalue())
                video_path = tmp_file.name
            
            st.success("Video uploaded successfully")
            
            # Frame selection
            st.subheader("Frame Selection")
            
            # Load first frame to get total frames
            if st.session_state.current_frame is None:
                frame_data = load_video_frame(video_path, 0)
                if frame_data[0] is not None:
                    st.session_state.current_frame, total_frames = frame_data
            
            frame_number = st.number_input(
                "Frame number to annotate",
                min_value=0,
                max_value=total_frames-1 if 'total_frames' in locals() else 100,
                value=0,
                help="Select which frame to use for point annotation"
            )
            
            # Load selected frame
            if st.button("Load Frame"):
                frame_data = load_video_frame(video_path, frame_number)
                if frame_data[0] is not None:
                    st.session_state.current_frame = frame_data[0]
                    st.rerun()
            
            # Point management
            st.subheader("Point Management")
            st.info(f"Selected Points: {len(st.session_state.selected_points)}")
            st.info(f"Annotated Points: {len(st.session_state.real_world_coords)}")
            
            if st.button("Clear All Points"):
                st.session_state.selected_points = []
                st.session_state.real_world_coords = []
                st.rerun()
            
            # Real-world coordinate input
            if len(st.session_state.selected_points) > len(st.session_state.real_world_coords):
                st.subheader("Real-World Coordinates")
                point_idx = len(st.session_state.real_world_coords)
                image_point = st.session_state.selected_points[point_idx]
                
                st.write(f"**Point {point_idx + 1}**: Image coordinates ({image_point[0]:.1f}, {image_point[1]:.1f})")
                
                with st.form(f"coord_form_{point_idx}"):
                    description = st.text_input("Point description", placeholder="e.g., Corner of building")
                    col_x, col_y = st.columns(2)
                    
                    with col_x:
                        x_coord = st.number_input("X coordinate (meters)", format="%.2f")
                    with col_y:
                        y_coord = st.number_input("Y coordinate (meters)", format="%.2f")
                    
                    if st.form_submit_button("Add Coordinate"):
                        st.session_state.real_world_coords.append({
                            "x": x_coord,
                            "y": y_coord,
                            "description": description
                        })
                        st.success("Coordinate added!")
                        st.rerun()
            
            # Save functionality
            if len(st.session_state.selected_points) >= 4 and len(st.session_state.selected_points) == len(st.session_state.real_world_coords):
                st.subheader("Save Ground Truth")
                
                save_name = st.text_input("File name", value=f"{uploaded_video.name}_ground_truth.json")
                
                if st.button("Save Points"):
                    output_path = f"outputs/ground_truth/{save_name}"
                    save_ground_truth_points(output_path, uploaded_video.name, frame_number)
    
    with col1:
        if st.session_state.current_frame is not None:
            st.subheader("Click to Select Points")
            st.write("Click on recognizable landmarks whose real-world coordinates you know")
            
            # Create canvas for point selection
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=3,
                stroke_color="#FF0000",
                background_color="#FFFFFF",
                background_image=Image.fromarray(st.session_state.current_frame),
                update_streamlit=True,
                height=400,
                drawing_mode="point",
                point_display_radius=8,
                key="canvas",
            )
            
            # Process canvas clicks
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                current_points = []
                
                for obj in objects:
                    if obj["type"] == "circle":
                        x = obj["left"] + obj["radius"]
                        y = obj["top"] + obj["radius"]
                        current_points.append([x, y])
                
                # Update session state if points changed
                if len(current_points) != len(st.session_state.selected_points):
                    st.session_state.selected_points = current_points
                    # Truncate real_world_coords if we have fewer points now
                    if len(current_points) < len(st.session_state.real_world_coords):
                        st.session_state.real_world_coords = st.session_state.real_world_coords[:len(current_points)]
                    st.rerun()
            
            # Display current points
            if st.session_state.selected_points:
                st.subheader("Selected Points Summary")
                for i, (img_point, world_coord) in enumerate(zip(
                    st.session_state.selected_points, 
                    st.session_state.real_world_coords + [None] * len(st.session_state.selected_points)
                )):
                    if world_coord:
                        st.write(f"**Point {i+1}**: ({img_point[0]:.1f}, {img_point[1]:.1f}) → ({world_coord['x']:.2f}m, {world_coord['y']:.2f}m) - {world_coord['description']}")
                    else:
                        st.write(f"**Point {i+1}**: ({img_point[0]:.1f}, {img_point[1]:.1f}) → ⏳ Awaiting coordinates")
        else:
            st.info("Upload a video file to start selecting points")

def load_video_frame(video_path: str, frame_number: int = 0):
    """Extract specific frame from video for annotation"""
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_number >= total_frames:
            frame_number = 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_number} from video")
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), total_frames
        
    except Exception as e:
        st.error(f"Error loading video frame: {str(e)}")
        return None, 0

def save_ground_truth_points(output_path: str, video_name: str, frame_number: int):
    """Save point pairs in JSON format for later homography calculation"""
    ground_truth_data = {
        "video_name": video_name,
        "frame_number": frame_number,
        "image_points": st.session_state.selected_points,
        "world_points": st.session_state.real_world_coords,
        "timestamp": str(pd.Timestamp.now()),
        "total_points": len(st.session_state.selected_points)
    }
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(ground_truth_data, f, indent=2)
    
    st.success(f"Ground truth points saved to {output_path}")
