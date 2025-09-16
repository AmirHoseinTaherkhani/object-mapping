import cv2
import streamlit as st
import tempfile
import os
import numpy as np
from typing import Tuple, Optional

class VideoProcessor:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def get_video_info(self, video_path: str) -> dict:
        """Get video information"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'file_size': os.path.getsize(video_path) / (1024 * 1024)
        }
        cap.release()
        return info
    
    def resize_video(self, input_path: str, output_path: str, 
                     target_width: int = 640, target_height: int = 480,
                     quality_factor: float = 0.7, target_fps: Optional[int] = None) -> bool:
        """Resize and compress video"""
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return False
            
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = target_fps if target_fps else original_fps
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
            
            frame_skip = 1
            if target_fps and target_fps < original_fps:
                frame_skip = int(original_fps / target_fps)
            
            frame_count = 0
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    resized_frame = cv2.resize(frame, (target_width, target_height))
                    
                    if quality_factor < 1.0:
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality_factor * 100)]
                        _, encimg = cv2.imencode('.jpg', resized_frame, encode_param)
                        resized_frame = cv2.imdecode(encimg, 1)
                    
                    out.write(resized_frame)
                
                frame_count += 1
                progress_bar.progress(min(frame_count / total_frames, 1.0))
            
            cap.release()
            out.release()
            progress_bar.empty()
            return True
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return False
