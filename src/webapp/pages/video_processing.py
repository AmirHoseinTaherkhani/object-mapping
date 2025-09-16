import streamlit as st
import tempfile
import os
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from video_processor import VideoProcessor

def render_video_processing_page():
    st.header("📹 Video Processing & Optimization")
    st.write("Optimize large videos for better tracking performance")
    
    # Upload video
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload videos up to 4GB. Large videos will be automatically optimized."
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(uploaded_file.read())
        temp_input.close()
        
        # Initialize processor
        processor = VideoProcessor()
        
        # Get video info
        video_info = processor.get_video_info(temp_input.name)
        if video_info:
            st.subheader("Original Video Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
            with col2:
                st.metric("FPS", f"{video_info['fps']:.1f}")
            with col3:
                st.metric("File Size", f"{video_info['file_size']:.1f} MB")
            
            # Processing options
            st.subheader("Processing Options")
            col1, col2 = st.columns(2)
            
            with col1:
                target_width = st.slider("Width", 240, 1280, 640)
                target_height = st.slider("Height", 180, 720, 480)
            
            with col2:
                quality = st.slider("Quality", 0.1, 1.0, 0.7, 0.1)
                reduce_fps = st.checkbox("Reduce FPS")
                if reduce_fps:
                    target_fps = st.slider("Target FPS", 5, 30, 15)
                else:
                    target_fps = None
            
            # Process button
            if st.button("🚀 Process Video"):
                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='_processed.mp4')
                temp_output.close()
                
                success = processor.resize_video(
                    temp_input.name, temp_output.name,
                    target_width, target_height, quality, target_fps
                )
                
                if success:
                    processed_info = processor.get_video_info(temp_output.name)
                    st.success("Video processed successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("New Resolution", f"{processed_info['width']}x{processed_info['height']}")
                    with col2:
                        st.metric("New FPS", f"{processed_info['fps']:.1f}")
                    with col3:
                        st.metric("New Size", f"{processed_info['file_size']:.1f} MB")
                    
                    compression_ratio = (1 - processed_info['file_size'] / video_info['file_size']) * 100
                    st.metric("Size Reduction", f"{compression_ratio:.1f}%")
                    
                    # Download processed video
                    with open(temp_output.name, 'rb') as f:
                        st.download_button(
                            "📥 Download Processed Video",
                            f.read(),
                            f"processed_{uploaded_file.name}",
                            "video/mp4"
                        )
                    
                    # Store path for other pages
                    st.session_state['processed_video_path'] = temp_output.name
