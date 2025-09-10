import streamlit as st
import sys
import os

# Add webapp directory to path for imports
sys.path.append(os.path.dirname(__file__))
# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import styling components
from components.styling import (
    inject_custom_css,
    render_app_header,
    render_feature_card,
    render_metric_card,
    add_footer
)

def main():
    st.set_page_config(
        page_title="Video-to-Map Object Tracking System",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styling
    inject_custom_css()
    
    # Use styled header instead of st.title
    render_app_header(
        "Video-to-Map Object Tracking System",
        "Real-time surveillance footage analysis with coordinate mapping"
    )
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Home", "Ground Truth Annotation", "Real-time Mapping", "API Test"]
    )
    
    if page == "Home":
        show_home_page()
    elif page == "Ground Truth Annotation":
        show_ground_truth_page()
    elif page == "Real-time Mapping":
        show_realtime_mapping_page()
    elif page == "API Test":
        show_api_test_page()
    
    # Add footer
    add_footer()

def show_home_page():
    st.header("Complete Object Tracking Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_feature_card(
            "Ground Truth Annotation",
            "Create reference points for perspective transformation. Upload surveillance video, select frame landmarks, input real-world coordinates, and export JSON for homography calculation.",
            "🎯"
        )
        
    with col2:
        render_feature_card(
            "Real-time Mapping",
            "Live video processing with coordinate mapping. Includes object detection and tracking, real-world coordinate transformation, side-by-side visualization, and CSV export with trajectories.",
            "🗺️"
        )
    
    st.markdown("---")
    st.subheader("System Capabilities")
    
    # Performance metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        render_metric_card("4x", "GPU Speedup", "#00d4aa")
    
    with metrics_col2:
        render_metric_card("45+", "Max FPS", "#ffa726")
    
    with metrics_col3:
        render_metric_card("94%", "Track Consistency", "#00d4aa")
    
    with metrics_col4:
        render_metric_card("<30ms", "Total Latency", "#ffa726")
    
    # Feature cards
    features_col1, features_col2, features_col3 = st.columns(3)
    
    with features_col1:
        render_feature_card(
            "Detection & Tracking",
            "YOLO-based object detection with persistent ID assignment, GPU acceleration (MPS/CUDA), and configurable confidence thresholds.",
            "🔍"
        )
    
    with features_col2:
        render_feature_card(
            "Coordinate Mapping",
            "Homography transformation with pixel-to-meter conversion, error validation, and real-time processing capabilities.",
            "📐"
        )
    
    with features_col3:
        render_feature_card(
            "Visualization",
            "Side-by-side display with object trails, fade effects, interactive map canvas, and built-in performance monitoring.",
            "📊"
        )

def show_ground_truth_page():
    from pages.ground_truth import render_ground_truth_page
    render_ground_truth_page()

def show_realtime_mapping_page():
    from pages.realtime_mapping import render_realtime_mapping_page
    render_realtime_mapping_page()

def show_api_test_page():
    render_feature_card(
        "API Testing Interface",
        "Test the FastAPI detection endpoint with custom images and parameters.",
        "🧪"
    )
    
    # Upload image for detection
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Run Detection"):
            st.info("API integration coming soon. For now, use the real-time mapping page.")

if __name__ == "__main__":
    main()
