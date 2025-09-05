import streamlit as st
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    st.set_page_config(
        page_title="Video-to-Map Object Tracking System",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 Video-to-Map Object Tracking System")
    st.markdown("---")
    
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

def show_home_page():
    st.header("Complete Video-to-Map Object Tracking System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Ground Truth Annotation")
        st.write("Create reference points for perspective transformation:")
        st.write("• Upload surveillance video")
        st.write("• Select frame and click landmarks")
        st.write("• Input real-world coordinates")
        st.write("• Export JSON for homography calculation")
        
    with col2:
        st.subheader("🗺️ Real-time Mapping")
        st.write("Live video processing with coordinate mapping:")
        st.write("• Object detection and tracking")
        st.write("• Real-world coordinate transformation")
        st.write("• Side-by-side video and map visualization")
        st.write("• CSV export with trajectories")
    
    st.markdown("---")
    st.subheader("System Features")
    
    features_col1, features_col2, features_col3 = st.columns(3)
    
    with features_col1:
        st.write("**Detection & Tracking:**")
        st.write("• YOLO-based object detection")
        st.write("• Persistent ID assignment")
        st.write("• GPU acceleration (MPS/CUDA)")
        st.write("• Configurable confidence thresholds")
    
    with features_col2:
        st.write("**Coordinate Mapping:**")
        st.write("• Homography transformation")
        st.write("• Pixel-to-meter conversion")
        st.write("• Error validation")
        st.write("• Real-time processing")
    
    with features_col3:
        st.write("**Visualization:**")
        st.write("• Side-by-side display")
        st.write("• Object trails with fade effects")
        st.write("• Interactive map canvas")
        st.write("• Performance monitoring")

def show_ground_truth_page():
    from pages.ground_truth import render_ground_truth_page
    render_ground_truth_page()

def show_realtime_mapping_page():
    from pages.realtime_mapping import render_realtime_mapping_page
    render_realtime_mapping_page()

def show_api_test_page():
    st.header("API Testing Interface")
    st.write("Test the FastAPI detection endpoint")
    
    # Upload image for detection
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Run Detection"):
            st.info("API integration coming soon. For now, use the real-time mapping page.")

if __name__ == "__main__":
    main()
