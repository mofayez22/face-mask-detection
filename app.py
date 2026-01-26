import streamlit as st
import PIL.Image
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import io
import json
import os
from inference import *
import yaml
import logging
import platform
from inference import run_inference_on_frame
from analytics_utils import *
from video_utils import *


# Add this helper function to load config
def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        st.warning(f"Config file not found at {config_path}. Using defaults.")
        return None
    except Exception as e:
        st.error(f"Error loading config: {str(e)}")
        return None


# Add error boundary wrapper
def safe_run(func):
    """Decorator for error handling"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            logging.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            return None

    return wrapper


# Add input validation
def validate_image(image):
    """Validate uploaded image"""
    if image is None:
        return False, "No image provided"

    # Check file size (limit to 10MB)
    if hasattr(image, "size"):
        max_size = 10 * 1024 * 1024  # 10MB
        if image.size > max_size:
            return False, f"Image too large. Max size: {max_size/(1024*1024):.1f}MB"

    # Check image dimensions
    try:
        img = PIL.Image.open(image)
        width, height = img.size
        if width > 4096 or height > 4096:
            return False, "Image dimensions too large. Max: 4096x4096"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"

    return True, None


# Add logging configuration
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
    )


# Add session state initialization
def init_session_state():
    """Initialize session state variables"""
    if "detection_history" not in st.session_state:
        st.session_state.detection_history = []
    if "total_images_processed" not in st.session_state:
        st.session_state.total_images_processed = 0


# Add performance metrics tracking
def track_performance(func):
    """Track function execution time"""
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        st.session_state.last_inference_time = duration
        return result

    return wrapper


# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="AI Face Mask Detection System",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CUSTOM CSS ====================
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: 600;
        border-radius: 8px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==================== HELPER FUNCTIONS ====================


def calculate_compliance_rate(stats):
    """Calculate mask compliance percentage"""
    total = stats["total_detections"]
    if total == 0:
        return 0
    compliant = stats["with_mask"]
    return (compliant / total) * 100


def export_results(stats, image_name):
    """Export detection results as JSON"""
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": image_name,
        "statistics": stats,
        "compliance_rate": calculate_compliance_rate(stats),
    }
    return json.dumps(report, indent=2)


# ==================== MAIN APPLICATION ====================
def main():

    # Load config
    config = load_config()

    # Setup logging
    setup_logging()

    # Initialize session state
    init_session_state()

    if "video_processing_done" not in st.session_state:
        st.session_state.video_processing_done = False


    st.markdown(
        '<h1 class="main-header">😷 AI Face Mask Detection System</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Detection Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

        st.subheader("Display Options")
        show_labels = st.checkbox("Show Labels", value=True)
        show_conf = st.checkbox("Show Confidence", value=True)

        st.markdown("---")
        st.info("💡 Adjust thresholds to balance sensitivity and accuracy.")

    model, error = load_model()

    if error:
        st.error(f"❌ Error loading model: {error}")
        st.info("Please ensure your model is trained and the path is correct.")
        st.code("Expected path: /content/best.pt")
        return

    st.success("✅ Model loaded successfully!")


    tab1, tab2, tab3 = st.tabs(["📸 Detection", "📊 Analytics", "ℹ️ Instructions"])

    with tab1:
        video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

        if video_file:
          
            col1, col2 = st.columns(2)

            """if platform == "windows":
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                suffix = ".avi"
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mpv4")
                suffix = ".mp4"
            """
                
            with col1:
                st.subheader("🎥 Original Video")
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as temp_vid:
                    temp_vid.write(video_file.read())
                    input_video_path = temp_vid.name

                    with open(input_video_path, 'rb') as in_video_file:
                        in_video_bytes = in_video_file.read()
                        st.video(in_video_bytes)

    
            with col2:

                st.subheader("🎥 Annotated Video")

                if "video_result" not in st.session_state:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    video_bytes, video_stats, compliance = process_video(
                        model,
                        input_video_path,
                        conf_threshold,
                        progress_bar=progress_bar,
                        status_text=status_text
                    )

                    st.session_state.video_result = video_bytes
                    st.session_state.video_stats = video_stats
                    st.session_state.video_compliance = compliance

                    progress_bar.empty()
                    status_text.empty()


                st.video(video_bytes)

                st.download_button(
                    "⬇️ Download output video",
                    data=st.session_state.video_result,
                    file_name="output.mp4",
                    mime="video/mp4"
                )

                """st.session_state.video_processing_done = True
                progress_bar = st.empty()
                status_text = st.empty()"""

            st.markdown("---")
            st.subheader("📊 Statistics")   
            render_analytics(video_stats, compliance)

        uploaded_file = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png", "bmp"]
        )

        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            image = PIL.Image.open(uploaded_file)

            with col1:
                st.subheader("📷 Original Image")
                st.image(image, use_column_width=True)

            with st.spinner("🔍 Analyzing..."):
                results = run_inference(
                    model, image, conf=conf_threshold, iou=iou_threshold
                )  
                stats = analyze_detections(results, conf_threshold)
                compliance = calculate_compliance_rate(stats)

            with col2:
                st.subheader("🎯 Detection Results")
                res_plotted = results[0].plot(labels=show_labels, conf=show_conf)
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st.image(res_plotted_rgb, use_column_width=True)
            
            st.markdown("---")
            st.subheader("📊 Statistics")   
            render_analytics(stats, compliance)

            st.markdown("---")
            if stats["total_detections"] > 0:
                if compliance >= 80:
                    st.success(f"✅ High compliance: {compliance:.1f}%")
                elif compliance >= 50:
                    st.warning(f"⚠️ Moderate compliance: {compliance:.1f}%")
                else:
                    st.error(f"❌ Low compliance: {compliance:.1f}%")

            if stats["detections"]:
                with st.expander("🔍 Detailed Detections"):
                    for i, det in enumerate(stats["detections"], 1):
                        st.write(f"**{i}.** {det['class']} - {det['confidence']:.2%}")

            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                json_data = export_results(stats, uploaded_file.name)
                st.download_button(
                    "📥 Download Report",
                    json_data,
                    f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                )
            with col_exp2:
                img_bytes = io.BytesIO()
                PIL.Image.fromarray(res_plotted_rgb).save(img_bytes, format="PNG")
                st.download_button(
                    "📥 Download Image",
                    img_bytes.getvalue(),
                    f"annotated_{uploaded_file.name}",
                    "image/png",
                )

            st.session_state["stats"] = stats
            st.session_state["compliance"] = compliance

    with tab2:
        st.subheader("📊 Analytics Dashboard")
        if "stats" in st.session_state:
            import pandas as pd

            stats = st.session_state["stats"]
            compliance = st.session_state["compliance"]

            data = {
                "Category": ["With Mask", "Without Mask"],
                "Count": [stats["with_mask"], stats["without_mask"]],
            }
            df = pd.DataFrame(data)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Distribution")
                st.bar_chart(df.set_index("Category"))
            with col2:
                st.subheader("Metrics")
                st.metric("Total Detected", stats["total_detections"])
                st.metric("Compliance Rate", f"{compliance:.1f}%")
        else:
            st.info("📸 Upload an image to view analytics")

    with tab3:
        st.markdown("""
        ### 🚀 Quick Start
        1. Upload an image using the file uploader
        2. Adjust detection settings in the sidebar
        3. View results and download reports

        ### ⚙️ Settings Guide
        - **Confidence (0.5 recommended)**: Higher = fewer false positives
        - **IoU (0.45 recommended)**: Controls overlap filtering

        ### 📊 Compliance Levels
        - High: ≥80% - Excellent
        - Moderate: 50-79% - Acceptable
        - Low: <50% - Needs attention
        """)


if __name__ == "__main__":
    main()

print("✅ app.py created successfully!")
