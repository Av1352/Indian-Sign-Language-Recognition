import streamlit as st
import os
import tempfile
import uuid
from PIL import Image
from utils.preprocess import Preprocess
from utils.predict import predict
import pandas as pd
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Utility function for base64 encoding
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Page config
st.set_page_config(
    page_title="Indian Sign Language Recognition",
    page_icon="assets/logo.png",
    layout="wide"
)

# Background image
bg_image_path = "assets/header.png"
try:
    bg_base64 = get_base64(bg_image_path)
except Exception as e:
    st.error(f"Could not load background image: {e}")
    st.stop()

# CSS styling
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{bg_base64}");
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
}}

h1, h2, h3, h4, h5, h6, p {{
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}}

div.stButton > button {{
    background-color: #a23d36;
    color: #fff;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
    font-size: 16px;
    transition: transform 0.2s;
}}
div.stButton > button:hover {{
    transform: scale(1.05);
}}

[data-testid="stSidebar"] {{
    background-color: rgba(33, 37, 41, 0.95);
    color: #fff;
    padding: 1rem;
    border-right: 1px solid #222;
}}

img {{
    border-radius: 6px;
}}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("assets/logo.png", width=80)
st.sidebar.title("📊 Model Performance")
st.sidebar.metric("Validation Accuracy", "91.3%")

with st.sidebar.expander("Model Details", expanded=True):
    st.markdown("""
- **Architecture:** Custom CNN  
- **Input:** 100x100x1 grayscale
- **Classes:** 36 (0-9, a-z)
- **Inference Model:** No augmentation
""")

# Classification report
report_data = {
    "Class": [
        '0','1','2','3','4','5','6','7','8','9',
        'a','b','c','d','e','f','g','h','i','j',
        'k','l','m','n','o','p','q','r','s','t',
        'u','v','w','x','y','z'
    ],
    "Precision": [
        0.84,0.95,0.81,0.88,0.93,0.96,0.99,0.82,0.74,0.95,
        1.00,1.00,0.98,0.99,0.71,0.93,0.93,0.99,0.99,0.99,
        0.98,1.00,0.97,0.77,0.99,0.95,0.71,0.90,0.79,0.98,
        0.73,0.98,1.00,0.99,0.98,1.00
    ],
    "Recall": [
        0.96,0.97,0.90,0.99,0.93,0.95,0.94,0.91,1.00,0.84,
        0.98,0.69,0.89,0.96,0.85,0.70,0.95,0.93,0.87,0.76,
        0.95,0.90,0.73,0.98,0.75,0.99,1.00,0.79,1.00,0.97,
        1.00,0.81,0.99,0.99,0.92,0.95
    ],
    "F1-Score": [
        0.90,0.96,0.85,0.93,0.93,0.95,0.96,0.86,0.85,0.89,
        0.99,0.82,0.93,0.98,0.77,0.80,0.94,0.96,0.92,0.86,
        0.97,0.95,0.83,0.86,0.86,0.97,0.83,0.84,0.89,0.97,
        0.84,0.88,0.99,0.99,0.95,0.97
    ],
    "Support": [201]*36
}

report_df = pd.DataFrame(report_data)

with st.sidebar.expander("Classification Report", expanded=False):
    st.dataframe(report_df, height=300)

# Sidebar images
st.sidebar.header("Evaluation Visuals")
tab_pred, tab_conf = st.sidebar.tabs(["Predictions", "Confusion Matrix"])

with tab_pred:
    if os.path.exists("files/output.png"):
        st.image("files/output.png", caption="Model Predictions", use_container_width=True)

with tab_conf:
    if os.path.exists("files/conf_matrix.png"):
        st.image("files/conf_matrix.png", caption="Confusion Matrix", use_container_width=True)

# Main content
st.image("assets/logo.png", width=100)
st.title("INDIAN SIGN LANGUAGE RECOGNITION SYSTEM")
st.write("Helping you communicate with the world.")

st.header("About")
st.write("""
Our system detects hand gestures and predicts corresponding sign language symbols.
Designed to help bridge communication between hearing and non-hearing individuals.
""")

st.header("Upload or Capture Hand Image")
uploaded_file = st.file_uploader("Upload a hand sign image", type=["png", "jpg", "jpeg"])
camera_img = st.camera_input("Or take a picture")

# Initialize preprocessing
pre = Preprocess()

# Session-specific directory
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_dir = os.path.join("temp_images", st.session_state.session_id)
os.makedirs(session_dir, exist_ok=True)

input_image_path = None
if uploaded_file:
    input_image_path = os.path.join(session_dir, "user.png")
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
elif camera_img:
    input_image_path = os.path.join(session_dir, "user.png")
    img = Image.open(camera_img)
    img.save(input_image_path)

if input_image_path:
    st.subheader("Input Image")
    st.image(input_image_path, caption="Uploaded Input", width=400)

    roi_img_path = os.path.join(session_dir, "roi.png")
    processed_img_path = os.path.join(session_dir, "processed.png")
    
    # Clean up old files
    for old_file in [roi_img_path, processed_img_path]:
        if os.path.exists(old_file):
            os.remove(old_file)
    
    try:
        # Process image
        saved_roi_path = pre.roi_hand(input_img_path=input_image_path, output_img_path=roi_img_path)
        
        if not os.path.exists(roi_img_path):
            st.error("Failed to detect hand region. Please try a clearer image.")
            st.stop()
            
        st.subheader("Region of Interest (ROI)")
        st.image(roi_img_path, caption="Detected Hand Region", width=400)

        saved_processed_path = pre.preprocess_images(input_img_path=roi_img_path, output_img_path=processed_img_path)
        
        if not os.path.exists(processed_img_path):
            st.error("Failed to preprocess image.")
            st.stop()
            
        st.subheader("Preprocessed Image")
        st.image(processed_img_path, caption="Model Input", width=400)
        
        st.subheader("🔬 Debug: What the model sees")

        # Read the processed image
        debug_img = cv2.imread(processed_img_path, cv2.IMREAD_GRAYSCALE)
        st.write(f"Image shape: {debug_img.shape}")
        st.write(f"Pixel range: {debug_img.min()} to {debug_img.max()}")
        st.write(f"Mean pixel value: {debug_img.mean():.1f}")
        st.write(f"Non-zero pixels: {np.count_nonzero(debug_img)}/{debug_img.size}")

        # Show a histogram
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.hist(debug_img.flatten(), bins=50)
        ax.set_title("Pixel Distribution")
        st.pyplot(fig)

        # Predict
        label, confidence = predict(image_path=processed_img_path)
        st.success(f"Predicted Sign: **{label}** (Confidence: {confidence:.2f}%)")
        
    except ValueError as e:
        st.error(f"Detection error: {str(e)}")
    except FileNotFoundError as e:
        st.error(f"File error: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload an image or take a picture to continue.")

st.markdown("---")
st.markdown("#### Made with ❤️ to bridge the hearing/non-hearing gap.")