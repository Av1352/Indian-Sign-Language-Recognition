import streamlit as st
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
from utils.preprocess import Preprocess
from utils.predict import predict
import pandas as pd
import base64

# --- Utility Functions ---
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# --- Page Config ---
st.set_page_config(
    page_title="Indian Sign Language Recognition",
    page_icon="streamlit_app/logo.png",
    layout="wide"
)

# --- Background Image ---
bg_image_path = "streamlit_app/header.png"
try:
    bg_base64 = get_base64(bg_image_path)
except Exception as e:
    st.error(f"Could not load background image: {e}")
    st.stop()

# --- CSS ---
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

# --- Sidebar ---
st.sidebar.image("streamlit_app/logo.png", width=80)
st.sidebar.title("📊 Model Performance")
st.sidebar.metric("TTA Test Accuracy", "90.6%")

with st.sidebar.expander("Model Details", expanded=True):
    st.markdown("""
- **Architecture:** Custom CNN  
- **Input:** 100x100x1  
- **Epochs:** 30  
- **Metrics:** F1, Confusion Matrix
""")

# --- Classification Report (as a dataframe) ---

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
    "Support": [
        201]*36
}

report_df = pd.DataFrame(report_data)

with st.sidebar.expander("Classification Report", expanded=True):
    st.dataframe(report_df, height=300)

# --- Sidebar Tabs: Predictions & Confusion Matrix ---
st.sidebar.header("Evaluation Visuals")
tab_pred, tab_conf = st.sidebar.tabs(["Predictions", "Confusion Matrix"])

with tab_pred:
    if os.path.exists("files/output.png"):
        st.image("files/output.png", caption="Model Predictions", use_container_width=True)

with tab_conf:
    if os.path.exists("files/conf_matrix.png"):
        st.image("files/conf_matrix.png", caption="Confusion Matrix", use_container_width=True)


# --- Main Panel ---
st.image("streamlit_app/logo.png", width=100)
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

pre = Preprocess()
utils_dir = os.path.join(os.getcwd(), "utils")
os.makedirs(utils_dir, exist_ok=True)

input_image_path = None
if uploaded_file:
    input_image_path = os.path.join(utils_dir, "user.png")
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
elif camera_img:
    input_image_path = os.path.join(utils_dir, "user.png")
    img = Image.open(camera_img)
    img.save(input_image_path)

if input_image_path:
    st.subheader("Input Image")
    st.image(input_image_path, caption="Uploaded Input", width=400)

    roi_img_path = os.path.join(utils_dir, "roi.png")
    processed_img_path = os.path.join(utils_dir, "processed.png")
    gradcam_img_path = os.path.join(utils_dir, "gradcam.png")

    pre.roi_hand(input_img_path=input_image_path, output_img_path=roi_img_path)
    pre.preprocess_images(input_img_path=roi_img_path, output_img_path=processed_img_path)

    st.subheader("Region of Interest (ROI)")
    st.image(roi_img_path, caption="Detected Hand Region", width=400)

    st.subheader("Preprocessed Image")
    st.image(processed_img_path, caption="Model Input", width=400)

    if os.path.exists(gradcam_img_path):
        st.subheader("Grad-CAM Visualization")
        st.image(gradcam_img_path, caption="Model Focus", width=400)

    label, confidence = predict(image_path=processed_img_path)
    st.success(f"Predicted Sign: **{label}** (Confidence: {confidence:.2f})")
else:
    st.info("Please upload an image or take a picture to continue.")

st.markdown("---")
st.button("GET STARTED!", help="Begin prediction!")
st.markdown("#### Made with ❤️ to bridge the hearing/non-hearing gap.")