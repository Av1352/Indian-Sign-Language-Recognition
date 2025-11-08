import streamlit as st
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
from utils.preprocess import Preprocess
from utils.predict import predict
import base64

def get_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

img_file = 'streamlit_app/header.png'

try:
    bg_base64 = get_base64(img_file)
except Exception as e:
    st.error(f"Could not encode {img_file}: {e}")
    st.stop()
    
# --- THEME: Red + Dark (matches Flask/About) ---
st.set_page_config(page_title="Indian Sign Language Recognition", page_icon="streamlit_app/logo.png", layout="wide")
st.markdown(
    f"""
    <style>
    /* Use header image as full background, cover entire viewport */
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .overlay-section {{
        background: rgba(162, 61, 54, 0.88);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
    }}
    h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stText {{
        color: #fff !important;
    }}
    div.stButton > button {{
        background-color: #a23d36;
        color: #fff;
        font-weight: bold;
        border-radius: 25px;
        font-size: 18px;
    }}
    [data-testid="stSidebar"] {{
        background-color: #212529;
        color: #fff;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# --- SIDEBAR: Metrics and Plots ---
st.sidebar.image("streamlit_app/logo.png", width=80)
st.sidebar.title("📊 Model Performance")
st.sidebar.metric("Test Accuracy", "93.4%")
st.sidebar.metric("ROC AUC", "0.96")

with st.sidebar.expander("Model Details", expanded=True):
    st.markdown("""
    **Architecture:** Custom CNN  
    **Input:** 64x64x3  
    **Epochs:** 25  
    **Metrics:** F1, ROC, Confusion Matrix
    """)

st.sidebar.header("Evaluation Visuals")
tab_roc, tab_conf = st.sidebar.tabs(["ROC Curve", "ConfMatrix"])
if os.path.exists("files/roc_curve.png"):
    with tab_roc:
        st.image("files/roc_curve.png", caption="ROC Curve", use_container_width=True)
if os.path.exists("files/conf_matrix.png"):
    with tab_conf:
        st.image("files/conf_matrix.png", caption="Confusion Matrix", use_container_width=True)
if os.path.exists("files/accuracy plot.png"):
    st.sidebar.image("files/accuracy plot.png", caption="Accuracy Curve", use_container_width=True)
if os.path.exists("files/loss - cnn - 99.png"):
    st.sidebar.image("files/loss - cnn - 99.png", caption="Loss Curve", use_container_width=True)

# --- MAIN PANEL: App logic ---
st.image("streamlit_app/logo.png", width=100)
st.title("INDIAN SIGN LANGUAGE RECOGNITION SYSTEM")
st.write("We can help you communicate with the world.")
st.header("About")
st.write("Our aim is to create sign language recognition system which will detect the sign performed by the users, identify the gesture and then proceed to print the prediction.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("## Upload or Capture Hand Image")

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
    st.markdown("### Input Image")
    st.image(input_image_path, caption="Uploaded Input", use_container_width=True)

    roi_img_path = os.path.join(utils_dir, "roi.png")
    processed_img_path = os.path.join(utils_dir, "processed.png")
    gradcam_img_path = os.path.join(utils_dir, "gradcam.png")

    # Preprocessing pipeline
    pre.roi_hand(input_img_path=input_image_path, output_img_path=roi_img_path)
    pre.preprocess_images(input_img_path=roi_img_path, output_img_path=processed_img_path)

    st.markdown("### Region of Interest (ROI)")
    st.image(roi_img_path, caption="Detected Hand Region", use_container_width=True)

    st.markdown("### Preprocessed Image (Model Input)")
    st.image(processed_img_path, caption="Edge-detected Hand", use_container_width=True)

    # Grad-CAM display if available
    if os.path.exists(gradcam_img_path):
        st.markdown("### Grad-CAM Visualization")
        st.image(gradcam_img_path, caption="Model Focus (Grad-CAM)", use_container_width=True)

    # Prediction
    prediction = predict(image_path=processed_img_path)
    st.success(f"The predicted sign is: **{prediction}**")

else:
    st.info("Please upload an image or take a picture to continue.")

st.markdown("---")
st.button("GET STARTED!", help="Begin prediction!", type="primary")

st.markdown("#### Made with ❤️ to bridge the hearing/non-hearing gap.")

