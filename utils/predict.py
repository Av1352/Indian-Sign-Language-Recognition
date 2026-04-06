import cv2
import joblib
import numpy as np
import mediapipe as mp
import streamlit as st

LANDMARK_MODEL_PATH = "NEW_Models/landmark_model.pkl"


@st.cache_resource
def load_landmark_model(model_path=LANDMARK_MODEL_PATH):
    payload = joblib.load(model_path)
    return payload["model"], payload["label_encoder"]


def extract_landmarks(image_path):
    """
    Run MediaPipe Hands on image_path, return a (1, 42) float32 array of
    bbox-normalised landmark coordinates — same pipeline as extract_landmarks.py.
    Raises ValueError if no hand is detected.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode        = True,
        max_num_hands            = 1,
        min_detection_confidence = 0.3,
    ) as hands:
        result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        raise ValueError("No hand detected — please upload a clearer image.")

    landmarks = [(lm.x, lm.y) for lm in result.multi_hand_landmarks[0].landmark]

    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    features = []
    for x, y in landmarks:
        features.append((x - x_min) / x_range)
        features.append((y - y_min) / y_range)

    return np.array(features, dtype=np.float32).reshape(1, -1)


def predict(image_path):
    clf, le = load_landmark_model()
    features = extract_landmarks(image_path)
    proba      = clf.predict_proba(features)[0]
    class_idx  = int(np.argmax(proba))
    confidence = float(proba[class_idx]) * 100
    label      = le.inverse_transform([class_idx])[0]
    return label, confidence
