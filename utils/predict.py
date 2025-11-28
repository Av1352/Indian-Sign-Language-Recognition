import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
import os
from huggingface_hub import hf_hub_download
import shutil

IMG_SIZE = 100

# EXACT order from training - alphabetically sorted
CLASS_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j',
    20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't',
    30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z'
}

@st.cache_resource
def load_model():
    local_model_path = "best_model.h5"
    
    if not os.path.exists(local_model_path):
        hf_model_path = hf_hub_download(
            repo_id="Av1352/indian-sign-language-model",
            filename="best_model.h5"
        )
        shutil.copy2(hf_model_path, local_model_path)
        st.info(f"Model downloaded: {os.path.getsize(local_model_path) / (1024*1024):.1f} MB")
    
    model = tf.keras.models.load_model(local_model_path, compile=False, safe_mode=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Debug model info
    st.write(f"**Model loaded:** Input shape: {model.input_shape}, Output shape: {model.output_shape}")
    
    return model

@st.cache_data
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    
    # Debug: Show original image stats
    st.write(f"**Original image:** Shape: {img.shape}, Min: {img.min()}, Max: {img.max()}, Mean: {img.mean():.1f}")
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Debug: Show after resize
    st.write(f"**After resize:** Shape: {img.shape}, Non-zero pixels: {np.count_nonzero(img)}/{img.size}")
    
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    
    return img

def predict(image_path):
    model = load_model()
    img = preprocess_image(image_path)
    
    pred = model.predict(img, verbose=0)
    
    # FULL DEBUG OUTPUT
    st.write("**🔍 Full Prediction Analysis:**")
    st.write(f"Prediction array shape: {pred.shape}")
    st.write(f"Sum of predictions: {pred.sum():.4f} (should be ~1.0)")
    st.write(f"Min prediction: {pred.min():.6f}")
    st.write(f"Max prediction: {pred.max():.6f}")
    
    # Show ALL predictions
    st.write("**All 36 class probabilities:**")
    for i in range(len(CLASS_MAP)):
        st.write(f"{i}: '{CLASS_MAP[i]}' = {pred[0][i]*100:.4f}%")
    
    # Show top 5
    st.write("**Top 5 Predictions:**")
    top_5_idx = np.argsort(pred[0])[-5:][::-1]
    for idx in top_5_idx:
        st.write(f"✓ {CLASS_MAP[idx]}: {pred[0][idx]*100:.2f}%")
    
    class_idx = int(np.argmax(pred))
    confidence = float(np.max(pred)) * 100
    
    return CLASS_MAP[class_idx], confidence