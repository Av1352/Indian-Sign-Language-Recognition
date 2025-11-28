import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
import os
from huggingface_hub import hf_hub_download
import shutil

IMG_SIZE = 100

CLASS_MAP = {
    i: c for i, c in enumerate(
        ['0','1','2','3','4','5','6','7','8','9',
        'a','b','c','d','e','f','g','h','i','j',
        'k','l','m','n','o','p','q','r','s','t',
        'u','v','w','x','y','z']
    )
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
    
    model = tf.keras.models.load_model(local_model_path, compile=False, safe_mode=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

@st.cache_data
def preprocess_image(path="utils/processed.png"):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def predict(image_path="utils/processed.png"):
    model = load_model()
    img = preprocess_image(image_path)
    pred = model.predict(img, verbose=0)
    class_idx = int(np.argmax(pred))
    confidence = float(np.max(pred)) * 100
    return CLASS_MAP[class_idx], confidence