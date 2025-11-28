import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
import os

IMG_SIZE = 100

CLASS_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j',
    20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't',
    30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z'
}

@st.cache_resource
def load_model(model_path="Models/best_model_inference.keras"):
    # DEBUG INFO
    abs_path = os.path.abspath(model_path)
    file_size = os.path.getsize(abs_path) / (1024 * 1024)
    
    st.write(f"🔍 Loading model from: {abs_path}")
    st.write(f"📊 File size: {file_size:.2f} MB")
    st.write(f"✅ File exists: {os.path.exists(abs_path)}")
    
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # CHECK MODEL LAYERS
    st.write(f"📋 Model has {len(model.layers)} layers")
    st.write(f"🔹 First layer: {model.layers[0].name}")
    st.write(f"🔹 Second layer: {model.layers[1].name}")
    
    # Should NOT have data_augmentation
    has_augmentation = any('augmentation' in layer.name for layer in model.layers)
    if has_augmentation:
        st.error("⚠️ MODEL STILL HAS AUGMENTATION LAYER!")
    else:
        st.success("✅ No augmentation layer found")
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

@st.cache_data
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def predict(image_path):
    model = load_model()
    img = preprocess_image(image_path)
    pred = model.predict(img, verbose=0)
    class_idx = int(np.argmax(pred))
    confidence = float(np.max(pred)) * 100
    return CLASS_MAP[class_idx], confidence