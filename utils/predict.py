import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
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
def load_model(model_path="Models/best_model.keras"):
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_data
def preprocess_image(path="utils/processed.png"):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)   # (100,100,1)
    img = np.expand_dims(img, axis=0)    # (1,100,100,1)
    return img

def predict(image_path="utils/processed.png", model_path="Models/best_model.keras"):
    model = load_model(model_path)
    img = preprocess_image(image_path)
    pred = model.predict(img)
    class_idx = int(np.argmax(pred))
    confidence = float(np.max(pred))
    return CLASS_MAP[class_idx], confidence