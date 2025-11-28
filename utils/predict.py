import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
import os
import requests

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
    # Check if model exists and is actually a model file (not a Git LFS pointer)
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        # Git LFS pointer files are tiny (< 200 bytes)
        if file_size < 500:
            st.warning(f"Model file appears to be a Git LFS pointer ({file_size} bytes). Downloading actual model...")
            # Download from GitHub's LFS
            url = "https://github.com/Av1352/Indian-Sign-Language-Recognition/raw/main/Models/best_model.keras"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Model downloaded successfully!")
            else:
                st.error(f"Failed to download model: {response.status_code}")
                raise FileNotFoundError("Could not download model from GitHub")
    else:
        st.error(f"Model file not found at {model_path}")
        st.write(f"Current directory: {os.getcwd()}")
        st.write(f"Directory contents: {os.listdir('.')}")
        if os.path.exists('Models'):
            st.write(f"Models directory: {os.listdir('Models')}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Try to import tensorflow_addons for AdamW
        import tensorflow_addons as tfa
        custom_objects = {'AdamW': tfa.optimizers.AdamW, 'Addons>AdamW': tfa.optimizers.AdamW}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        return model
    except ImportError:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer='adam',
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
    img = np.expand_dims(img, axis=-1)   # (100,100,1)
    img = np.expand_dims(img, axis=0)    # (1,100,100,1)
    return img

def predict(image_path="utils/processed.png", model_path="Models/best_model.keras"):
    model = load_model(model_path)
    img = preprocess_image(image_path)
    pred = model.predict(img, verbose=0)
    class_idx = int(np.argmax(pred))
    confidence = float(np.max(pred)) * 100
    return CLASS_MAP[class_idx], confidence