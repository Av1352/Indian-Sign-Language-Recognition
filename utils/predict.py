import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
import os

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
    # Get the absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple possible paths
    possible_paths = [
        model_path,  # Relative path
        os.path.join(current_dir, "..", model_path),  # From utils/ directory
        os.path.join("/mount/src/indian-sign-language-recognition", model_path),  # Absolute Streamlit path
        os.path.abspath(model_path),  # Absolute from current
    ]
    
    model_file = None
    for path in possible_paths:
        st.write(f"Checking path: {path}")  # Debug output
        if os.path.exists(path):
            model_file = path
            st.write(f"✅ Found model at: {path}")
            break
    
    if model_file is None:
        # List what's actually available
        st.error("Model file not found! Checking available files...")
        st.write(f"Current directory: {os.getcwd()}")
        st.write(f"Files in current dir: {os.listdir('.')}")
        if os.path.exists('Models'):
            st.write(f"Files in Models/: {os.listdir('Models')}")
        raise FileNotFoundError(f"Could not find model at any of these paths: {possible_paths}")
    
    try:
        # Try to import tensorflow_addons for AdamW
        import tensorflow_addons as tfa
        custom_objects = {'AdamW': tfa.optimizers.AdamW, 'Addons>AdamW': tfa.optimizers.AdamW}
        model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)
        return model
    except ImportError:
        model = tf.keras.models.load_model(model_file, compile=False)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        model = tf.keras.models.load_model(model_file, compile=False)
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