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
    local_model_path = "local_best_model.keras"
    
    try:
        # Download from Hugging Face
        hf_model_path = hf_hub_download(
            repo_id="Av1352/indian-sign-language-model",
            filename="best_model.keras"
        )
        
        # Copy to a local path with proper permissions
        shutil.copy2(hf_model_path, local_model_path)
        st.success(f"Model copied successfully ({os.path.getsize(local_model_path)} bytes)")
        
    except Exception as e:
        st.error(f"Failed to download from Hugging Face: {e}")
        raise
    
    # Now try loading from the local copy
    try:
        import tensorflow_addons as tfa
        custom_objects = {'AdamW': tfa.optimizers.AdamW, 'Addons>AdamW': tfa.optimizers.AdamW}
        model = tf.keras.models.load_model(local_model_path, custom_objects=custom_objects)
        st.success("✅ Model loaded successfully!")
        return model
    except ImportError:
        st.warning("Loading without TensorFlow Addons...")
        model = tf.keras.models.load_model(local_model_path, compile=False)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Loading error: {str(e)}")
        # Try one more time with compile=False
        model = tf.keras.models.load_model(local_model_path, compile=False)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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