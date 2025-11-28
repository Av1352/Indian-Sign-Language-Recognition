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
        try:
            hf_model_path = hf_hub_download(
                repo_id="Av1352/indian-sign-language-model",
                filename="best_model.h5"  # Changed to .h5
            )
            shutil.copy2(hf_model_path, local_model_path)
            st.success(f"Model downloaded ({os.path.getsize(local_model_path) / (1024*1024):.1f} MB)")
        except Exception as e:
            st.error(f"Download failed: {e}")
            raise
    
    try:
        import tensorflow_addons as tfa
        custom_objects = {'AdamW': tfa.optimizers.AdamW}
        model = tf.keras.models.load_model(local_model_path, custom_objects=custom_objects)
        return model
    except:
        model = tf.keras.models.load_model(local_model_path, compile=False)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

@st.cache_data
def preprocess_image(path="utils/processed.png"):
    # Read the preprocessed image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    
    # Make sure it's 100x100
    if img.shape != (100, 100):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Debug: Check image statistics
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image min/max: {img.min()}/{img.max()}")
    print(f"Image mean: {img.mean():.2f}")
    
    # Normalize to [0, 1]
    img = img.astype("float32") / 255.0
    
    # Add channel dimension and batch dimension
    img = np.expand_dims(img, axis=-1)   # (100, 100, 1)
    img = np.expand_dims(img, axis=0)    # (1, 100, 100, 1)
    
    return img

def predict(image_path="utils/processed.png"):
    model = load_model()
    img = preprocess_image(image_path)
    
    # Show the preprocessed image in the app for debugging
    st.subheader("Model Input (After Normalization)")
    display_img = (img[0, :, :, 0] * 255).astype(np.uint8)
    st.image(display_img, caption="What the model sees", width=200)
    
    pred = model.predict(img, verbose=0)
    
    # Show all predictions for debugging
    st.subheader("All Predictions")
    top_5_idx = np.argsort(pred[0])[-5:][::-1]
    for idx in top_5_idx:
        st.write(f"{CLASS_MAP[idx]}: {pred[0][idx]*100:.2f}%")
    
    class_idx = int(np.argmax(pred))
    confidence = float(np.max(pred)) * 100
    
    return CLASS_MAP[class_idx], confidence