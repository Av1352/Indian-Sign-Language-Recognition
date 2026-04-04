import numpy as np
import cv2
import tensorflow as tf
import streamlit as st

IMG_SIZE = 100

# Alphabetically sorted - matches image_dataset_from_directory
CLASS_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j',
    20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't',
    30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z'
}

@st.cache_resource
def load_model(model_path="Models/best_model.keras"):
    """Load model with AdamW optimizer handling"""
    try:
        # Try loading with TensorFlow Addons
        import tensorflow_addons as tfa
        custom_objects = {'AdamW': tfa.optimizers.AdamW}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except:
        # Fallback: load without optimizer and recompile
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    return model

@st.cache_data
def preprocess_image(path):
    """
    Preprocess image for model input.
    NOTE: Model has Rescaling(1./255) layer, so we DON'T normalize here!
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")

    print(f"[DEBUG predict] Loaded shape: {img.shape}, min={img.min()}, max={img.max()}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Keep [0-255] range — model's internal Rescaling(1./255) handles normalisation
    img = img.astype("float32")
    print(f"[DEBUG predict] Tensor: shape={img.shape}, min={img.min():.3f}, max={img.max():.3f}")
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def predict(image_path):
    model = load_model()
    img = preprocess_image(image_path)
    pred = model.predict(img, verbose=0)
    top5_idx = np.argsort(pred[0])[::-1][:5]
    print(f"[DEBUG predict] Top-5: {[(CLASS_MAP[i], f'{pred[0][i]*100:.2f}%') for i in top5_idx]}")
    class_idx = int(np.argmax(pred))
    confidence = float(np.max(pred)) * 100
    if class_idx >= len(CLASS_MAP):
        return "Unknown", 0.0
    return CLASS_MAP[class_idx], confidence
