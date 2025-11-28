import os
import cv2
import imageio.v2 as imageio
import mediapipe as mp
import numpy as np

class Preprocess:
    def __init__(self):
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands()

    def roi_hand(self, input_img_path, output_img_path):
        """Detect hand region using MediaPipe and extract ROI."""
        img = imageio.imread(input_img_path)
        result = self.hands.process(img)
        hand_landmarks = result.multi_hand_landmarks
        h, w, c = img.shape
        roi = None
        
        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max = y_max = 0
                x_min, y_min = w, h
                
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_max = max(x_max, x)
                    x_min = min(x_min, x)
                    y_max = max(y_max, y)
                    y_min = min(y_min, y)
                
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                roi = img[y_min:y_max, x_min:x_max]
                break
            
            if roi is not None and roi.size > 0:
                os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
                roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(output_img_path, roi_bgr)
                
                if success and os.path.exists(output_img_path):
                    return output_img_path
                else:
                    raise RuntimeError(f"Failed to save ROI")
        
        raise ValueError("No hand detected — please upload a clearer image.")

    def preprocess_images(self, input_img_path, output_img_path):
        """
        Enhanced preprocessing to bridge training/real-world gap:
        - Grayscale conversion
        - Histogram equalization (normalize lighting)
        - Resize to 100x100
        - Optional: slight blur to reduce noise
        """
        if not os.path.exists(input_img_path):
            raise FileNotFoundError(f"{input_img_path} not found.")
        
        img = cv2.imread(input_img_path)
        if img is None or img.size == 0:
            raise FileNotFoundError(f"Could not read {input_img_path}.")
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply histogram equalization to normalize lighting
        # This makes bright and dark images more consistent
        gray = cv2.equalizeHist(gray)
        
        # Apply slight Gaussian blur to reduce noise/texture differences
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Resize to 100x100
        img_resized = cv2.resize(gray, (100, 100))
        
        # Save
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        success = cv2.imwrite(output_img_path, img_resized)
        
        if success and os.path.exists(output_img_path):
            return output_img_path
        else:
            raise RuntimeError(f"Failed to save preprocessed image")