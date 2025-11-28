import os
import numpy as np
import cv2
import imageio.v2 as imageio
import mediapipe as mp

class Preprocess:
    def __init__(self):
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands()

    def roi_hand(self, input_img_path, output_img_path):
        """Detect hand region using MediaPipe"""
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
            
            if roi is not None:
                os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
                cv2.imwrite(output_img_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
                return output_img_path
        
        if roi is None:
            raise ValueError("No hand detected — please upload a clearer image.")

    def preprocess_images(self, input_img_path, output_img_path):
        """MATCH TRAINING: Edge detection on black background"""
        if not os.path.exists(input_img_path):
            raise FileNotFoundError(f"{input_img_path} not found.")
        
        img = cv2.imread(input_img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read {input_img_path}.")
        
        # Resize first
        img = cv2.resize(img, (100, 100))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection (THIS IS WHAT TRAINING HAD!)
        edges = cv2.Canny(gray, 50, 150)
        
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        cv2.imwrite(output_img_path, edges)
        
        if os.path.exists(output_img_path):
            return output_img_path
        else:
            raise RuntimeError(f"Failed to save preprocessed image to {output_img_path}")