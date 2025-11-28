import os
import cv2
import imageio.v2 as imageio
import mediapipe as mp

class Preprocess:
    def __init__(self):
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands()

    def roi_hand(self, input_img_path, output_img_path):
        """
        Detect hand region using MediaPipe and extract ROI.
        Training data was already cropped to hands, so we do the same.
        """
        img = imageio.imread(input_img_path)
        result = self.hands.process(img)
        hand_landmarks = result.multi_hand_landmarks
        h, w, c = img.shape
        roi = None
        
        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max = y_max = 0
                x_min, y_min = w, h
                
                # Find bounding box of hand
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_max = max(x_max, x)
                    x_min = min(x_min, x)
                    y_max = max(y_max, y)
                    y_min = min(y_min, y)
                
                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Extract ROI
                roi = img[y_min:y_max, x_min:x_max]
            
            if roi is not None:
                os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
                # Convert RGB to BGR for cv2
                cv2.imwrite(output_img_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
                return output_img_path
        
        if roi is None:
            raise ValueError("No hand detected — please upload a clearer image.")

    def preprocess_images(self, input_img_path, output_img_path):
        """
        Simple preprocessing to match training:
        - Convert to grayscale
        - Resize to 100x100
        
        NO edge detection, NO skin masking - training used raw grayscale images!
        The model's data augmentation and normalization layers handle the rest.
        """
        if not os.path.exists(input_img_path):
            raise FileNotFoundError(f"{input_img_path} not found.")
        
        # Read as grayscale
        img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read {input_img_path}.")
        
        # Resize to 100x100 - matches training
        img_resized = cv2.resize(img, (100, 100))
        
        # Save
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        cv2.imwrite(output_img_path, img_resized)
        
        if os.path.exists(output_img_path):
            return output_img_path
        else:
            raise RuntimeError(f"Failed to save preprocessed image")