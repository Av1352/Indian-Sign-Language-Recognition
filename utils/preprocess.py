import os
import numpy as np
import cv2
import imageio.v2 as imageio
import mediapipe as mp

class Preprocess:
    def __init__(self):
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands()

    def roi_hand(self, input_img_path='utils/user.png', output_img_path='utils/roi.png'):
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
                
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                roi = img[y_min:y_max, x_min:x_max]
            
            if roi is not None:
                os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
                cv2.imwrite(output_img_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
                return output_img_path
        
        if roi is None:
            raise ValueError("No hand detected — please upload a clearer image.")

    def preprocess_images(self, input_img_path='utils/roi.png', output_img_path='utils/processed.png'):
        img = cv2.imread(input_img_path)
        if img is None:
            raise FileNotFoundError(f"{input_img_path} not found.")
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        skin_color_lower = np.array([0, 40, 30], np.uint8)
        skin_color_upper = np.array([43, 255, 255], np.uint8)
        skin_mask = cv2.inRange(hsv_img, skin_color_lower, skin_color_upper)
        skin_mask = cv2.medianBlur(skin_mask, 5)
        skin_mask = cv2.addWeighted(skin_mask, 0.5, skin_mask, 0.5, 0.0)
        
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        hand = cv2.bitwise_and(gray_img, gray_img, mask=skin_mask)
        canny = cv2.Canny(hand, 60, 60)
        
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        cv2.imwrite(output_img_path, canny)
        
        if os.path.exists(output_img_path):
            return output_img_path
        else:
            raise RuntimeError(f"Failed to save preprocessed image")