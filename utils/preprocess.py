import os
import cv2
import imageio.v2 as imageio
import mediapipe as mp
import numpy as np

class Preprocess:
    def __init__(self):
        # self.mphands = mp.solutions.hands
        # self.hands = self.mphands.Hands()
        pass

    # def roi_hand(self, input_img_path, output_img_path):
    #     """
    #     Detect hand region using MediaPipe and extract ROI.
    #     """
    #     img = imageio.imread(input_img_path)
    #     result = self.hands.process(img)
    #     hand_landmarks = result.multi_hand_landmarks
    #     h, w, c = img.shape
    #     roi = None
        
    #     if hand_landmarks:
    #         for handLMs in hand_landmarks:
    #             x_max = y_max = 0
    #             x_min, y_min = w, h
                
    #             # Find bounding box of hand
    #             for lm in handLMs.landmark:
    #                 x, y = int(lm.x * w), int(lm.y * h)
    #                 x_max = max(x_max, x)
    #                 x_min = min(x_min, x)
    #                 y_max = max(y_max, y)
    #                 y_min = min(y_min, y)
                
    #             # Add padding
    #             padding = 20
    #             x_min = max(0, x_min - padding)
    #             y_min = max(0, y_min - padding)
    #             x_max = min(w, x_max + padding)
    #             y_max = min(h, y_max + padding)
                
    #             # Extract ROI
    #             roi = img[y_min:y_max, x_min:x_max]
    #             break  # Use first detected hand
            
    #         if roi is not None and roi.size > 0:
    #             os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    #             # Convert RGB to BGR for cv2
    #             roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    #             success = cv2.imwrite(output_img_path, roi_bgr)
                
    #             if success and os.path.exists(output_img_path):
    #                 return output_img_path
    #             else:
    #                 raise RuntimeError(f"Failed to save ROI to {output_img_path}")
        
    #     raise ValueError("No hand detected — please upload a clearer image.")
    
    def roi_hand(self, input_img_path, output_img_path):
        img = cv2.imread(input_img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read {input_img_path}")

        h, w, _ = img.shape

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 40, 30], np.uint8)
        upper = np.array([43, 255, 255], np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No hand region found for ROI.")

        cnt = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(cnt)

        padding = 20
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(img.shape[1], x + w_box + padding)
        y_max = min(img.shape[0], y + h_box + padding)

        roi = img[y_min:y_max, x_min:x_max]
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        if not cv2.imwrite(output_img_path, roi):
            raise RuntimeError(f"Failed to save ROI to {output_img_path}")
        return output_img_path

    # def preprocess_images(self, input_img_path, output_img_path):
    #     """Match training: do all processing FIRST, then resize at the end if needed"""
    #     if not os.path.exists(input_img_path):
    #         raise FileNotFoundError(f"{input_img_path} not found.")
        
    #     # Read image (don't resize yet!)
    #     img = cv2.imread(input_img_path)
    #     if img is None or img.size == 0:
    #         raise FileNotFoundError(f"Could not read {input_img_path}.")
        
    #     # Converting image to grayscale
    #     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    #     # Converting image to HSV format
    #     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
    #     # Defining boundary level for skin color in HSV
    #     skin_color_lower = np.array([0, 40, 30], np.uint8)
    #     skin_color_upper = np.array([43, 255, 255], np.uint8)
        
    #     # Producing mask
    #     skin_mask = cv2.inRange(hsv_img, skin_color_lower, skin_color_upper)
        
    #     # Removing Noise from mask
    #     skin_mask = cv2.medianBlur(skin_mask, 5)
    #     skin_mask = cv2.addWeighted(skin_mask, 0.5, skin_mask, 0.5, 0.0)
        
    #     # Applying Morphological operations
    #     kernel = np.ones((5, 5), np.uint8)
    #     skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
    #     # Extracting hand by applying mask
    #     hand = cv2.bitwise_and(gray_img, gray_img, mask=skin_mask)
        
    #     # Get edges by Canny edge detection
    #     canny = cv2.Canny(hand, 60, 60)
    #     kernel = np.ones((2, 2), np.uint8)
    #     canny = cv2.dilate(canny, kernel, iterations=1)
        
    #     # NOW resize to 100x100 (after all processing)
    #     canny = cv2.resize(canny, (100, 100))
        
    #     # Save
    #     os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    #     success = cv2.imwrite(output_img_path, canny)
        
    #     if success and os.path.exists(output_img_path):
    #         return output_img_path
    #     else:
    #         raise RuntimeError(f"Failed to save preprocessed image")
    
    def preprocess_images(self, input_img_path, output_img_path):
        if not os.path.exists(input_img_path):
            raise FileNotFoundError(f"{input_img_path} not found.")

        # Read as BGR (cv2 default) — equivalent to training's imageio RGB read
        # when paired with COLOR_BGR2GRAY / COLOR_BGR2HSV below
        img = cv2.imread(input_img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read {input_img_path}.")

        print(f"[DEBUG preprocess] ROI shape: {img.shape}, min={img.min()}, max={img.max()}")

        # Replicate training pipeline exactly (preprocessing.ipynb cell 4)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv_img  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Primary range: covers light-to-medium skin tones
        skin_mask  = cv2.inRange(hsv_img, np.array([0,  20,  70], np.uint8),
                                          np.array([20, 255, 255], np.uint8))
        # Secondary range: covers darker/deeper skin tones (hue wraps near 180)
        skin_mask2 = cv2.inRange(hsv_img, np.array([160, 20,  70], np.uint8),
                                          np.array([180, 255, 255], np.uint8))
        skin_mask  = cv2.bitwise_or(skin_mask, skin_mask2)

        skin_mask = cv2.medianBlur(skin_mask, 5)
        skin_mask = cv2.addWeighted(skin_mask, 0.5, skin_mask, 0.5, 0.0)

        kernel    = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        hand  = cv2.bitwise_and(gray_img, gray_img, mask=skin_mask)
        canny = cv2.Canny(hand, 60, 60)

        # Resize AFTER Canny — matches how image_dataset_from_directory
        # resized the training Canny images to (100, 100) during model training
        canny = cv2.resize(canny, (100, 100))

        print(f"[DEBUG preprocess] After Canny+resize: shape={canny.shape}, min={canny.min()}, max={canny.max()}")

        out_dir = os.path.dirname(output_img_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if not cv2.imwrite(output_img_path, canny):
            raise RuntimeError(f"Failed to save preprocessed image to {output_img_path}")
        return output_img_path
