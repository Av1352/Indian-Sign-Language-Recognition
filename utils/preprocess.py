import os
import cv2
import numpy as np
import mediapipe as mp
import imageio.v2 as imageio


class Preprocess:
    def __init__(self):
        self.mphands = mp.solutions.hands
        self.hands   = self.mphands.Hands(
            static_image_mode        = True,
            max_num_hands            = 1,
            min_detection_confidence = 0.3,
        )

    def roi_hand(self, input_img_path, output_img_path):
        """No-op shim — landmark pipeline needs no ROI crop. Copies input to output."""
        img = cv2.imread(input_img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read {input_img_path}")
        out_dir = os.path.dirname(output_img_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_img_path, img)
        return output_img_path

    def preprocess_images(self, input_img_path, output_img_path):
        """No-op shim — landmark pipeline needs no Canny preprocessing. Copies input to output."""
        img = cv2.imread(input_img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read {input_img_path}")
        out_dir = os.path.dirname(output_img_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_img_path, img)
        return output_img_path
