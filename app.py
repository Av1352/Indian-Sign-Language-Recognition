import numpy as np
import cv2
import os
import sys
import time
import operator

import capture as hand
import preprocess as preprocess
from predict import predict

def capture_image():
    hand.capture()
    # img = cv2.imread("user.png")

def preprocess_image():
    preprocess.preprocess_images()

def predict_sign():
    prediction = predict()
    print(prediction)

    global sign
    sign = prediction
    print(sign)

def main():
    capture_image()
    preprocess_image()
    predict_sign()

if __name__ == '__main__':
    main()
