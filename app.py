import numpy as np
import cv2
import os
import sys
import time
import operator

import capture as hand
import predict as predict

hand.capture()
img = cv2.imread("user.png")

prediction = predict.predict()
print(prediction)

global sign
sign = prediction
print(sign)
