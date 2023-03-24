import numpy as np
import cv2
import os
import sys
import time
import operator

import hand_capture as face
import camera as camera


face.capture()
img = cv2.imread("user.png")

prediction = camera.camera()
print(prediction)

global sign
sign = prediction
print(sign)
