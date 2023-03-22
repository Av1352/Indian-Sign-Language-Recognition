import numpy as np
import cv2
import os
import sys
import time
import operator

import hand_capture as face
import camera as camera

# #Application :

# hs = Hunspell('en_US')
# cap = cv2.VideoCapture(0)
# current_image = None
# current_image2 = None
# json_file = open("Models\model_new.json", "r")
# model_json = json_file.read()
# json_file.close()

# loaded_model = model_from_json(model_json)
# loaded_model.load_weights("Models\model_new.h5")

face.capture()
img = cv2.imread("user.png")

prediction = camera.camera()
print(prediction)

global sign
sign = prediction
print(sign)
