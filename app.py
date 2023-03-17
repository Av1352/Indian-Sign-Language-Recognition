import numpy as np
import cv2
import os
import sys
import time
import operator

from string import ascii_uppercase

import tkinter as tk
from PIL import Image, ImageTk

from hunspell import Hunspell
import enchant

from keras.models import model_from_json

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

#Application :

hs = Hunspell('en_US')
cap = cv2.VideoCapture(0)
current_image = None
current_image2 = None
json_file = open("Models\model_new.json", "r")
model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(model_json)
loaded_model.load_weights("Models\model_new.h5")
