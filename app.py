import numpy as np
import cv2
import os
import sys
import time
import operator
from flask import Flask
from flask import render_template, request, url_for, redirect, session, make_response, flash
import capture as hand
import preprocess as preprocess
from predict import predict

app = Flask(__name__)
app_root = os.path.abspath(os.path.dirname(__file__))

app.secret_key = os.urandom(10)


@app.route('/', methods=['GET', 'POST'])
def index():

    return render_template('index.html')

@app.route('/click')
def capture_image():
    hand.capture()
    # img = cv2.imread("user.png")
    preprocess.roi_hand()
    preprocess.preprocess_images()
    prediction = predict()
    print(prediction)

    return render_template('index.html', item=prediction)


if __name__ == '__main__':
    app.run(debug=True)
