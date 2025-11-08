import numpy as np
import cv2
import os
import sys
import time
import operator
from flask import Flask
from flask import render_template, request, url_for, redirect, session, make_response, flash
from utils.capture import capture
from utils.preprocess import Preprocess
from utils.predict import predict


app = Flask(__name__)
app_root = os.path.abspath(os.path.dirname(__file__))

app.secret_key = os.urandom(10)

pre = Preprocess()

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/click')
def capture_image():
    hand.capture()
    # img = cv2.imread("user.png")
    pre.roi_hand()
    pre.preprocess_images()
    global prediction
    prediction = predict()

    print(prediction)
    print(type(prediction))


    return render_template('index.html', item=prediction)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if file and file.filename:
        filepath = os.path.join('utils', 'user.png')
        file.save(filepath)
        # Optionally run preprocess/predict here or redirect to main
        # hand.capture()
        # img = cv2.imread("user.png")
        pre.roi_hand()
        pre.preprocess_images()
        global prediction
        prediction = predict()

        print(prediction)
        print(type(prediction))

        return render_template('index.html', item=prediction)

if __name__ == '__main__':
    app.run(debug=True)
