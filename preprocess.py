import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import imageio.v2 as imageio
import mediapipe as mp

mphands = mp.solutions.hands
hands = mphands.Hands()
handCascade = mp.solutions.drawing_utils

def roi_hand():
    image = 'user.png'
    img = imageio.imread(image)
    result = hands.process(img)
    hand_landmarks = result.multi_hand_landmarks
    h, w, c = img.shape
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            rect = cv2.rectangle(img, (x_min, y_min),
                (x_max, y_max), (0, 255, 0), 2)
            roi = img[y_min:y_max, x_min: x_max]
        cv2.imwrite(filename='roi.png', img=roi)

# Preprocessing all the images to extract ROI i.e. hands
def preprocess_images():
    image = 'user.png'
    #reading image
    img=imageio.imread(image)
    #Converting image to grayscale
    gray_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #Converting image to HSV format
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    #Defining boundary level for skin color in HSV
    skin_color_lower= np.array([0,40,30],np.uint8)
    skin_color_upper= np.array([43,255,255],np.uint8)

    #Producing mask
    skin_mask=cv2.inRange(hsv_img,skin_color_lower,skin_color_upper)
    #Removing Noise from mask
    skin_mask=cv2.medianBlur(skin_mask,5)
    skin_mask=cv2.addWeighted(skin_mask,0.5,skin_mask,0.5,0.0)
    
    #Applying Morphological operations
    kernel=np.ones((5,5),np.uint8)
    skin_mask=cv2.morphologyEx(skin_mask,cv2.MORPH_CLOSE,kernel)
    #Extracting hand by applying mask
    hand=cv2.bitwise_and(gray_img,gray_img,mask=skin_mask)

    #Get edges by Canny edge detection
    canny=cv2.Canny(hand,60,60)
    #save preprocessed images
    filename='processed.png'
    cv2.imwrite(filename, canny)
    print('Preprocessed image saved')
    # return canny

if __name__ == '__main__':
    roi_hand()
    preprocess_images()