import numpy as np
import cv2
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import time
import pandas as pd
import mediapipe as mp

show_text = [0]


def camera():
    model = Sequential()
    model.add(Conv2D(32, (3, 3),activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding="same",activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.40))
    model.add(Dense(96, activation='relu'))
    model.add(Dropout(0.40))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(41, activation='softmax'))


    model.load_weights('model.h5')
    cv2.ocl.setUseOpenCL(False)
	
    dict = {1 : '1', 2: '2', 3 : '3', 4 : '4', 5 : '5', 6: '6', 7 : '7', 8 : '8', 9 : '9',
              10: 'A', 11 : 'B', 12 : 'C', 13 : 'D', 14 : 'E', 15 : 'F', 16 : 'G', 17 : 'H',
              18 : 'HELLO', 19 : 'I', 20 : 'J', 21 : 'K', 22 : 'L', 23 : 'M', 24 : 'N', 25 : 'NO',
              26 : 'O', 27 : 'P', 28 : 'PLEASE', 29 : 'Q', 30 : 'R', 31 : 'S', 32 : 'SORRY',  
              33 : 'T', 34 : 'THANKS', 35 : 'U', 36 : 'V', 37 : 'W', 38 : 'X', 39 : 'Y', 
              40 : 'YES' , 41 : 'Z'}
    
    mphands = mp.solutions.hands
    hands = mphands.Hands()
    handCascade = mp.solutions.drawing_utils
    image = cv2.imread("user.jpg")
    h, w, c = image.shape
    framergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    

    hand_landmarks = result.multi_hand_landmarks

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
        
        

        roi_rgb_frame = framergb[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_rgb_frame, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        global maxindex
        maxindex = int(np.argmax(prediction))
        show_text[0] = maxindex
        print(show_text)
        print(dict[maxindex])
        global hand
        hand = dict[maxindex]
        print(hand)
        return hand

# def music_rec():
# 	# print('---------------- Value ------------', music_dist[show_text[0]])

# 	df = pd.read_csv(music_dist[show_text[0]])
# 	df = df[['', 'Album', 'Artist']]
# 	df = df.head(15)
# 	return df


if __name__ == '__main__':
    camera()
