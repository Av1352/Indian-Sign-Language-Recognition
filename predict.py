import numpy as np
import cv2

from keras.models import Sequential, model_from_json
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
dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 
        9: '9', 10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 
        17: 'h', 18: 'i', 19: 'j', 20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o', 
        25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't', 30: 'u', 31: 'v', 32: 'w', 
        33: 'x', 34: 'y', 35: 'z'}

IMG_SIZE = 100

image = cv2.imread('processed.png')
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
image = np.array(image).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# h, w, c = image.shape

def predict():
    input_shape = (100, 100, 1)
    n_classes = 36
    model = Sequential()
    # The first two layers with 32 filters of window size 3x3
    model.add(Conv2D(32, (3, 3), padding='same',
                    activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    model.load_weights('Models/model.h5')
    print("Loaded model from disk")

    # model.compile(optimizer='adam',
    #           loss='categorical_crossentropy',
    #           metrics=['accuracy'])


    cv2.ocl.setUseOpenCL(False)
    prediction = model.predict(image)
    # print(prediction)

    global maxindex
    maxindex = int(np.argmax(prediction))
    show_text[0] = maxindex

    print("dict: ", dict[maxindex])
    global hand
    hand = dict[maxindex]
    print("Hand", hand)

    return hand


# def music_rec():
# 	# print('---------------- Value ------------', music_dist[show_text[0]])

# 	df = pd.read_csv(music_dist[show_text[0]])
# 	df = df[['', 'Album', 'Artist']]
# 	df = df.head(15)
# 	return df


if __name__ == '__main__':
    predict()
