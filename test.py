import cv2
from time import sleep
import mediapipe as mp

import numpy as np
import pandas as pd


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing import image

def capture():

    mphands = mp.solutions.hands
    hands = mphands.Hands()
    handCascade = mp.solutions.drawing_utils

    video_capture = cv2.VideoCapture(0)

    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

    # Capture frame-by-frame
        ret, frame = video_capture.read()
        h, w, c = frame.shape
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks

        # Draw a rectangle around the faces
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
                cv2.rectangle(frame, (x_min, y_min),
                            (x_max, y_max), (0, 255, 0), 2)
                handCascade.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
                roi_rgb_frame = framergb[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(
                    cv2.resize(roi_rgb_frame, (128, 128)), -1), 0)               
                
                
    # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            check, frame = video_capture.read()
            cv2.imshow("Capturing", frame)
            cv2.imwrite(filename='user.png', img=frame)
            video_capture.release()
            cv2.waitKey(1650)
            print("Image Saved")
            print("Program End")
            cv2.destroyAllWindows()
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            print("Turning off camera.")
            video_capture.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

        # Display the resulting frame
        cv2.imshow('Video', frame)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

def model_predict():
    
    IMG_SIZE = 128

    image = cv2.imread('user.png')
    image = cv2.resize(cv2.imread("user.png"), (IMG_SIZE,IMG_SIZE))
    image = np.array(image).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    h, w = 128,128
    model = Sequential()
    model.add(Conv2D(32, (3, 3),activation='relu', input_shape=(128, 128, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding="same",activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.40))
    model.add(Dense(96, activation='relu'))
    model.add(Dropout(0.40))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(41, activation='softmax'))
    model.load_weights('model.h5')
    print("Loaded model from disk")    
    
    show_text = [0]
    dict = {0 : '1', 
            1: '2', 
            2 : '3', 
            3 : '4', 
            4 : '5', 
            5: '6', 
            6 : '7', 
            7 : '8', 
            8 : '9',
            9: 'A', 
            10 : 'B', 
            11 : 'C', 
            12 : 'D', 
            13 : 'E', 
            14 : 'F', 
            15 : 'G', 
            16 : 'H',
            17 : 'HELLO', 
            18 : 'I', 
            19 : 'J', 
            20 : 'K', 
            21 : 'L', 
            22 : 'M', 
            23 : 'N', 
            24 : 'NO',
            25 : 'O', 
            26 : 'P', 
            27 : 'PLEASE', 
            28 : 'Q', 
            29 : 'R', 
            30 : 'S', 
            31 : 'SORRY',  
            32 : 'T', 
            33 : 'THANKS', 
            34 : 'U', 
            35 : 'V', 
            36 : 'W', 
            37 : 'X', 
            38 : 'Y', 
            39 : 'YES',
            40 : 'Z'}

    
    prediction = model.predict(image)
    global maxindex
    maxindex = int(np.argmax(prediction))
    show_text[0] = maxindex

    print("dict: ",dict[maxindex])
    global hand
    hand = dict[maxindex]
    print("Hand", hand)
    

def main():
    cropped_img = capture()
    model_predict()


if __name__ == '__main__':
    main()
    