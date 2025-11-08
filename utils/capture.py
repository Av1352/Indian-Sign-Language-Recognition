import cv2
import sys
import datetime as dt
from time import sleep
import mediapipe as mp
import numpy as np

mphands = mp.solutions.hands
hands = mphands.Hands()
handCascade = mp.solutions.drawing_utils

video_capture = cv2.VideoCapture(0)


def capture():
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

        # Draw a rectangle around the hands
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
                handCascade.draw_landmarks(
                    frame, handLMs, mphands.HAND_CONNECTIONS)


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

        # Display the resulting frames
        cv2.imshow('Video', frame)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture()
