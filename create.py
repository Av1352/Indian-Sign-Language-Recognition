import cv2
import uuid
import os
import time

IMAGES_PATH = os.path.join('data\images')

actions_numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
actions_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
actions_words = ['hello', 'bye', 'yes', 'no', 'good morning', 'please', 'help', 'sorry',
                'thank you', 'okay']

no_imgs = 10

def main():
    numbers()
    letters()
    words()
    
def numbers():
    for action in actions_numbers:
        os.mkdir('data\images\\'+ action)
        cap = cv2.VideoCapture(0)
        print('Collecting imnages for {}'. format(action))
        time.sleep(5)
        for imgnum in range(no_sequences):
            ret, frame = cap.read()
            image_name = ps.path.join(
                IMAGES_PATH, action, action+'.'+'{}.jpg'.format(str(uuid.uuid1())))
            cv2.imwrite(image_name, frame)
            time.sleep(2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

def letters():
    for action in actions_letters:
        os.mkdir('data\images\\' + action)
        cap = cv2.VideoCapture(0)
        print('Collecting imnages for {}'. format(action))
        time.sleep(5)
        for imgnum in range(no_sequences):
            ret, frame = cap.read()
            image_name = ps.path.join(
                IMAGES_PATH, action, action+'.'+'{}.jpg'.format(str(uuid.uuid1())))
            cv2.imwrite(image_name, frame)
            time.sleep(2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

def words():
    for action in actions_words:
        os.mkdir('data\images\\' + action)
        cap = cv2.VideoCapture(0)
        print('Collecting imnages for {}'. format(action))
        time.sleep(5)
        for imgnum in range(no_sequences):
            ret, frame = cap.read()
            image_name = ps.path.join(
                IMAGES_PATH, action, action+'.'+'{}.jpg'.format(str(uuid.uuid1())))
            cv2.imwrite(image_name, frame)
            time.sleep(2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

if __name__=='__main__':
    main()