import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence=0.3)

directory = './chordshapes'
data = []
labels = []

for dir in os.listdir(directory):
    for path in os.listdir(os.path.join(directory,dir)):
        data_aux = []
        x_ = []
        y_ = []
        img = cv2.imread(os.path.join(directory, dir, path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for landmark in results.multi_hand_landmarks:
                for i in range(len(landmark.landmark)):
                    x = landmark.landmark[i].x
                    y = landmark.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                for i in range(len(landmark.landmark)):
                    x = landmark.landmark[i].x
                    y = landmark.landmark[i].y
                    data_aux.append(x-min(x_))
                    data_aux.append(y-min(y_))
            data.append(data_aux)
            labels.append(dir)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()