import cv2
import pickle
import mediapipe as mp
import numpy as np

chords = ["D", "A", "E", "Am", "Dm", "Em", "G", "C"]

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()
while True:
    data_aux = []
    x_ = []
    y_ = []
    success, frame = cap.read()
    if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks,mp_hands.HAND_CONNECTIONS)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
            max_length = model.n_features_in_
            if len(data_aux) < max_length:
                data_aux.extend([0] * (max_length - len(data_aux)))

            prediction = model.predict([np.asarray(data_aux)])

            predicted_chord = prediction[0]

            cv2.putText(frame, "Current chord is {}".format(predicted_chord), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 0, 0), 4)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()