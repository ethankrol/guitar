import os
import cv2

directory = "./chordshapes"
if not os.path.exists(directory):
    os.makedirs(directory)

chords = ["D", "A", "E", "Am", "Dm", "Em", "G", "C"]
dataset_size = 100

cap = cv2.VideoCapture(0)

for count, c in enumerate(chords):
    if not os.path.exists(os.path.join(directory,str(c))):
        os.makedirs(os.path.join(directory,str(c)))
    print(f'Collecting data for chord {c}')
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press q to capture', (100, 50), cv2.QT_FONT_BLACK, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(5) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(directory, str(c), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
