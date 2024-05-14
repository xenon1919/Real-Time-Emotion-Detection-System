from tkinter import Tk, Button
from tkinter import ttk
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
cap = None

def open_camera():
    global cap
    cap = cv2.VideoCapture(0)
    detect_emotion()

def close_camera():
    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()

def close_program():
    close_camera()
    root.destroy()

def detect_emotion():
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Find the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        close_program()
    else:
        root.after(10, detect_emotion)  # Call detect_emotion again after 10 milliseconds

root = Tk()
root.title("Emotion Detection Program")
root.geometry("300x150")

style = ttk.Style()
style.configure('TButton', font=('calibri', 10), padding=5)

open_button = ttk.Button(root, text="Open Camera", command=open_camera)
open_button.pack(pady=5)

close_button = ttk.Button(root, text="Close Camera", command=close_camera)
close_button.pack(pady=5)

close_program_button = ttk.Button(root, text="Close Program", command=close_program)
close_program_button.pack(pady=5)

root.mainloop()
