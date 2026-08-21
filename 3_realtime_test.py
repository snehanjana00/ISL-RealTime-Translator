import cv2
import pickle
import numpy as np
import pyttsx3
import time
import os

# Exact path theke model load kora
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'model.p')

model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']

# Initialize Text-To-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Sign Labels (0: Hello, 1: Thank You, 2: Help)
labels_dict = {'0': 'Hello', '1': 'Thank You', '2': 'Help'}

cap = cv2.VideoCapture(0)

last_spoken_time = 0
last_prediction = ""

print("Starting Real-Time ISL Translator... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess current camera frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, (64, 64))
    flatten_data = img_resized.flatten()

    # Model Prediction
    prediction = model.predict([flatten_data])
    predicted_char = labels_dict.get(str(prediction[0]), "Unknown")

    # Display prediction on camera feed
    cv2.rectangle(frame, (20, 20), (450, 90), (0, 0, 0), -1)
    cv2.putText(frame, f"Sign: {predicted_char}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow('ISL Real-Time Translator', frame)

    # Speak prediction every 3 seconds if sign changes
    current_time = time.time()
    if predicted_char != last_prediction and (current_time - last_spoken_time) > 3:
        engine.say(predicted_char)
        engine.runAndWait()
        last_prediction = predicted_char
        last_spoken_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()