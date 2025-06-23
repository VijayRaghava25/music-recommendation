import cv2
import numpy as np
from keras.models import load_model
from pygame import mixer
import os
import random

# Load your trained CNN model
model = load_model('emotion_model/FER_Best_Model.h5')

# Emotion labels (in the same order as used in training)
emotions = ['Sad', 'Neutral', 'Happy']

# Emotion folder mapping
emotion_folders = {
    'Happy': 'songs/happy',
    'Neutral': 'songs/neutral',
    'Sad': 'songs/sad'
}

# Initialize pygame mixer for playing music
mixer.init()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam and capture one frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to capture image from webcam.")
    exit()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    print("No face detected.")
    exit()

# Assume first detected face
(x, y, w, h) = faces[0]
face = gray[y:y+h, x:x+w]
face = cv2.resize(face, (48, 48))
face = face.astype('float32') / 255.0
face = np.reshape(face, (1, 48, 48, 1))

# Predict emotion
prediction = model.predict(face)
emotion_index = np.argmax(prediction)
predicted_emotion = emotions[emotion_index]

print(f"Predicted Emotion: {predicted_emotion}")

# Pick a random song from the emotion folder
folder_path = emotion_folders[predicted_emotion]
songs = os.listdir(folder_path)

if not songs:
    print(f"No songs found in {folder_path}")
    exit()

song_to_play = os.path.join(folder_path, random.choice(songs))
print(f"Playing: {song_to_play}")

# Play the song
mixer.music.load(song_to_play)
mixer.music.play()

# Optional: Display image with prediction
cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow("Detected Emotion", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
