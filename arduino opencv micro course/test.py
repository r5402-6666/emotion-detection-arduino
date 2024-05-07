import cv2
import serial
from deepface import DeepFace
import numpy as np
from time import sleep

# Initialize Arduino communication
arduino = serial.Serial('COM5', 9600)  # Change 'COM5' to your Arduino port
sleep(2)  # Wait for Arduino to initialize

# Initialize face and emotion detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = DeepFace.build_model("Emotion")

# Define emotion labels
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Function to control LEDs based on emotion
def control_leds(emotion):
    if emotion == "happy":
        arduino.write(b'H')  # Example: Send 'H' for happy
    elif emotion == "sad":
        arduino.write(b'S')  # Example: Send 'S' for sad
    # Define other emotions and corresponding LED patterns

# Function to detect emotion from face
def detect_emotion(frame):
    # Convert the frame to the correct format if necessary
    if frame.dtype != np.uint8:
        frame = cv2.convertScaleAbs(frame)

    # Ensure the frame is in the correct format (BGR)
    if frame.ndim == 2:  # If the frame is grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = face_cascade.detectMultiScale(frame_rgb, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = frame_rgb[y:y + h, x:x + w]
        
        # Resize and preprocess the face
        face = cv2.resize(face_roi, (48, 48))
        face = face.astype("float") / 255.0
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        face = cv2.convertScaleAbs(face)

        # Predict emotion
        preds = emotion_model.predict(face)
        emotion_label = EMOTIONS[np.argmax(preds)]

        # Control LEDs based on emotion
        control_leds(emotion_label)

        # Draw emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    return frame

# Main loop
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = detect_emotion(frame)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
