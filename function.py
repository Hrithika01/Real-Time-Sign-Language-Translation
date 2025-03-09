import cv2
import numpy as np
import os
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR to RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB to BGR
    return image, results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3)
            return np.concatenate([rh])
    else:
        return np.zeros(21*3)  # Ensure a consistent output shape when no hand is detected

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Adjusted actions to match your defined gestures
actions = np.array(['Hello', 'Yes', 'No', 'Thank you', 'Sorry', 'Eat', 'Like', 'Want', 'I Love You', 'Which', 'Peace', 'Bathroom', 'Good', 'Bad', 'Stop', 'Money', 'Happy', 'Phone', 'Hear', 'See', 'Drink', 'Walk'])

no_sequences = 30
sequence_length = 30
