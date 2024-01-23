import cv2
import mediapipe as mp
import numpy as np
import ctypes

# Function to change system volume
def change_volume(level):
    # Set volume using ctypes library
    # Define minimum and maximum volume levels as per your system requirements
    min_vol = 0  # Minimum volume
    max_vol = 100  # Maximum volume
    volume = np.interp(level, [0, 200], [min_vol, max_vol])
    
    # Set the system volume using ctypes library (Windows platform)
    try:
        ctypes.windll.user32.SendMessageW(-1, 0x319, 0, volume * 65536)  # Change system volume
    except Exception as e:
        print("Volume adjustment failed:", e)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    # Convert the image to RGB and process it with Mediapipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the landmark coordinates
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Get the coordinates of index finger and thumb in the frame
            height, width, _ = frame.shape
            thumb_x, thumb_y = int(thumb.x * width), int(thumb.y * height)
            index_finger_x, index_finger_y = int(index_finger.x * width), int(index_finger.y * height)

            # Calculate the distance between thumb and index finger
            distance = np.sqrt((thumb_x - index_finger_x)**2 + (thumb_y - index_finger_y)**2)

            # Adjust volume based on distance between thumb and index finger
            if distance < 200:  # Define a suitable threshold for gesture recognition
                change_volume(distance)

            # Draw points on thumb and index finger
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 0), -1)
            cv2.circle(frame, (index_finger_x, index_finger_y), 10, (0, 255, 0), -1)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
capture.release()
cv2.destroyAllWindows()
