import cv2
import mediapipe as mp

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Define landmark indices for finger tips and joints
tip_indices = [4, 8, 12, 16, 20]
joint_indices = [2, 6, 10, 14, 18]

def count_fingers(hand_landmarks):
    # Initialize finger count
    finger_count = 0

    # Check each finger
    for tip, joint in zip(tip_indices, joint_indices):
        # Get y-coordinates of tip and joint
        tip_y = hand_landmarks[tip].y
        joint_y = hand_landmarks[joint].y

        # Check if finger is extended (open)
        if tip_y < joint_y:
            finger_count += 1

    return finger_count

def draw_landmarks(image, landmarks):
    # Create a mediapipe hands drawing utility
    mp_drawing = mp.solutions.drawing_utils

    # Initialize the drawing module
    drawing_module = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Draw landmarks on the image
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        mp_hands.HAND_CONNECTIONS,
        landmark_drawing_spec=drawing_module,
        connection_drawing_spec=drawing_module,
    )

def draw_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1,
              color=(0, 0, 255), thickness=2):
    # Draw text on the image
    cv2.putText(image, text, position, font, font_scale, color, thickness)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Get finger count from landmarks
            finger_count = count_fingers(landmarks.landmark)

            # Draw landmarks on the frame
            draw_landmarks(frame, landmarks)

            # Draw finger count on the frame
            draw_text(frame, f"Fingers: {finger_count}", (10, 30))

    # Display the resized frame
    resized_frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Finger Counting", resized_frame)

    # Check for exit key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

























