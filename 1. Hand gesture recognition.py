'''
Logic of the code

importing all the libraries
Hand recognising library is initiatiated
a logic should be written in class for the detection of hand. if yes, words on screen.
open webcam
as long as webcam is open, use the detection class logic
get it from bgr to rgb
flip the image
if results match with the hand landmarks,
for every handmark present in results
say that you recognised the gesture and print on it.
add skeletons on the hands
display the frame
if q is pressed, exit it.
'''
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def recognize_gesture(hand_landmarks):
    if all(lm.y > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y for lm in
           hand_landmarks.landmark[1:]):
        return "First"
    else:
        return "Open Hand"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            gesture = recognize_gesture(hand_landmarks)
            cv2.putText(frame, f"Gesture: {gesture}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks,
                                                      mp_hands.HAND_CONNECTIONS)
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
