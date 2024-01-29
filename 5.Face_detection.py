import cv2
import mediapipe as mp

# Function to detect facial landmarks and analyze facial expressions
def detect_facial_landmarks():
    # Initialize Mediapipe Face Detection and Face Mesh modules
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.3)

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Read a frame from the webcam
        success, frame = cap.read()
        if not success:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face detection
        result_detection = face_detection.process(rgb_frame)

        # If faces are detected, process facial landmarks
        if result_detection.detections:
            for detection in result_detection.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Convert the BGR image to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform face mesh detection
                result_mesh = face_mesh.process(rgb_frame)

                # If facial landmarks are detected, analyze facial expressions
                if result_mesh.multi_face_landmarks:
                    for face_landmarks in result_mesh.multi_face_landmarks:
                        # Extract facial landmarks
                        for landmark in face_landmarks.landmark:
                            x, y = int(landmark.x * iw), int(landmark.y * ih)
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Display the frame with facial landmarks
        cv2.imshow('Facial Landmark Detection', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function for facial landmark detection and expression analysis
detect_facial_landmarks()
