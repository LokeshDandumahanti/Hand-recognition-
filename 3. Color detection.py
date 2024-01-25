'''
1. import all libraries
2. convert from bgr to hsv
3. define upper and lower boundries of color to be detected
4. create a mask for the specified color range
5. use bitwise and mask and original image to show only the detected color
6. Load image
7. Perform color detection on the image
8. Display the original and processed images
'''
import cv2
import numpy as np

def color_detection(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40,40,40])
    upper_green = np.array([80,255,255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    color_detected = cv2.bitwise_and(image, image, mask=mask)

    return color_detected

image_path = "C:/Users/vijay/OneDrive/Desktop/4th SSH/Projects/OpenCV Projects/image1.jpg"
image = cv2.imread(image_path)

detected_image = color_detection(image)

cv2.imshow('Original Image', image)
cv2.imshow('Color Detected', detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
