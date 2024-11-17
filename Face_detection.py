#!/usr/bin/env python3
import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

start_coord = (0, 0)
end_coord = (100, 200)
color = (0, 0, 0)
thickness = 5

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height (CAP = capture)(PROP = property)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cam.read()
    if ret == False:
        print("Camera reading failed, know why?")
        quit()
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # scaleFactor=1.1: This controls the image scaling. A value of 1.1 means the image size is reduced by 10% each time it’s scaled down. Smaller values make detection slower but can improve accuracy.
    # minNeighbors=5: This determines how many "neighbor" rectangles are required around a candidate rectangle to consider it a face. Higher values reduce false positives but may miss some faces.
    # minSize=(30, 30): This sets the minimum size of detected objects. (30, 30) means faces smaller than 30x30 pixels will be ignored, helping reduce noise. 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 10)

    #cv2.rectangle(frame, start_coord, end_coord, color, thickness)

    # Display the captured frame
    cv2.imshow('Face_detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()
