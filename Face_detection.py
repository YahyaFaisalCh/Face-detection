#!/usr/bin/env python3
import cv2
import numpy as np
import os
import glob

def list_image_files(folder_path):
    # Define the file extensions you want to include
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']
    image_files = []
    # Loop through each extension and find matching files
    for i in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, i)))
    return image_files

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



folder_path = input('Folder path: ')
image_files = list_image_files(folder_path)
for file in image_files:
    filename = file.split('\\')[-1]
    print("Processing " + filename + " ...")
    outfile = filename.split('.')[0] + '_faces.' + filename.split('.')[1]
    image = cv2.imread(file)
    if image is None:
        print('image reading failed.')
        continue
    # Grayscales the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # scaleFactor=1.1: This controls the image scaling. A value of 1.1 means the image size is reduced by 10% each time it’s scaled down. Smaller values make detection slower but can improve accuracy.
    # minNeighbors=5: This determines how many "neighbor" rectangles are required around a candidate rectangle to consider it a face. Higher values reduce false positives but may miss some faces.
    # minSize=(30, 30): This sets the minimum size of detected objects. (30, 30) means faces smaller than 30x30 pixels will be ignored, helping reduce noise. 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15, minSize=(100, 100))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 10)
    print("saving " + outfile + ' ...')
    # imwrite is where the file is saving, and it is automatically saved in working directory
    outpath = folder_path+'\\output\\'
    # Create the directory
    os.makedirs(folder_path + '\\output', exist_ok=True)
    ret = cv2.imwrite(outpath+outfile, image)
    if not ret:
        print("The file cannot be saved!")
    else:
        print(outpath)
    #resized_image = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
    #cv2.imshow(file, resized_image)
    #cv2.waitKey(0)
quit()




# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height (CAP = capture)(PROP = property)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
ret = None

#ret, frame = cam.read()





# Display the captured frame
cv2.imshow('Face_detection', resized_frame)

# Press 'q' to exit the loop
cv2.waitKey(0) == ord('q')

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()
