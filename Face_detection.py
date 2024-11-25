import cv2
import numpy as np
import os
import glob 

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

option = input('face detection from camera[1] or photos[anything else]: ')
if option == "1":
    # Start video capture from the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    quit()

def list_image_files(folder_path):
    # Define the file extensions you want to include
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']
    image_files = []
    # Loop through each extension and find matching files
    for i in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, i)))
    # Returning a list full of images with their complete folder path
    return image_files

folder_path = input('Path of your folder: ')
image_files = list_image_files(folder_path)

for file in image_files:
    # Getting the file name from a complete folder path
    filename = file.split('\\')[-1]
    print("Processing " + filename + " ...")
    
    # Adding "faces" to the file name
    outfile = filename.split('.')[0] + '_faces.' + filename.split('.')[1]
    image = cv2.imread(file)
    if image is None:
        print('image reading failed.')
        continue

    # Grayscales the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    facenumber = 1
    
    # scaleFactor=1.1: This controls the image scaling. A value of 1.1 means the image size is reduced by 10% each time it’s scaled down. Smaller values make detection slower but can improve accuracy.
    # minNeighbors=5: This determines how many "neighbor" rectangles are required around a candidate rectangle to consider it a face. Higher values reduce false positives but may miss some faces.
    # minSize=(30, 30): This sets the minimum size of detected objects. (30, 30) means faces smaller than 30x30 pixels will be ignored, helping reduce noise. 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15, minSize=(100, 100))
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 10)
        
        # Cropping the image to only get faces
        cropped_image = image[y:y+h, x:x+w]
        
        # Adding number to the file name
        facefilename = outfile.split('.')[0] + str(facenumber) +'.' + outfile.split('.')[1]
        
        # Creating a new directory and then saving with imwrite
        os.makedirs(folder_path + '\\cropped_images', exist_ok=True)
        cv2.imwrite(folder_path+'\\cropped_images'+'\\'+facefilename, cropped_image)
        
        facenumber += 1

    # imwrite is where the file is saving, and it is automatically saved in working directory
    print("saving " + outfile + ' ...')  
    outpath = folder_path+'\\output\\'
    
    # Creating a new directory and then saving with imwrite
    os.makedirs(folder_path + '\\output', exist_ok=True)
    ret = cv2.imwrite(outpath+outfile, image)
    if not ret:
        print("The file cannot be saved!")
    else:
        print(outpath)

    #resized_image = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
    #cv2.imshow(file, resized_image)
    #cv2.waitKey(0)