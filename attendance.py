import cv2
import numpy as np
import os
import pickle
import time
import csv
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

# Initialize video capture
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open video device.")
    exit()

# Load Haar Cascade for face detection
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if facedetect.empty():
    print("Error: Could not load Haar cascade file.")
    exit()

# Load names and face data from pickle files
try:
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/face_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
except FileNotFoundError:
    print("Error: Required data files not found.")
    exit()

# Train K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# CSV column names
COL_NAMES = ['Name', 'Time', 'Date']

print("Press 'q' to quit or 'o' to record attendance.")

while True:
    ret, img = video.read()
    if not ret:
        print("Error: Failed to capture video frame.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract the detected face region
        crop_img = img[y:y+h, x:x+w, :]
        try:
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        except Exception as e:
            print("Error resizing face image:", e)
            continue
        
        # Predict the name of the detected face
        output = knn.predict(resized_img)

        # Get current timestamp and date
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        timestamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')

        # Attendance record entry
        attendance = [str(output[0]), str(timestamp), str(date)]
        
        # Display rectangle and label around the detected face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), (0, 255, 0), -1)
        cv2.putText(img, str(output[0]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save attendance on pressing 'o'
        if cv2.waitKey(1) & 0xFF == ord('o'):
            time.sleep(1)  # Small delay to prevent rapid key triggers
            filename = f'Attendance/attendance_{date}.csv'
            file_exists = os.path.isfile(filename)
            
            # Write attendance record
            with open(filename, 'a' if file_exists else 'w', newline='') as csvFile:
                writer = csv.writer(csvFile)
                if not file_exists:
                    writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            print(f"Attendance recorded for {output[0]} at {timestamp} on {date}.")

    # Show the video frame
    cv2.imshow('Attendance System by Nethaji', img)

    # Quit the application on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()