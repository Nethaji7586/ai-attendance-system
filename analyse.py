import cv2
import numpy as np
import os
import pickle

# Initialize video capture
videos = cv2.VideoCapture(0)
if not videos.isOpened():
    print("Error: Could not open video device.")
    exit()

# Load Haar Cascade for face detection
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if facedetect.empty():
    print("Error: Could not load Haar cascade file.")
    exit()

face_data = []
i = 0

# Get user input for name
name = input("Enter your name: ").strip()
if not name:
    print("Error: Name cannot be empty.")
    exit()

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

print("Press 'q' to quit or wait until 100 face captures are complete.")

while True:
    ret, frame = videos.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        try:
            resized_img = cv2.resize(crop_img, (50, 50))
        except Exception as e:
            print("Error resizing face image:", e)
            continue

        if len(face_data) < 100 and i % 10 == 0:
            face_data.append(resized_img)
            cv2.putText(frame, f"Captured: {len(face_data)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Capture', frame)
    i += 1

    if len(face_data) >= 100:
        print("Capture Complete!")
        cv2.putText(frame, "Capture Complete!", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Capture', frame)
        cv2.waitKey(2000)  # Show the message for 2 seconds
        break

    k = cv2.waitKey(1)
    if k == ord('q'):
        print("Exiting capture early.")
        break

videos.release()
cv2.destroyAllWindows()

# Convert face data to numpy array
face_data = np.array(face_data)
face_data = face_data.reshape(len(face_data), -1)

# Load or initialize names list
if os.path.exists('data/names.pkl'):
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
else:
    names = []

# Append new names
names.extend([name] * len(face_data))

# Save updated names
with open('data/names.pkl', 'wb') as f:
    pickle.dump(names, f)

# Load or initialize face data
if os.path.exists('data/face_data.pkl'):
    with open('data/face_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_data, axis=0)
else:
    faces = face_data

# Save updated face data
with open('data/face_data.pkl', 'wb') as f:
    pickle.dump(faces, f)

print("Data saved successfully.")