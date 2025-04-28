import cv2
import os

# Load OpenCV's pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load known faces
known_faces = {}

for file in os.listdir('known_faces'):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(f'known_faces/{file}', 0)
        faces = face_cascade.detectMultiScale(img, 1.1, 4)
        if len(faces) > 0:
            known_faces[os.path.splitext(file)[0]] = img
            print(f"Loaded known face: {file}")

# Load visitor image /home/abdallah-mohamed/Documents/smart-door-greeter/visitor1.jpeg
visitor_img = cv2.imread('visitor1.jpeg', 0)
if visitor_img is None:
    print("❌ Visitor image not found.")
else:
    print("✅ Visitor image loaded successfully!")

visitor_faces = face_cascade.detectMultiScale(visitor_img, 1.1, 4)

if len(visitor_faces) == 0:
    print("❌ No face found in visitor image.")
else:
    print("✅ Face detected in visitor image!")
