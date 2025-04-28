import cv2
import os

print("Testing face detection...")

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Test with known face
known_face_path = 'known_faces/abdallah.jpeg'
if os.path.exists(known_face_path):
    print(f"\nTesting with known face: {known_face_path}")
    img = cv2.imread(known_face_path)
    if img is None:
        print("❌ Failed to load image")
    else:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)

        # Try different detection parameters
        detection_params = [
            {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (30, 30)},
            {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (20, 20)},
            {"scaleFactor": 1.2, "minNeighbors": 6, "minSize": (40, 40)}
        ]

        for params in detection_params:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=params["scaleFactor"],
                minNeighbors=params["minNeighbors"],
                minSize=params["minSize"]
            )

            if len(faces) > 0:
                print(f"✅ Face detected with params: {params}")
                # Draw rectangles and save debug image
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imwrite('debug_images/test_detection_result.jpg', img)
                print(f"Debug image saved to debug_images/test_detection_result.jpg")
                break
            else:
                print(f"❌ No face detected with params: {params}")
else:
    print(f"❌ Known face image not found at {known_face_path}")
