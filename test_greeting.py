import cv2
import os
import json
import numpy as np
from datetime import datetime
import sys
import traceback

# Print Python version for debugging
print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")

# Config
FAMILY_JSON = 'family.json'
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Initialize face detection and recognition
try:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("Successfully initialized OpenCV face detection and recognition")
except Exception as e:
    print(f"Error initializing OpenCV: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# Load family messages
try:
    with open(FAMILY_JSON, 'r') as f:
        family_messages = json.load(f)
    print(f"Loaded family messages: {family_messages}")
except Exception as e:
    print(f"Error loading family.json: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# Load and prepare known faces for recognition
known_faces = []
known_labels = []
label_names = {}
label_counter = 0

print("Loading known faces...")
try:
    for file in os.listdir('known_faces'):
        if file.endswith(('.jpg', '.jpeg')):
            name = os.path.splitext(file)[0].lower()
            img_path = os.path.join('known_faces', file)
            print(f"Loading face image from: {img_path}")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Failed to load image: {file}")
                continue

            # Detect face in the known face image
            faces = face_cascade.detectMultiScale(
                img, scaleFactor=1.1, minNeighbors=5)

            if len(faces) > 0:
                print(f"Found face in known image {name}")
                (x, y, w, h) = faces[0]  # Use the first detected face
                face_img = img[y:y+h, x:x+w]  # Extract face region
                face_img = cv2.resize(face_img, (100, 100))  # Normalize size

                # Add to training data
                known_faces.append(face_img)
                known_labels.append(label_counter)
                label_names[label_counter] = name
                print(f"Loaded known face: {name} (label {label_counter})")
                label_counter += 1
            else:
                print(f"No face detected in known image: {name}")
except Exception as e:
    print(f"Error loading known faces: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# Train the face recognizer if we have known faces
if known_faces:
    try:
        face_recognizer.train(known_faces, np.array(known_labels))
        print("Face recognizer trained successfully!")
    except Exception as e:
        print(f"Error training face recognizer: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
else:
    print("No known faces found for training!")
    sys.exit(1)


def process_image(image_path):
    """Process an image and detect/recognize faces"""
    print(f"\nProcessing image: {image_path}")

    try:
        # Verify file exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return "unknown"

        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return "unknown"

        print(f"Loaded image with shape: {img.shape}")

        # Convert to grayscale and enhance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Save preprocessed image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gray_path = os.path.join(DEBUG_DIR, f"gray_{timestamp}.jpg")
        cv2.imwrite(gray_path, gray)
        print(f"Saved grayscale image to: {gray_path}")

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        )

        if len(faces) == 0:
            print("No faces detected in image")
            return "unknown"

        print(f"Found {len(faces)} faces in image")
        visitor_name = "unknown"
        min_confidence = 100  # Lower confidence is better in LBPH

        # Process each detected face
        debug_img = img.copy()
        for (x, y, w, h) in faces:
            # Extract and normalize face
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))

            # Save face for debugging
            face_path = os.path.join(
                DEBUG_DIR, f"detected_face_{timestamp}.jpg")
            cv2.imwrite(face_path, face)
            print(f"Saved detected face to: {face_path}")

            try:
                # Attempt recognition
                label, confidence = face_recognizer.predict(face)
                print(
                    f"Recognition result - Label: {label}, Confidence: {confidence}")

                if confidence < min_confidence and confidence < 65:  # Lower threshold for better recognition
                    min_confidence = confidence
                    visitor_name = label_names.get(label, "unknown")
                    print(
                        f"Recognized as: {visitor_name} with confidence {confidence}")

                    # Draw on debug image
                    cv2.rectangle(debug_img, (x, y),
                                  (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(debug_img, f"{visitor_name} ({confidence:.1f})",
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error during face recognition: {str(e)}")
                traceback.print_exc()

        # Save annotated image
        annotated_path = os.path.join(DEBUG_DIR, f"annotated_{timestamp}.jpg")
        cv2.imwrite(annotated_path, debug_img)
        print(f"Saved annotated image to: {annotated_path}")

        # Get appropriate greeting
        message = family_messages.get(visitor_name, family_messages['unknown'])
        print(f"\n👋 GREETING: {message}")

        # Log the processed image
        with open("processed_media.log", "a") as f:
            log_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | image | {image_path} | Visitor: {visitor_name}"
            if visitor_name != "unknown":
                log_entry += f" | Confidence: {min_confidence:.2f}"
            log_entry += "\n"
            f.write(log_entry)
        print(f"📝 Logged to processed_media.log")

        return visitor_name

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        traceback.print_exc()
        return "unknown"


print("\n=== STARTING TESTS ===")

# Test with your own image
known_path = "known_faces/abdallah.jpeg"
print("\n=== Testing with known face (you) ===")
process_image(known_path)

# Test with visitor image
visitor_path = "visitor1.jpeg"
print(f"\n=== Testing with visitor image: {visitor_path} ===")
if os.path.exists(visitor_path):
    process_image(visitor_path)
else:
    print(f"Visitor image not found at: {visitor_path}")
    print("Available files in current directory:")
    for file in os.listdir('.'):
        if file.endswith(('.jpg', '.jpeg')):
            print(f" - {file}")

print("\nTest complete! Check the processed_media.log file for logs.")
