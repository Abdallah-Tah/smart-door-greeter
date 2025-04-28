import cv2
import os
import json
import requests
import asyncio
from aiohttp import ClientSession
from blinkpy.blinkpy import Blink
from blinkpy.auth import Auth
import numpy as np
from PIL import Image
import io
from datetime import datetime

# Config
FAMILY_JSON = 'family.json'
HA_WEBHOOK_URL = 'http://localhost:8123/api/webhook/doorbell'
CREDENTIALS_FILE = "blink_credentials.json"
DEBUG_DIR = "debug_images"

# Create debug directory if it doesn't exist
os.makedirs(DEBUG_DIR, exist_ok=True)

# Initialize face detection and recognition
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load family data
with open(FAMILY_JSON, 'r') as f:
    family_messages = json.load(f)

# Load and prepare known faces for recognition
known_faces = []
known_labels = []
label_names = {}
label_counter = 0

print("Loading known faces...")
for file in os.listdir('known_faces'):
    if file.endswith(('.jpg', '.jpeg')):
        name = os.path.splitext(file)[0].lower()
        img_path = os.path.join('known_faces', file)
        print(f"Loading face image from: {img_path}")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Failed to load image: {file}")
            continue

        # Save debug copy of loaded face
        debug_known_face = os.path.join(DEBUG_DIR, f"known_{name}.jpg")
        cv2.imwrite(debug_known_face, img)
        print(f"Saved known face debug image: {debug_known_face}")

        # Detect face in the known face image
        faces = face_cascade.detectMultiScale(
            img, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            print(f"Found face in known image {name}")
            (x, y, w, h) = faces[0]  # Use the first detected face
            face_img = img[y:y+h, x:x+w]  # Extract face region
            face_img = cv2.resize(face_img, (100, 100))  # Normalize size

            # Save debug copy of detected face
            debug_face = os.path.join(DEBUG_DIR, f"known_{name}_detected.jpg")
            cv2.imwrite(debug_face, face_img)
            print(f"Saved detected face debug image: {debug_face}")

            # Add to training data
            known_faces.append(face_img)
            known_labels.append(label_counter)
            label_names[label_counter] = name
            print(f"Loaded known face: {name} (label {label_counter})")
            label_counter += 1
        else:
            print(f"No face detected in known image: {name}")

# Train the face recognizer if we have known faces
if known_faces:
    face_recognizer.train(known_faces, np.array(known_labels))
    print("Face recognizer trained successfully!")
else:
    print("No known faces found for training!")

processed_images = set()


async def authenticate():
    """Authenticate with Blink servers"""
    try:
        blink = Blink(session=ClientSession())

        # Load saved credentials
        with open(CREDENTIALS_FILE, 'r') as f:
            auth_data = json.load(f)
            auth = Auth(auth_data)

        blink.auth = auth
        await blink.start()
        return blink
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        raise


async def process_doorbell_image(image_bytes):
    """Process an image from the doorbell camera"""
    try:
        # Save raw bytes for debugging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = os.path.join(DEBUG_DIR, f"raw_{timestamp}.jpg")
        with open(raw_path, "wb") as f:
            f.write(image_bytes)
        print(f"Saved raw doorbell image: {raw_path}")

        # Convert bytes to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_color is None:
            print("Failed to decode image bytes!")
            return None

        # Save color image for debugging
        color_path = os.path.join(DEBUG_DIR, f"color_{timestamp}.jpg")
        cv2.imwrite(color_path, img_color)
        print(f"Saved color image: {color_path}")

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # Save grayscale image for debugging
        gray_path = os.path.join(DEBUG_DIR, f"gray_{timestamp}.jpg")
        cv2.imwrite(gray_path, img_gray)
        print(f"Saved grayscale image: {gray_path}")

        # Detect faces with different parameters
        faces = face_cascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            print("No face detected in doorbell image")
            return None

        print(f"Found {len(faces)} faces in doorbell image")

        # Draw rectangles around detected faces and save debug image
        debug_img = img_color.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        faces_path = os.path.join(DEBUG_DIR, f"faces_{timestamp}.jpg")
        cv2.imwrite(faces_path, debug_img)
        print(f"Saved face detection debug image: {faces_path}")

        # Try to recognize each detected face
        visitor_name = "unknown"
        min_confidence = 100  # Lower confidence is better in LBPH

        for (x, y, w, h) in faces:
            face_img = img_gray[y:y+h, x:x+w]
            # Must match training size
            face_img = cv2.resize(face_img, (100, 100))

            # Save extracted face for debugging
            face_path = os.path.join(DEBUG_DIR, f"face_{timestamp}.jpg")
            cv2.imwrite(face_path, face_img)
            print(f"Saved extracted face: {face_path}")

            try:
                label, confidence = face_recognizer.predict(face_img)
                print(
                    f"Recognition result - Label: {label}, Confidence: {confidence}")

                # Lower confidence is better in LBPH
                if confidence < min_confidence and confidence < 80:  # Threshold for accepting a match
                    min_confidence = confidence
                    visitor_name = label_names.get(label, "unknown")
                    print(
                        f"Recognized as: {visitor_name} with confidence {confidence}")
            except Exception as e:
                print(f"Error during face recognition: {str(e)}")

        # Get and send greeting
        message = family_messages.get(visitor_name, family_messages['unknown'])
        try:
            requests.post(HA_WEBHOOK_URL, json={"message": message})
            print(f"Sending greeting for {visitor_name}: {message}")
        except:
            print(f"Would announce: {message}")

        return visitor_name

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def monitor_doorbell():
    """Monitor the Blink doorbell for new events"""
    blink = await authenticate()
    last_refresh = datetime.now()

    try:
        print("🔍 Starting doorbell monitoring...")
        await blink.refresh(force=True)  # Initial force refresh

        while True:
            try:
                now = datetime.now()
                # Force refresh every 30 seconds
                if (now - last_refresh).seconds > 30:
                    await blink.refresh(force=True)
                    last_refresh = now
                else:
                    await blink.refresh()

                doorbell = blink.cameras.get("Doorbell")
                if doorbell:
                    print("\n=== Doorbell Status ===")
                    print(f"Armed: {doorbell.armed}")
                    print(f"Motion Detected: {doorbell.motion_detected}")
                    print(f"Motion Enabled: {doorbell.motion_enabled}")

                    # Check for motion events
                    if hasattr(doorbell, 'motion_detected') and doorbell.motion_detected:
                        print("\n🚨 Motion detected!")

                        if doorbell.thumbnail and doorbell.thumbnail not in processed_images:
                            print(f"\n📸 Processing new image from motion event")

                            # Download and process image
                            async with ClientSession() as session:
                                async with session.get(doorbell.thumbnail) as response:
                                    if response.status == 200:
                                        image_bytes = await response.read()
                                        visitor = await process_doorbell_image(image_bytes)
                                        processed_images.add(
                                            doorbell.thumbnail)
                                        print(
                                            f"✅ Processed motion event, detected: {visitor}")

                # Short sleep between checks
                await asyncio.sleep(2)

            except asyncio.CancelledError:
                print("\n🛑 Stopping monitoring...")
                break
            except Exception as e:
                print(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(5)

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if blink and blink.auth.session:
            await blink.auth.session.close()

if __name__ == "__main__":
    asyncio.run(monitor_doorbell())
