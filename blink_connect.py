import cv2
import os
import json
import asyncio
from aiohttp import ClientSession
from blinkpy.blinkpy import Blink
from blinkpy.auth import Auth
import numpy as np
from datetime import datetime, timedelta

# Config
CREDENTIALS_FILE = "blink_credentials.json"
FAMILY_JSON = "family.json"
DEBUG_DIR = "debug_images"
LOG_FILE = "processed_media.log"
os.makedirs(DEBUG_DIR, exist_ok=True)


async def log_processed_media(media_type, file_path, visitor_name, confidence=None):
    """Log processed media information to file"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} | {media_type} | {file_path} | Visitor: {visitor_name}"
        if confidence is not None:
            log_entry += f" | Confidence: {confidence:.2f}"
        log_entry += "\n"
        
        print(f"Writing to log: {log_entry}")  # Debug print
        
        with open(LOG_FILE, "a") as f:
            f.write(log_entry)
        print(f"📝 Successfully logged {media_type} processing")
    except Exception as e:
        print(f"❌ Error writing to log file: {str(e)}")
        import traceback
        traceback.print_exc()

# Initialize face detection and recognition
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load family messages
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


async def authenticate():
    """Authenticate with Blink"""
    try:
        blink = Blink(session=ClientSession())
        with open(CREDENTIALS_FILE, 'r') as f:
            auth_data = json.load(f)
            auth = Auth(auth_data)

        blink.auth = auth
        await blink.start()
        await blink.refresh(force=True)

        return blink
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


async def download_media(session, url, save_path):
    """Download media from Blink cloud"""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                with open(save_path, 'wb') as f:
                    f.write(content)
                return True
    except Exception as e:
        print(f"Error downloading media: {str(e)}")
    return False


async def process_motion_events(blink, doorbell):
    """Process motion events from Blink cloud storage"""
    try:
        print("\n=== Checking for Motion Events ===")
        print(f"Camera Status: {doorbell.name}")
        print(
            f"Motion Detection: {'Enabled' if doorbell.motion_enabled else 'Disabled'}")

        # Check for motion events
        if hasattr(doorbell, 'motion_detected'):
            print(
                f"Motion State: {'DETECTED' if doorbell.motion_detected else 'None'}")

        if hasattr(doorbell, 'last_motion') and doorbell.last_motion:
            print(f"Last Motion: {doorbell.last_motion}")

        # Print all available attributes for debugging
        print("\nCamera Properties:")
        for attr in dir(doorbell):
            if not attr.startswith('_'):  # Skip private attributes
                try:
                    value = getattr(doorbell, attr)
                    if not callable(value):  # Skip methods
                        print(f"{attr}: {value}")
                except Exception:
                    pass

        # Check for motion and media
        if doorbell.motion_detected:
            print("\n🚨 Motion Event Detected!")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Try to get video
            if hasattr(doorbell, 'video') and doorbell.video:
                video_path = os.path.join(DEBUG_DIR, f"motion_{timestamp}.mp4")
                print(f"\n📹 Downloading motion video...")

                async with ClientSession() as session:
                    if await download_media(session, doorbell.video, video_path):
                        print("✅ Video downloaded successfully")
                        print("Processing video for face recognition...")
                        visitor = await process_video_for_recognition(video_path, timestamp)
                        if visitor != "unknown":
                            message = family_messages.get(
                                visitor, family_messages['unknown'])
                            print(f"👋 {message}")
                        await log_processed_media("video", video_path, visitor, None)
                    else:
                        print("❌ Failed to download video")

            # Try to get thumbnail
            if hasattr(doorbell, 'thumbnail') and doorbell.thumbnail:
                print(f"\n📸 Processing motion thumbnail...")
                async with ClientSession() as session:
                    async with session.get(doorbell.thumbnail) as response:
                        if response.status == 200:
                            image_bytes = await response.read()
                            image_path = os.path.join(
                                DEBUG_DIR, f"thumbnail_{timestamp}.jpg")

                            with open(image_path, "wb") as f:
                                f.write(image_bytes)
                            print("✅ Thumbnail saved successfully")

                            visitor = await process_image(image_bytes, timestamp)
                            if visitor != "unknown":
                                print(f"✨ Recognized {visitor} in thumbnail!")
                            await log_processed_media("image", image_path, visitor, None)
                        else:
                            print(
                                f"❌ Failed to download thumbnail: {response.status}")

        print("\n=== Motion Check Complete ===")

    except Exception as e:
        print(f"Error processing motion events: {str(e)}")
        import traceback
        traceback.print_exc()


async def process_video_for_recognition(video_path, timestamp):
    """Process video frames for face recognition"""
    frames = []
    best_match = ("unknown", 100)  # (name, confidence)

    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 5th frame to improve performance
            if frame_count % 5 == 0:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)

                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(20, 20)
                )

                if len(faces) > 0:
                    # Save frame with detected faces
                    frame_path = os.path.join(
                        DEBUG_DIR, f"frame_{timestamp}_{frame_count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    print(f"Saved frame with faces: {frame_path}")

                    for (x, y, w, h) in faces:
                        face = gray[y:y+h, x:x+w]
                        face = cv2.resize(face, (100, 100))

                        # Save extracted face
                        face_path = os.path.join(
                            DEBUG_DIR, f"face_{timestamp}_{frame_count}.jpg")
                        cv2.imwrite(face_path, face)

                        try:
                            label, confidence = face_recognizer.predict(face)
                            print(
                                f"Frame {frame_count} recognition - Label: {label}, Confidence: {confidence}")

                            # Lower threshold for better recognition
                            if confidence < best_match[1] and confidence < 65:
                                best_match = (label_names.get(
                                    label, "unknown"), confidence)
                                print(
                                    f"New best match: {best_match[0]} with confidence {best_match[1]}")

                                # Save annotated frame
                                debug_frame = frame.copy()
                                cv2.rectangle(debug_frame, (x, y),
                                              (x+w, y+h), (255, 0, 0), 2)
                                cv2.putText(debug_frame, f"{best_match[0]} ({confidence:.1f})",
                                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.imwrite(os.path.join(
                                    DEBUG_DIR, f"recognized_{timestamp}_{frame_count}.jpg"), debug_frame)
                        except Exception as e:
                            print(f"Error during face recognition: {str(e)}")

            frame_count += 1

        cap.release()

        # Log the processed video
        await log_processed_media("video", video_path, best_match[0], best_match[1])
        return best_match[0]

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        return "unknown"


async def process_image(image_bytes, timestamp):
    """Process an image and detect/recognize faces"""
    try:
        # Save raw image
        raw_path = os.path.join(DEBUG_DIR, f"raw_{timestamp}.jpg")
        with open(raw_path, "wb") as f:
            f.write(image_bytes)
        print(f"Saved raw image to {raw_path}")

        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to decode image!")
            return "unknown"

        # Convert to grayscale and enhance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Save preprocessed image
        cv2.imwrite(os.path.join(DEBUG_DIR, f"gray_{timestamp}.jpg"), gray)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        )

        if len(faces) == 0:
            print("No faces detected in image")
            await log_processed_media("image", raw_path, "unknown")
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
            face_path = os.path.join(DEBUG_DIR, f"face_{timestamp}.jpg")
            cv2.imwrite(face_path, face)

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

        # Save annotated image
        cv2.imwrite(os.path.join(
            DEBUG_DIR, f"detected_{timestamp}.jpg"), debug_img)

        # Log the processed image
        await log_processed_media("image", raw_path, visitor_name, min_confidence if visitor_name != "unknown" else None)
        return visitor_name

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return "unknown"


async def main():
    processed_events = set()
    blink = None
    last_refresh = datetime.now()

    try:
        print("🔄 Connecting to Blink...")
        blink = await authenticate()
        print("✅ Connected to Blink successfully")

        # Initial setup
        doorbell = blink.cameras.get("Doorbell")
        if doorbell:
            print(f"Found doorbell camera: {doorbell.name}")
            print(f"Initial state - Motion Enabled: {doorbell.motion_enabled}")
        else:
            print("❌ No doorbell camera found!")
            return

        print("\n🔍 Starting cloud monitoring...")

        while True:
            try:
                now = datetime.now()
                # Force refresh every 30 seconds
                if (now - last_refresh).seconds > 30:
                    await blink.refresh(force=True)
                    last_refresh = now

                    if doorbell:
                        # Process any new motion events
                        await process_motion_events(blink, doorbell)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                print("\n🛑 Stopping monitoring...")
                break
            except Exception as e:
                print(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(5)

    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if blink and blink.auth.session:
            try:
                await blink.auth.session.close()
                print("✅ Cleaned up and closed connection")
            except:
                pass

if __name__ == "__main__":
    asyncio.run(main())
