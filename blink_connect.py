import cv2
import os
import json
import asyncio
from aiohttp import ClientSession
from blinkpy.blinkpy import Blink
from blinkpy.auth import Auth
import numpy as np
from datetime import datetime, timedelta
import uuid

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

        # Check if we need to authenticate with username/password
        if auth_data.get('token') is None or auth_data.get('account_id') is None:
            print("Token or account ID missing, performing full authentication...")
            # We need to authenticate with username/password
            username = auth_data.get('username')
            password = auth_data.get('password')

            if not username or not password:
                raise ValueError(
                    "Username and password required in credentials file")

            # Create a new Auth instance with just the username/password
            blink.auth = Auth({'username': username, 'password': password})
            await blink.start()

            # Save the updated credentials with token
            updated_creds = blink.auth.login_attributes
            updated_creds.update({
                'username': username,
                'password': password,
                'uid': auth_data.get('uid', 'BlinkCamera_' + str(uuid.uuid4())),
                'device_id': auth_data.get('device_id', 'Blinkpy')
            })

            # Save updated credentials back to file
            print("Saving updated authentication credentials...")
            with open(CREDENTIALS_FILE, 'w') as f:
                json.dump(updated_creds, f)
        else:
            # Use existing credentials
            auth = Auth(auth_data)
            blink.auth = auth
            await blink.start()

        await blink.refresh(force=True)

        # Print account info for debugging
        print("\nBlink Account Information:")
        if blink.auth.login_attributes:
            for key, value in blink.auth.login_attributes.items():
                if key not in ['password', 'token']:  # Don't print sensitive info
                    print(f"{key}: {value}")

        # Print network info
        print("\nBlink Networks:")
        for network_id, network in blink.networks.items():
            if isinstance(network, dict):
                print(f"Network ID: {network_id}")
                for key, value in network.items():
                    if key != 'cameras':  # Don't print the entire camera objects
                        print(f"  {key}: {value}")

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


async def get_cloud_videos(blink, max_videos=1):
    """Fetch the single most recent video from today from Blink cloud or local storage"""
    try:
        print("\n🔍 Retrieving the most recent video from today...")
        await blink.refresh(force=True)

        # Get account ID directly from auth attributes
        account_id = None
        if hasattr(blink, 'auth') and hasattr(blink.auth, 'login_attributes'):
            account_id = blink.auth.login_attributes.get('account_id')
            print(f"Found account ID in auth attributes: {account_id}")

        if not account_id:
            # Find the doorbell camera and network
            doorbell = None
            for name, camera in blink.cameras.items():
                if name.lower() == "doorbell" or "door" in name.lower():
                    doorbell = camera
                    print(f"Found doorbell camera: {name}")
                    break

            if not doorbell:
                print(
                    "❌ Could not find doorbell camera, trying to use any available camera")
                if blink.cameras:
                    doorbell = next(iter(blink.cameras.values()))
                    print(f"Using camera: {doorbell.name}")
                else:
                    print("❌ No cameras found")
                    return []

            # Try to find the account ID from networks
            for network_id_key, network in blink.networks.items():
                if isinstance(network, dict) and 'account_id' in network:
                    account_id = network['account_id']
                    print(f"Found account ID in networks: {account_id}")
                    break

        if not account_id:
            print("❌ Could not determine account ID")
            return []

        print(f"Using Account ID: {account_id}")

        # Get the base URL and auth token
        base_url = None
        if hasattr(blink, 'urls') and hasattr(blink.urls, 'base_url'):
            base_url = blink.urls.base_url
        else:
            host = 'rest-prod.immedia-semi.com'  # Default host
            if hasattr(blink, 'auth') and hasattr(blink.auth, 'login_attributes') and blink.auth.login_attributes:
                host = blink.auth.login_attributes.get('host', host)
            base_url = f"https://{host}"
            print(f"Using constructed base URL: {base_url}")

        # Get auth headers directly from the auth object
        headers = {}
        if hasattr(blink, 'auth'):
            if hasattr(blink.auth, 'header') and blink.auth.header:
                headers = blink.auth.header
                print("Using auth.header for authentication")
            elif hasattr(blink.auth, 'token') and blink.auth.token:
                token = blink.auth.token
                headers = {'TOKEN_AUTH': token}
                print("Using constructed TOKEN_AUTH header")

        if not headers:
            print("❌ Could not determine authentication headers")
            return []

        print(f"Headers: {list(headers.keys())}")
        videos_found = []

        # Calculate the 'since' time as the beginning of the current day
        today_start_dt = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0)
        since_str = today_start_dt.strftime('%Y-%m-%dT%H:%M:%S%z')
        if len(since_str) > 5 and since_str[-3] != ':':
            since_str = since_str[:-2] + ':' + since_str[-2:]
        since_param = since_str.replace('+', '%2B')
        print(f"Using 'since' parameter for today: {since_param}")

        async with ClientSession() as session:
            # APPROACH 1: Try the event history API first (has doorbell rings)
            print("\n🔔 Checking doorbell events API for today...")
            events_url = f"{base_url}/api/v1/accounts/{account_id}/events?page=1&since={since_param}"

            try:
                async with session.get(events_url, headers=headers) as response:
                    print(f"Events API response status: {response.status}")
                    if response.status == 200:
                        events_data = await response.json()
                        if 'events' in events_data:
                            print(
                                f"Found {len(events_data['events'])} total events today")
                            doorbell_events = 0

                            for event in events_data['events']:
                                if 'video_url' in event:
                                    is_doorbell_event = (
                                        event.get('camera_name', '').lower() == 'doorbell' or
                                        'door' in event.get('camera_name', '').lower() or
                                        event.get('type', '') == 'motion' or
                                        event.get('type', '') == 'doorbell'
                                    )

                                    if is_doorbell_event:
                                        doorbell_events += 1
                                        videos_found.append({
                                            'id': event.get('id', f"event_{len(videos_found)}"),
                                            'created_at': event.get('created_at', 'unknown'),
                                            'device_name': event.get('camera_name', 'unknown'),
                                            'event_type': event.get('type', 'unknown'),
                                            'video_url': event['video_url']
                                        })

                            print(
                                f"Found {doorbell_events} doorbell/motion events with videos today")
                    else:
                        error_text = await response.text()
                        print(f"Events API error: {error_text}")
            except Exception as e:
                print(f"Error accessing events API: {str(e)}")
                import traceback
                traceback.print_exc()

            # APPROACH 2: Try the media API (has all saved videos)
            print("\n📹 Checking media API for today's saved videos...")
            media_url = f"{base_url}/api/v1/accounts/{account_id}/media/changed?since={since_param}"

            try:
                async with session.get(media_url, headers=headers) as response:
                    print(f"Media API response status: {response.status}")
                    if response.status == 200:
                        media_data = await response.json()
                        if 'media' in media_data:
                            print(
                                f"Found {len(media_data['media'])} media items today")

                            for media in media_data['media']:
                                if any(v.get('id') == media.get('id') for v in videos_found):
                                    continue

                                video_url = None
                                media_value = media.get('media')
                                if isinstance(media_value, dict):
                                    video_url = media_value.get(
                                        'mp4') or media_value.get('mp4_url')
                                elif isinstance(media_value, str) and media_value.endswith('.mp4'):
                                    video_url = media_value
                                elif 'address' in media:
                                    video_url = media.get('address')

                                if video_url:
                                    if not any(v.get('id') == media.get('id') for v in videos_found):
                                        videos_found.append({
                                            'id': media.get('id', f"media_{len(videos_found)}"),
                                            'created_at': media.get('created_at', 'unknown'),
                                            'device_name': media.get('device_name', 'unknown'),
                                            'event_type': 'media',
                                            'video_url': video_url
                                        })
                    else:
                        error_text = await response.text()
                        print(f"Media API error: {error_text}")
            except Exception as e:
                print(f"Error accessing media API: {str(e)}")
                import traceback
                traceback.print_exc()

            # APPROACH 3: Try another media endpoint format
            print("\n📹 Trying alternative media endpoint for today...")
            alt_media_url = f"{base_url}/api/v2/videos/changed?since={since_param}"

            try:
                async with session.get(alt_media_url, headers=headers) as response:
                    print(
                        f"Alternative Media API response status: {response.status}")
                    if response.status == 200:
                        media_data = await response.json()
                        if 'videos' in media_data:
                            videos_list = media_data['videos']
                            print(
                                f"Found {len(videos_list)} videos in alternative endpoint today")

                            for video in videos_list:
                                if 'address' in video:
                                    if not any(v.get('id') == video.get('id') for v in videos_found):
                                        videos_found.append({
                                            'id': video.get('id', f"alt_video_{len(videos_found)}"),
                                            'created_at': video.get('created_at', 'unknown'),
                                            'device_name': video.get('camera_name', 'unknown'),
                                            'event_type': 'video',
                                            'video_url': video['address']
                                        })
                    else:
                        error_text = await response.text()
                        print(f"Alternative Media API error: {error_text}")
            except Exception as e:
                print(f"Error accessing alternative media API: {str(e)}")
                import traceback
                traceback.print_exc()

            # APPROACH 4: Try Local Storage (Sync Module 2 USB)
            sync_id = None
            network_id_for_local = None
            sync_module_name = None
            if hasattr(blink, 'sync') and blink.sync:
                # Get the first network ID (key) and sync module (value)
                network_id_for_local = next(
                    iter(blink.sync.keys()))  # Get the key (ID)
                sync_module = blink.sync[network_id_for_local]
                if hasattr(sync_module, 'sync_id'):
                    sync_id = sync_module.sync_id
                elif hasattr(sync_module, 'id'):
                    sync_id = sync_module.id
                else:
                    print("   Could not find ID attribute for sync module.")

                sync_module_name = sync_module.name
                if sync_id:
                    # Ensure network_id_for_local is the numerical ID
                    print(
                        f"\n💾 Found Sync Module '{sync_module_name}' (ID: {sync_id}) on Network ID: {network_id_for_local}")
                else:
                    print(
                        f"\n💾 Found Sync Module '{sync_module_name}' but could not determine its ID.")
            else:
                print("\n💾 No Sync Modules found in blink object.")

            if sync_id and network_id_for_local:
                # Ensure network_id_for_local is used correctly in the URL
                print(
                    f"   Requesting local storage manifest for Network ID {network_id_for_local}...")
                manifest_request_url = f"{base_url}/api/v1/accounts/{account_id}/networks/{network_id_for_local}/sync_modules/{sync_id}/local_storage/manifest/request"
                manifest_request_id = None
                try:
                    async with session.post(manifest_request_url, headers=headers) as response:
                        print(
                            f"   Local Manifest Request API status: {response.status}")
                        if response.status == 200 or response.status == 202:
                            manifest_req_data = await response.json()
                            manifest_request_id = manifest_req_data.get('id')
                            print(
                                f"   Manifest request submitted. Request ID: {manifest_request_id}")
                        else:
                            error_text = await response.text()
                            print(
                                f"   Local Manifest Request API error: {error_text}")
                except Exception as e:
                    print(f"   Error requesting local manifest: {str(e)}")

                if manifest_request_id:
                    manifest_url = f"{base_url}/api/v1/accounts/{account_id}/networks/{network_id_for_local}/sync_modules/{sync_id}/local_storage/manifest/request/{manifest_request_id}"
                    manifest_data = None
                    manifest_id = None
                    clips = []
                    print("   Polling for manifest completion...")
                    for attempt in range(6):
                        await asyncio.sleep(2)
                        try:
                            async with session.get(manifest_url, headers=headers) as response:
                                print(
                                    f"   Manifest Retrieval Poll status: {response.status} (Attempt {attempt+1})")
                                if response.status == 200:
                                    manifest_data = await response.json()
                                    if manifest_data.get("status") == "complete":
                                        manifest_id = manifest_data.get(
                                            "manifest_id")
                                        clips = manifest_data.get("clips", [])
                                        print(
                                            f"   ✅ Local Manifest retrieved successfully. Found {len(clips)} clips.")
                                        break
                                    else:
                                        print(
                                            f"   Manifest status: {manifest_data.get('status')}. Polling again...")
                                elif response.status == 202:
                                    print(
                                        "   Manifest still processing. Polling again...")
                                else:
                                    error_text = await response.text()
                                    print(
                                        f"   Local Manifest Retrieval API error: {error_text}")
                                    break
                        except Exception as e:
                            print(
                                f"   Error retrieving local manifest poll: {str(e)}")
                            break

                    if clips and manifest_id:
                        today_clips = []
                        for clip in clips:
                            created_at_str = clip.get('created_at')
                            if created_at_str:
                                try:
                                    clip_dt = datetime.fromisoformat(
                                        created_at_str.replace('Z', '+00:00'))
                                    if clip_dt.date() >= today_start_dt.date():
                                        today_clips.append(clip)
                                except ValueError:
                                    print(
                                        f"   Warning: Could not parse timestamp for local clip ID {clip.get('id')}: {created_at_str}")

                        if not today_clips:
                            print(
                                "   No local clips found from today in the manifest.")
                        else:
                            print(
                                f"   Found {len(today_clips)} local clips from today.")
                            try:
                                today_clips.sort(key=lambda x: x.get(
                                    'created_at', ''), reverse=True)
                            except Exception as sort_e:
                                print(
                                    f"   Warning: Could not sort today's local clips: {sort_e}")

                            latest_clip = today_clips[0]
                            clip_id = latest_clip.get("id")
                            clip_timestamp = latest_clip.get(
                                "created_at", "unknown")
                            clip_camera_name = latest_clip.get(
                                "camera_name", sync_module_name or "local_storage")

                            if clip_id:
                                if any(v.get('created_at') == clip_timestamp for v in videos_found):
                                    print(
                                        f"   Skipping latest local clip (ID: {clip_id}) as it might already be found via cloud API.")
                                else:
                                    print(
                                        f"   Requesting upload for latest local clip ID: {clip_id} from {clip_camera_name} at {clip_timestamp}")
                                    clip_request_url = f"{base_url}/api/v1/accounts/{account_id}/networks/{network_id_for_local}/sync_modules/{sync_id}/local_storage/manifest/{manifest_id}/clip/request/{clip_id}"
                                    clip_download_url = clip_request_url

                                    try:
                                        async with session.post(clip_request_url, headers=headers) as response:
                                            print(
                                                f"   Local Clip Upload Request API status: {response.status}")
                                            if response.status == 200 or response.status == 202:
                                                print(
                                                    f"   Upload requested for clip {clip_id}. Adding to list for download.")
                                                videos_found.append({
                                                    'id': f"local_{clip_id}",
                                                    'created_at': clip_timestamp,
                                                    'device_name': clip_camera_name,
                                                    'event_type': 'local_clip',
                                                    'video_url': clip_download_url
                                                })
                                            else:
                                                error_text = await response.text()
                                                print(
                                                    f"   Local Clip Upload Request API error: {error_text}")
                                    except Exception as e:
                                        print(
                                            f"   Error requesting local clip upload: {str(e)}")

                    elif manifest_request_id:
                        print(
                            "   ❌ Failed to retrieve completed manifest after polling.")

        if not videos_found:
            print(
                "❌ No videos found from today in Blink cloud storage or local storage manifest")
            return []

        sorted_videos = sorted(
            videos_found,
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )

        most_recent_video = sorted_videos[0]
        print(
            f"\n✅ Found the most recent video from today: {most_recent_video.get('event_type')} from {most_recent_video.get('device_name')} at {most_recent_video.get('created_at')}")
        return [most_recent_video]

    except Exception as e:
        print(f"Error getting cloud videos: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


async def process_cloud_videos(blink):
    """Process videos from Blink cloud storage (now expects a list with 0 or 1 video)"""
    try:
        videos = await get_cloud_videos(blink)

        if not videos:
            print("No recent video found from today.")
            return False

        video = videos[0]
        print(f"\n📽️ Processing the latest video from today...")
        videos_processed = 0

        video_id = video.get('id')
        created_at = video.get('created_at')
        device_name = video.get('device_name')
        event_type = video.get('event_type', 'unknown')

        print(
            f"\n📹 Processing {event_type} video from {device_name} created at {created_at}")

        log_check = f"{created_at}_{video_id}"
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                if log_check in f.read():
                    print(f"Video {video_id} already processed, skipping")
                    return False

        video_url = video.get('video_url')
        if not video_url:
            print("❌ No video URL found, skipping")
            return False

        # Prepend base_url if the video_url is relative
        if video_url.startswith('/'):
            # Need the base_url from the blink object or constructed earlier
            # Let's re-fetch/construct it here for simplicity, though passing it would be cleaner
            constructed_base_url = None
            if hasattr(blink, 'urls') and hasattr(blink.urls, 'base_url'):
                constructed_base_url = blink.urls.base_url
            else:
                host = 'rest-prod.immedia-semi.com'
                if hasattr(blink, 'auth') and hasattr(blink.auth, 'login_attributes') and blink.auth.login_attributes:
                    host = blink.auth.login_attributes.get('host', host)
                constructed_base_url = f"https://{host}"

            if constructed_base_url:
                full_video_url = f"{constructed_base_url}{video_url}"
                print(f"Constructed full video URL: {full_video_url}")
            else:
                print(
                    "❌ Could not determine base_url to construct full video URL, skipping")
                return False
        else:
            full_video_url = video_url  # Assume it's already a full URL

        print(f"Using Video URL for download: {full_video_url}")

        event_date = datetime.now().strftime("%Y%m%d")
        try:
            if created_at and created_at != 'unknown':
                if 'T' in created_at:
                    dt = datetime.fromisoformat(
                        created_at.replace('Z', '+00:00'))
                    event_date = dt.strftime("%Y%m%d")
        except Exception as e:
            print(f"Couldn't parse date: {str(e)}")

        event_tag = event_type if event_type != 'unknown' else 'event'
        timestamp = datetime.now().strftime("%H%M%S")
        video_filename = f"{event_tag}_{event_date}_{timestamp}_{device_name}.mp4"
        video_path = os.path.join(DEBUG_DIR, video_filename)

        headers = {}
        if hasattr(blink.auth, 'header'):
            headers = blink.auth.header
        elif hasattr(blink, 'auth') and hasattr(blink.auth, 'token'):
            token = blink.auth.token
            if token:
                headers = {'TOKEN_AUTH': token}

        async with ClientSession() as session:
            print(f"Downloading {event_type} video...")
            try:
                async with session.get(full_video_url, headers=headers) as response:
                    if response.status == 200:
                        video_content = await response.read()

                        if len(video_content) > 1000:
                            with open(video_path, 'wb') as f:
                                f.write(video_content)
                            print(
                                f"✅ Saved video to {video_path} ({len(video_content) / 1024:.1f} KB)")

                            print(f"🔍 Analyzing video for faces...")
                            visitor = await process_video_for_recognition(video_path, timestamp)

                            if visitor != "unknown":
                                message = family_messages.get(
                                    visitor, family_messages['unknown'])
                                print(f"👋 Recognition result: {visitor}")
                                print(f"   Greeting: {message}")
                            else:
                                print(
                                    "👤 No familiar faces recognized in this video")

                            await log_processed_media("cloud_video", video_path, visitor, None)

                            with open(LOG_FILE, 'a') as f:
                                f.write(f"Video ID: {log_check}\n")

                            videos_processed = 1
                        else:
                            print(
                                f"⚠️ Video content too small ({len(video_content)} bytes), might be invalid")
                    else:
                        error_text = await response.text()
                        print(
                            f"❌ Failed to download video: {response.status}")
                        print(f"   Error: {error_text}")
            except Exception as e:
                print(f"❌ Error during video download: {str(e)}")
                import traceback
                traceback.print_exc()

        if videos_processed > 0:
            print(f"\n✅ Latest video processing complete!")
            return True
        else:
            print(f"\n⚠️ Latest video processing failed or video was invalid.")
            return False

    except Exception as e:
        print(f"Error processing cloud videos: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


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
                label, confidence = face_recognizer.predict(
                    face)  # Correct indentation
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
    blink = None
    try:
        print("🔄 Connecting to Blink...")
        blink = await authenticate()
        print("✅ Connected to Blink successfully")

        # Process existing cloud videos
        print("\n🔍 Checking Blink cloud storage for existing videos...")
        await process_cloud_videos(blink)

        print("\n✅ Cloud video processing complete!")
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
