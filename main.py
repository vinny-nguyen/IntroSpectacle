import cv2 as cv
import mediapipe as mp
import time
import whisper
import cohere
import os
import aspose.words as aw
from dotenv import load_dotenv
import pymongo
import gridfs
import numpy as np
import face_recognition
import re
import threading
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment

load_dotenv()

cohere_api_key = os.environ.get('COHERE_API_KEY')
co = cohere.Client(cohere_api_key)

model = whisper.load_model("base")
if not cohere_api_key:
    raise ValueError("No Cohere API key found. Please set the COHERE_API_KEY environment variable.")

mongodb_uri = os.environ.get('MONGODB_URI')
if not mongodb_uri:
    raise ValueError("No MongoDB URI found. Please set the MONGODB_URI environment variable.")

client = pymongo.MongoClient(mongodb_uri)
db = client['mydatabase']  # Replace 'mydatabase' with your database name
fs = gridfs.GridFS(db)

def sanitize_filename(name):
    # Remove any characters that are invalid in filenames
    return re.sub(r'[\\/*?:"<>|]', "", name)

def load_known_faces_from_db():
    known_face_encodings = []
    known_face_names = []
    known_face_summaries = []

    for grid_out in fs.find():
        if grid_out.filename.endswith('.png'):  # Adjust if you have other formats
            # Read image data from GridFS
            img_data = grid_out.read()
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv.imdecode(nparr, cv.IMREAD_COLOR)

            # Convert the image from BGR to RGB
            rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_img)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])

                # Handle cases where metadata is None
                metadata = grid_out.metadata or {}
                known_face_names.append(metadata.get('name', 'Unknown'))
                known_face_summaries.append(metadata.get('summary', ''))
            else:
                print(f"No face found in image {grid_out.filename}")

    return known_face_encodings, known_face_names, known_face_summaries

def upload_photo_to_mongodb(filename, name, summary):
    try:
        with open(filename, 'rb') as f:
            contents = f.read()
            fs.put(contents, filename=filename, metadata={'name': name.strip(), 'summary': summary.strip()})
            print(f"Uploaded {filename} to MongoDB with metadata.")
    except Exception as e:
        print(f"An error occurred while uploading to MongoDB: {e}")

def textToWord():
    doc = aw.Document("transcription.txt")
    doc.save("transcription.docx")

def audio_recording_thread(stop_event):
    samplerate = 44100  # Sampling rate
    channels = 1        # Number of audio channels
    filename = 'output.wav'  # Temporary WAV file

    with sf.SoundFile(filename, mode='w', samplerate=samplerate, channels=channels) as file:
        def callback(indata, frames, time, status):
            if status:
                print(status)
            file.write(indata.copy())

        with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
            while not stop_event.is_set():
                sd.sleep(100)

def main():
    # Load known faces and metadata
    known_face_encodings, known_face_names, known_face_summaries = load_known_faces_from_db()

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    facial_recognition_enabled = True  # Control flag for facial recognition
    recording = False
    video_writer = None
    video_count = 1  # Counter for saved videos

    face_locations = []
    face_names = []
    face_summaries = []

    # Adjust this value to process face recognition every N frames
    process_every_n_frames = 10  # Increase this number to make it faster
    frame_counter = 0

    # Create a lock for thread synchronization
    lock = threading.Lock()

    # Initialize audio recording control variables
    audio_stop_event = threading.Event()
    audio_thread = None

    def face_recognition_thread(rgb_small_frame):
        nonlocal face_locations, face_names, face_summaries
        # Perform face detection and recognition
        # Use Mediapipe for face detection
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        results = face_detection.process(rgb_small_frame)
        temp_face_locations = []

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = rgb_small_frame.shape
                x_min = int(bboxC.xmin * iw)
                y_min = int(bboxC.ymin * ih)
                box_width = int(bboxC.width * iw)
                box_height = int(bboxC.height * ih)
                x_max = x_min + box_width
                y_max = y_min + box_height
                temp_face_locations.append((y_min, x_max, y_max, x_min))

        temp_face_encodings = face_recognition.face_encodings(rgb_small_frame, temp_face_locations)
        temp_face_names = []
        temp_face_summaries = []

        for face_encoding in temp_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            summary = ""
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    summary = known_face_summaries[best_match_index]
            temp_face_names.append(name)
            temp_face_summaries.append(summary)

        with lock:
            face_locations = temp_face_locations
            face_names = temp_face_names
            face_summaries = temp_face_summaries

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ih, iw, ic = frame.shape

        # Only perform facial recognition if enabled
        if facial_recognition_enabled:
            # Resize frame for faster processing
            small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

            frame_counter += 1

            if frame_counter % process_every_n_frames == 0:
                # Start a new thread for face recognition
                thread = threading.Thread(target=face_recognition_thread, args=(rgb_small_frame,))
                thread.start()

            # Display the results
            with lock:
                for (top, right, bottom, left), name, summary in zip(face_locations, face_names, face_summaries):
                    # Scale back up face locations since the frame we detected in was scaled to 0.5
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2

                    # Draw a rectangle around the face
                    # cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Set the box size
                    box_width, box_height = 200, 150  # Size of the white box

                    # Position the box to the right of the head
                    x = right + 10  # 10 pixels to the right of the face bounding box
                    y = top + int((bottom - top) / 2) - int(box_height / 2)  # Centered vertically

                    # Draw the white box
                    cv.rectangle(frame, (x, y), (x + box_width, y + box_height), (255, 255, 255), -1)

                    # Write the name and summary onto the white box
                    cv.putText(frame, name, (x + 5, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    # Split summary into lines
                    summary_lines = summary.split('\n')
                    for idx, line in enumerate(summary_lines):
                        cv.putText(frame, line, (x + 5, y + 40 + idx * 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Write frame to video if recording
        if recording:
            video_writer.write(frame)
            cv.putText(frame, 'Recording...', (10, ih - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv.imshow('Head Scanner', frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            recording = not recording  # Toggle recording state

            if recording:
                # Disable facial recognition
                facial_recognition_enabled = False

                # Start audio recording
                audio_stop_event.clear()
                audio_thread = threading.Thread(target=audio_recording_thread, args=(audio_stop_event,))
                audio_thread.start()
                print("Started audio recording")

                # Start video recording
                filename = "video.avi"
                fourcc = cv.VideoWriter_fourcc(*'XVID')
                fps = 60.0
                frame_size = (frame.shape[1], frame.shape[0])
                video_writer = cv.VideoWriter(filename, fourcc, fps, frame_size)
                print(f"Started video recording: {filename}")
            else:
                # Stop audio recording
                audio_stop_event.set()
                audio_thread.join()
                print("Stopped audio recording")

                # Convert output.wav to output.mp3
                sound = AudioSegment.from_wav('output.wav')
                sound.export('output.mp3', format='mp3')
                print("Converted audio to MP3 format")

                # Stop video recording
                video_writer.release()
                print(f"Stopped video recording: video_{video_count}.avi")
                video_count += 1
                video_writer = None

                # Re-enable facial recognition
                facial_recognition_enabled = True

                # Proceed with transcription and other post-processing
                name, summary = transcribe()
                sanitized_name = sanitize_filename(name)
                cv.imwrite(f"{sanitized_name}.png", frame)
                upload_photo_to_mongodb(f"{sanitized_name}.png", name, summary)

                # Reload known faces after adding new one
                known_face_encodings, known_face_names, known_face_summaries = load_known_faces_from_db()

        elif key == 27:  # 'Esc' key to exit
            if recording:
                # Ensure resources are released properly
                audio_stop_event.set()
                audio_thread.join()
                video_writer.release()
            break

    cap.release()
    cv.destroyAllWindows()

def transcribe():
    result = model.transcribe("output.mp3")
    transcription_text = result['text']
    summary = summarize_transcription(transcription_text)
    with open('transcription.txt', 'w', encoding='utf-8') as f:
        f.write(transcription_text)
        if summary:  # Only write the summary if it's not empty
            f.write('\n\nSummary:\n')
            f.write(summary)
    textToWord()
    summaryname = summarize_name(transcription_text)
    with open("name.txt", "w", encoding='utf-8') as f:
        f.write(summaryname)
    summarywords = summarize_words(transcription_text)
    with open("words.txt", "w", encoding='utf-8') as f:
        f.write(summarywords)
    return summaryname.strip(), summarywords.strip()  # Returning name and summary for later use

def summarize_name(text):
    prompt = f"Extract the name from the following text. Only the name should be displayed, no other words: {text}"
    response = co.chat(
        message=prompt,
        model='command-xlarge-nightly',  # Use the appropriate model
        temperature=0.5,  # Controls randomness
    )
    return response.text.strip()

def summarize_words(text):
    prompt = f"Extract 3 to 4 keywords that represent the person (e.g., sports, leetcode, video games). Only these 3-4 keywords should be displayed: {text}"
    response = co.chat(
        message=prompt,
        model='command-xlarge-nightly',  # Use the appropriate model
        temperature=0.5,  # Controls randomness
    )
    return response.text.strip()

def summarize_transcription(text):
    prompt = f"{text}"

    if len(prompt) < 250:  # If less than 250 chars
        return ''  # Return empty string, no summary needed

    response = co.summarize(
        text=prompt,
        length="short",
        format="paragraph",
        model='summarize-medium',  # Choose the appropriate model
        temperature=0.5,  # Controls randomness
        additional_command="Add an empty line and then add the summary with bullet points in a new line below the empty one"
    )

    summary = response.summary
    return summary.strip()

if __name__ == "__main__":
    main()
