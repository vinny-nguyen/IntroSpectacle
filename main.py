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

def upload_photo_to_mongodb(filename, name, summary):
    try:
        with open(filename, 'rb') as f:
            contents = f.read()
            fs.put(contents, filename=filename, metadata={'name': name, 'summary': summary})
            print(f"Uploaded {filename} to MongoDB with metadata.")
    except Exception as e:
        print(f"An error occurred while uploading to MongoDB: {e}")


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
                if grid_out.metadata is not None:
                    known_face_names.append(grid_out.metadata.get('name', 'Unknown'))
                    known_face_summaries.append(grid_out.metadata.get('summary', ''))
                else:
                    # Use default values if metadata is missing
                    known_face_names.append('Unknown')
                    known_face_summaries.append('')
            else:
                print(f"No face found in image {grid_out.filename}")

    return known_face_encodings, known_face_names, known_face_summaries














def textToWord():
    doc = aw.Document("transcription.txt")
    doc.save("transcription.docx") 
    #doc = aw.Document("summaryco.txt")
    #doc.save("summaryco.docx") 


def main():
    # Load known faces and metadata
    known_face_encodings, known_face_names, known_face_summaries = load_known_faces_from_db()

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    recording = False
    video_writer = None
    video_count = 1  # Counter for saved videos

    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Convert the BGR image to RGB.
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            ih, iw, ic = frame.shape

            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Initialize variables
            face_names = []
            face_summaries = []

            for face_encoding in face_encodings:
                # See if the face is a match for known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                summary = ""

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches and matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    summary = known_face_summaries[best_match_index]

                face_names.append(name)
                face_summaries.append(summary)

            # Display the results
            for (top, right, bottom, left), name, summary in zip(face_locations, face_names, face_summaries):
                # Draw a rectangle around the face
                cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Set the box size
                box_width, box_height = 200, 150  # Size of the white box

                # Position the box to the right of the head
                x = right + 10  # 10 pixels to the right of the face bounding box
                y = top + int((bottom - top) / 2) - int(box_height / 2)  # Centered vertically

                # Draw the white box
                cv.rectangle(frame, (x, y), (x + box_width, y + box_height), (255, 255, 255), -1)

                # Write the name and summary onto the white box
                cv.putText(frame, name, (x + 5, y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                # Split summary into lines if necessary
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
                    filename = "photo.png"
                    cv.imwrite(filename, frame)

                    # Start recording
                    filename = "video.avi"
                    fourcc = cv.VideoWriter_fourcc(*'XVID')
                    fps = 20.0
                    frame_size = (frame.shape[1], frame.shape[0])
                    video_writer = cv.VideoWriter(filename, fourcc, fps, frame_size)
                    print(f"Started recording: {filename}")
                else:
                    # Stop recording
                    video_writer.release()
                    print(f"Stopped recording: video_{video_count}.avi")
                    video_count += 1
                    video_writer = None
                    name, summary = transcribe()
                    upload_photo_to_mongodb(f"{name}.png", name, summary)

                    # Reload known faces after adding new one
                    known_face_encodings, known_face_names, known_face_summaries = load_known_faces_from_db()

            elif key == 27:  # 'Esc' key to exit
                if recording:
                    video_writer.release()
                break

        cap.release()
        cv.destroyAllWindows()



def transcribe():
    result = model.transcribe("audio.mp3")
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
    return summaryname, summarywords  # Returning name and summary for later use



def summarize_name(text):
    prompt = f" extract the name from here, only the name should be displayed no other words just name: {text}"
    
    #Use Cohere's summarize endpoint
    response = co.chat(

        message= prompt,
        model='command-r-plus-08-2024',  # Choose the appropriate model
        temperature=0.5,  # Controls randomness
    )

    
    return response.text

def summarize_words(text):
    prompt = f" extract 3 to 4 key words that represent the person e.g sports, leetcode, video games, no other words just these 3-4 key words: {text}"
    
    #Use Cohere's summarize endpoint
    response = co.chat(

        message= prompt,
        model='command-r-plus-08-2024',  # Choose the appropriate model
        temperature=0.5,  # Controls randomness
    )

    
    return response.text


def summarize_transcription(text):
    prompt = f"{text}"

    if len(prompt) < 250:  # If less than 250 chars
        return ''  # Return empty string, no summary needed
    
    # Use Cohere's summarize endpoint
    response = co.summarize(
        text=prompt,
        length="short",
        format="paragraph",
        model='summarize-medium',  # Choose the appropriate model
        temperature=0.5,  # Controls randomness
        additional_command="add an empty line and then add the summary with the bullet points in a new line below the empty one"
    )

    summary = response.summary
    return summary


if __name__ == "__main__":
    main()
