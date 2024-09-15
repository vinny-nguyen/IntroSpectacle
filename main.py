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


def textToWord():
    doc = aw.Document("transcription.txt")
    doc.save("transcription.docx") 
    #doc = aw.Document("summaryco.txt")
    #doc.save("summaryco.docx") 


def main():
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
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # To improve performance, mark the image as not writeable.
            image.flags.writeable = False
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            ih, iw, ic = image.shape

            if results.detections:
                # Find the detection with the largest bounding box area
                max_area = 0
                largest_detection = None

                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    bbox_area = bboxC.width * iw * bboxC.height * ih
                    if bbox_area > max_area:
                        max_area = bbox_area
                        largest_detection = detection

                if largest_detection is not None:
                    detection = largest_detection

                    # Get bounding box coordinates
                    bboxC = detection.location_data.relative_bounding_box
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                           int(bboxC.width * iw), int(bboxC.height * ih)

                    # Set the box size to 200x150 pixels
                    box_width, box_height = 200, 150  # Size of the white box

                    # Position the box to the right of the head
                    x = bbox[0] + bbox[2] + 10  # 10 pixels to the right of the face bounding box
                    y = bbox[1] + int(bbox[3] / 2) - int(box_height / 2)  # Centered vertically

                    # Draw the white box
                    cv.rectangle(image, (x, y), (x + box_width, y + box_height), (255, 255, 255), -1)

            # Write frame to video if recording
            if recording:
                video_writer.write(image)
                cv.putText(image, 'Recording...', (10, ih - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv.imshow('Head Scanner', image)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                recording = not recording  # Toggle recording state

                if recording:
                    filename = "photo.png"
                    cv.imwrite(filename, image)
                    
                    # Start recording
                    filename = "video.avi"
                    fourcc = cv.VideoWriter_fourcc(*'XVID')
                    fps = 20.0
                    frame_size = (image.shape[1], image.shape[0])
                    video_writer = cv.VideoWriter(filename, fourcc, fps, frame_size)
                    print(f"Started recording: {filename}")
                else:
                    # Stop recording
                    video_writer.release()
                    print(f"Stopped recording: video_{video_count}.avi")
                    video_count += 1
                    video_writer = None
                    name,summary = transcribe()
                    upload_photo_to_mongodb("photo.png", name, summary)

            elif key == 27:  # 'Esc' key to exit
                if recording:
                    video_writer.release()
                break

        cap.release()
        cv.destroyAllWindows()


def transcribe():
    result = model.transcribe("audio.wav")
    transcription_text = result['text']
    summary = summarize_transcription(transcription_text)
    with open('transcription.txt', 'w', encoding='utf-8') as f:
        f.write(transcription_text)
        f.write(summary)
    textToWord()
    summaryname = summarize_name(transcription_text)
    with open("name.txt", "w", encoding='utf-8') as f:
        f.write(summaryname)
    summarywords = summarize_words(transcription_text)
    with open("words.txt", "w", encoding='utf-8') as f:
        f.write(summarywords)
    return summaryname, summary #returning name and summary for later use


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

    if len(prompt) < 250: #if less than 250 chars
        summary = prompt
        return summary
    
    #Use Cohere's summarize endpoint
    response = co.summarize(
        text=prompt,
        length= "short",
        format = "paragraph",
        model='summarize-medium',  # Choose the appropriate model
        temperature=0.5,  # Controls randomness
        additional_command= "add an empty line and then add the summary with the bullet points in a new line below the empty one"
    )

    summary = response.summary
    return summary


if __name__ == "__main__":
    main()
