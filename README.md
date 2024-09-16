© IntroSpectacle. Erkam Boyacioglu, Ruddra Kantaria, Joyi Xue, and Vincent Nguyen @ Hack the North 2024. University of Waterloo, Waterloo, Ontario, Canada. All rights reserved.

## IntroSpectacle 👓
IntroSpectacle is a real-time facial detection system that helps users remember names, conversations, and hobbies during social interactions. Designed for first-year university students and older adults alike, it makes meeting new people easier and more meaningful by providing instant context and reminders.

## Inspiration 🌟
The inspiration for this project stems from the challenges of meeting many new people at once and struggling to remember names, conversations, and hobbies. This is common for students, particularly during events like frosh week, and for adults transitioning to a new job.

IntroSpectacle is also designed with older adults in mind, especially those experiencing memory difficulties. It helps them recall names, past interactions, and personal interests of those they meet, making social interactions easier and more meaningful. This project is for people of all ages, from young university students to wise grandparents.

## What it does 🤖
IntroSpectacle uses OpenCV and MediaPipe to recognize faces in real-time. It displays the person’s name and key information next to their face, simplifying social interactions. Additionally, the Cohere API is used to summarize conversations, generate keywords, and extract names, ensuring users remember important details about each person for future interactions.

## Features
Face Recognition: Detect and identify faces using OpenCV and MediaPipe.
Conversation Summarization: Automatically summarize conversations using the Cohere API to help recall important details.
Audio Recording: Record audio in sync with video input using PyAudio for later analysis.
Data Storage: Use MongoDB to store images, names, and metadata, ensuring seamless data retrieval and usage in real-time.
How we built it 🏗️
Facial Detection: Built using OpenCV and MediaPipe for real-time video input and facial detection.
Audio Integration: Integrated PyAudio for synchronized audio recording during video interactions.
Data Storage: MongoDB was used to store captured images, metadata, and facial recognition data.
Conversation Summarization: Implemented Cohere to extract names and summarize key points from conversations.
Challenges we faced 🎯
Our biggest challenge was narrowing down the project scope. We started with too many ideas, and it took time to settle on one clear, achievable concept. Being first-time hackers, we also faced a steep learning curve, from learning new libraries to debugging issues. Nevertheless, with support from mentors and volunteers, we powered through and embraced every obstacle.

## Accomplishments we’re proud of 🥇
We’re proud of our progress, especially since this is the first hackathon for most of us. From collaborating to squash bugs to learning new technologies, we constantly pushed ourselves out of our comfort zones. It's been a rewarding experience of growth, both in terms of technical skills and teamwork.

## What we learned 💡
We learned the value of perseverance, collaboration, and problem-solving under pressure. From understanding how OpenCV’s facial recognition works to integrating audio and video files simultaneously in Visual Studio, this hackathon has been a non-stop learning experience.

## What’s next for IntroSpectacle 🔮
Our mission is to develop IntroSpectacle into a practical and user-friendly product. We aim to transform the webcam setup into a smaller, sleeker device, with enhanced features to better support both young students and older adults with memory challenges.

