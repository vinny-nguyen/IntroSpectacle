const webcam = document.getElementById('webcam');
const startRecordingButton = document.getElementById('start-recording');
const stopRecordingButton = document.getElementById('stop-recording');
let mediaRecorder;
let chunks = [];

navigator.mediaDevices.getUserMedia({ video: true, audio: true })
    .then(stream => {
        webcam.srcObject = stream;

        startRecordingButton.addEventListener('click', () => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();

            mediaRecorder.ondataavailable = function(event) {
                chunks.push(event.data);
            };

            mediaRecorder.onstop = function() {
                const blob = new Blob(chunks, { 'type': 'video/mp4;' });
                chunks = [];
                // You can send this blob to the backend using fetch or save it locally
            };

            console.log('Recording started');
        });

        stopRecordingButton.addEventListener('click', () => {
            mediaRecorder.stop();
            console.log('Recording stopped');
        });
    })
    .catch(error => {
        console.error('Error accessing the webcam', error);
    });

// Handle form submission
const notesForm = document.getElementById('notes-form');
notesForm.addEventListener('submit', function(event) {
    event.preventDefault();

    const name = document.getElementById('name').value;
    const notes = document.getElementById('notes').value;

    // Send data to the backend using fetch
    fetch('/save-conversation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            name: name,
            notes: notes
            // Include conversation data if necessary
        })
    })
    .then(response => response.json())
    .then(data => {
        alert('Conversation saved successfully!');
        // Reset form
        notesForm.reset();
    })
    .catch(error => {
        console.error('Error:', error);
        alert('There was an error saving the conversation.');
    });
});