const startRecordingButton = document.getElementById('start-recording');
const stopRecordingButton = document.getElementById('stop-recording');
const webcam = document.getElementById('webcam');

let mediaRecorder;
let recordedChunks = [];

navigator.mediaDevices.getUserMedia({ video: true, audio: true })
    .then(stream => {
        webcam.srcObject = stream;

        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = function (event) {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = function () {
            const blob = new Blob(recordedChunks, { type: 'video/webm' });
            recordedChunks = [];
            
            // Send the recorded video to the backend
            const formData = new FormData();
            formData.append('video', blob);

            fetch('/upload', {
                method: 'POST',
                body: formData
            }).then(response => response.json())
              .then(data => console.log('Upload success:', data))
              .catch(error => console.error('Upload error:', error));
        };

        startRecordingButton.addEventListener('click', () => {
            mediaRecorder.start();
            startRecordingButton.disabled = true;
            stopRecordingButton.disabled = false;
        });

        stopRecordingButton.addEventListener('click', () => {
            mediaRecorder.stop();
            startRecordingButton.disabled = false;
            stopRecordingButton.disabled = true;
        });
    })
    .catch(error => console.error('Webcam error:', error));
