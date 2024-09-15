const startButton = document.getElementById('start-btn');
const stopButton = document.getElementById('stop-btn');
const video = document.getElementById('webcam');
const status = document.getElementById('status');
let mediaRecorder;
let chunks = [];

async function initWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (e) => chunks.push(e.data);

        mediaRecorder.onstop = () => {
            const blob = new Blob(chunks, { type: 'video/webm' });
            chunks = [];
            // Save or send the blob to the server, e.g., via Fetch or WebSocket
            console.log('Video recorded');
        };
    } catch (error) {
        console.error('Error accessing webcam', error);
        status.textContent = 'Error accessing webcam';
    }
}

startButton.addEventListener('click', () => {
    if (mediaRecorder) {
        mediaRecorder.start();
        status.textContent = 'Recording...';
        startButton.disabled = true;
        stopButton.disabled = false;
    }
});

stopButton.addEventListener('click', () => {
    if (mediaRecorder) {
        mediaRecorder.stop();
        status.textContent = 'Recording stopped.';
        startButton.disabled = false;
        stopButton.disabled = true;
    }
});

window.onload = initWebcam;
