const video = document.getElementById('video');
const emotionElement = document.getElementById('emotion');

// Get access to the webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        video.play();
    });

// Capture a frame from the webcam every 1000ms
setInterval(() => {
    // Create a canvas element and draw the current frame from the video element onto it
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get the image data from the canvas
    const frame = canvas.toDataURL('image/jpeg');

    // Send the frame to the Django view function for ResNet prediction
    fetch('/predict_emotion/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrftoken
        },
        body: JSON.stringify({ frame: frame, method: 'ResNet' })
    })
    .then(response => response.json())
    .then(data => {
        // Update the emotion element with the predicted emotion from ResNet
        emotionElement.textContent = 'ResNet: ' + data.emotion;
    });

    // Send the frame to the Django view function for DeepFace prediction
    fetch('/predict_emotion/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrftoken
        },
        body: JSON.stringify({ frame: frame, method: 'DeepFace' })
    })
    .then(response => response.json())
    .then(data => {
        // Append the predicted emotion from DeepFace to the emotion element
        emotionElement.textContent += ', DeepFace: ' + data.emotion;
    });
}, 1000);
