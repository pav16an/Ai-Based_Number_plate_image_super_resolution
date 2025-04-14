const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('start-webcam');
const stopBtn = document.getElementById('stop-webcam');
const captureBtn = document.getElementById('capture');
const resultsDiv = document.getElementById('results');

let stream = null;
let detectionInterval = null;
let isProcessing = false;

// Set canvas size to match video
function updateCanvasSize() {
    const aspectRatio = video.videoWidth / video.videoHeight;
    const maxWidth = 1280;
    const maxHeight = 720;
    
    let width = video.videoWidth;
    let height = video.videoHeight;
    
    if (width > maxWidth) {
        width = maxWidth;
        height = width / aspectRatio;
    }
    if (height > maxHeight) {
        height = maxHeight;
        width = height * aspectRatio;
    }
    
    canvas.width = width;
    canvas.height = height;
    console.log("Canvas size updated:", canvas.width, canvas.height);
}

// Start webcam
startBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'environment'
            } 
        });
        video.srcObject = stream;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        captureBtn.disabled = false;
        
        video.onloadedmetadata = () => {
            console.log("Webcam resolution:", video.videoWidth, video.videoHeight);
            updateCanvasSize();
            // Start detection loop
            detectionInterval = setInterval(processFrame, 1000);
        };
    } catch (err) {
        console.error("Error accessing webcam:", err);
        resultsDiv.innerHTML = `<div class="error-message">
            <p>Error: ${err.message}</p>
            <p>Make sure you've allowed camera access and are using HTTPS/localhost</p>
        </div>`;
    }
});

// Stop webcam
stopBtn.addEventListener('click', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        captureBtn.disabled = true;
        clearInterval(detectionInterval);
        resultsDiv.innerHTML = "";
    }
});

// Process each frame
async function processFrame() {
    if (video.readyState === video.HAVE_ENOUGH_DATA && !isProcessing) {
        isProcessing = true;
        captureBtn.disabled = true;
        captureBtn.innerHTML = '<span class="icon">⏳</span> Processing...';
        
        try {
            const ctx = canvas.getContext('2d');
            if (!ctx) {
                throw new Error("Could not get canvas context");
            }
            
            // Clear canvas and draw video frame
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Get image data with consistent quality
            const imageData = canvas.toDataURL('image/jpeg', 0.95);
            
            console.log("Sending frame for processing...");
            
            const response = await fetch('/api/webcam', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log("Response data:", data);
            
            if (data.error) {
                resultsDiv.innerHTML = `<div class="error-message">Error: ${data.error}</div>`;
                return;
            }
            
            // Display results
            if (data.license_plates && data.license_plates.length > 0) {
                resultsDiv.innerHTML = data.license_plates
                    .map(plate => `<div class="detection-result">
                        <span class="plate-number">${plate.number}</span>
                        <span class="plate-type">${plate.type}</span>
                        <span class="plate-confidence">${(plate.confidence * 100).toFixed(1)}%</span>
                    </div>`)
                    .join('');
                
                // Draw bounding boxes
                if (data.boxes) {
                    data.boxes.forEach(box => {
                        const [x1, y1, x2, y2] = box;
                        ctx.strokeStyle = '#00FF00';
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                    });
                }
            } else {
                resultsDiv.innerHTML = '<div class="no-plates">No license plates detected</div>';
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            }
        } catch (error) {
            console.error("Error processing frame:", error);
            resultsDiv.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
        } finally {
            isProcessing = false;
            captureBtn.disabled = false;
            captureBtn.innerHTML = '<span class="icon">📸</span> Capture';
        }
    }
}

// Manual capture
captureBtn.addEventListener('click', processFrame);

// Handle window resize
window.addEventListener('resize', updateCanvasSize);
