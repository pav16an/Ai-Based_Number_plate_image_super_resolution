// DOM Elements
const uploadForm = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const resultContainer = document.getElementById('result-container');
const resultsDiv = document.getElementById('results');
const processedImage = document.getElementById('processed-image');
const processingDiv = document.getElementById('processing');
const videoControls = document.getElementById('video-controls');
const stopVideoBtn = document.getElementById('stop-video');
const videoProgress = document.getElementById('video-progress');
const errorMessage = document.getElementById('error-message');
const successMessage = document.getElementById('success-message');

// API Endpoints
const API_ENDPOINTS = {
    PROCESS_FILE: '/api/process',
    WEBCAM: '/api/webcam',
    PLATES: '/api/plates'
};

// Global variables
let isVideoProcessing = false;
let videoProcessingInterval = null;
let detectedPlates = new Set();

// Utility Functions
function showProcessing() {
    processingDiv.classList.add('active');
    resultContainer.classList.remove('hidden');
    errorMessage.style.display = 'none';
    successMessage.style.display = 'none';
}

function hideProcessing() {
    processingDiv.classList.remove('active');
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    successMessage.style.display = 'none';
}

function showSuccess(message) {
    successMessage.textContent = message;
    successMessage.style.display = 'block';
    errorMessage.style.display = 'none';
}

function displayResults(data) {
    resultsDiv.innerHTML = '';
    
    if (data.license_plates && data.license_plates.length > 0) {
        data.license_plates.forEach(plate => {
            // Add to detected plates set
            detectedPlates.add(plate.number);
            
            const plateDiv = document.createElement('div');
            plateDiv.className = 'plate-result';
            plateDiv.innerHTML = `
                <span class="plate-number">License Plate: ${plate.number}</span>
                <span class="plate-type">Type: ${plate.type || 'Unknown'}</span>
                <span class="plate-confidence">Confidence: ${(plate.confidence * 100).toFixed(1)}%</span>
            `;
            resultsDiv.appendChild(plateDiv);
        });
        
        showSuccess(`Detected ${data.license_plates.length} license plate(s)`);
    } else {
        resultsDiv.innerHTML = '<p>No license plates detected.</p>';
        showError('No license plates detected in the image');
    }

    if (data.image) {
        processedImage.src = data.image;
        processedImage.style.display = 'block';
    }
}

// File Upload Handler
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const file = fileInput.files[0];
    if (!file) {
        showError('Please select a file');
        return;
    }

    // Validate file type
    if (!file.type.startsWith('image/') && !file.type.startsWith('video/')) {
        showError('Please upload an image or video file');
        return;
    }

    // Validate file size (16MB limit)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        showProcessing();
        
        // Check if file is video
        const isVideo = file.type.startsWith('video/');
        
        if (isVideo) {
            // Show video controls
            videoControls.classList.remove('hidden');
            isVideoProcessing = true;
            
            // Start video processing
            const response = await fetch(API_ENDPOINTS.PROCESS_FILE, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to process video');
            }

            const data = await response.json();
            
            // Start periodic updates for video processing
            videoProcessingInterval = setInterval(async () => {
                try {
                    const updateResponse = await fetch(API_ENDPOINTS.PLATES);
                    if (updateResponse.ok) {
                        const updateData = await updateResponse.json();
                        displayResults(updateData);
                        
                        // Update progress bar
                        if (updateData.progress) {
                            videoProgress.value = updateData.progress;
                        }
                    }
                } catch (error) {
                    console.error('Error updating video results:', error);
                }
            }, 1000);
            
            showSuccess('Video processing started');
        } else {
            // Process image
            const response = await fetch(API_ENDPOINTS.PROCESS_FILE, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to process file');
            }

            const data = await response.json();
            displayResults(data);
        }
    } catch (error) {
        console.error('Error processing file:', error);
        showError(error.message || 'An error occurred while processing the file');
    } finally {
        hideProcessing();
    }
});

// Stop video processing
stopVideoBtn.addEventListener('click', async () => {
    if (videoProcessingInterval) {
        clearInterval(videoProcessingInterval);
        videoProcessingInterval = null;
    }
    
    isVideoProcessing = false;
    videoControls.classList.add('hidden');
    
    try {
        // Send stop signal to server
        const response = await fetch('/api/stop-video', {
            method: 'POST'
        });
        
        if (response.ok) {
            showSuccess('Video processing stopped');
            
            // Get final results
            const finalResponse = await fetch(API_ENDPOINTS.PLATES);
            if (finalResponse.ok) {
                const finalData = await finalResponse.json();
                displayResults(finalData);
            }
        }
    } catch (error) {
        console.error('Error stopping video processing:', error);
        showError('Error stopping video processing');
    }
});

// Drag and drop functionality
const fileLabel = document.querySelector('.file-label');

fileLabel.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileLabel.classList.add('dragover');
});

fileLabel.addEventListener('dragleave', () => {
    fileLabel.classList.remove('dragover');
});

fileLabel.addEventListener('drop', (e) => {
    e.preventDefault();
    fileLabel.classList.remove('dragover');
    
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        // Update the file label text
        const fileName = e.dataTransfer.files[0].name;
        document.querySelector('.file-text').textContent = fileName;
    }
});

// Update file label when file is selected
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
        const fileName = fileInput.files[0].name;
        document.querySelector('.file-text').textContent = fileName;
    } else {
        document.querySelector('.file-text').textContent = 'Choose a file or drag it here';
    }
});

// Webcam Handler
let webcamStream = null;
let webcamInterval = null;

async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "environment" // Use back camera if available
            } 
        });
        const videoElement = document.getElementById('webcam-video');
        videoElement.srcObject = stream;
        webcamStream = stream;

        // Wait for video to be ready
        videoElement.onloadedmetadata = () => {
            videoElement.play();
            console.log("Webcam started successfully");
            
            // Start sending frames to server
            webcamInterval = setInterval(sendWebcamFrame, 1000); // Send frame every second
        };
    } catch (error) {
        console.error('Failed to access webcam:', error);
        showError('Failed to access webcam: ' + error.message);
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    if (webcamInterval) {
        clearInterval(webcamInterval);
        webcamInterval = null;
    }
    console.log("Webcam stopped");
}

async function sendWebcamFrame() {
    const videoElement = document.getElementById('webcam-video');
    if (!videoElement || !videoElement.videoWidth) {
        console.error("Video element not ready");
        return;
    }
    
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0);
    
    // Reduce quality to improve performance
    const imageData = canvas.toDataURL('image/jpeg', 0.7);
    console.log("Sending webcam frame to server");

    try {
        const response = await fetch(API_ENDPOINTS.WEBCAM, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });

        if (!response.ok) {
            throw new Error('Failed to process webcam frame');
        }

        const data = await response.json();
        console.log("Received response from server:", data);
        
        if (data.error) {
            console.error("Server error:", data.error);
            return;
        }
        
        displayResults(data);
    } catch (error) {
        console.error('Webcam processing error:', error);
    }
}

// Database Operations
async function loadPlates() {
    try {
        const response = await fetch(API_ENDPOINTS.PLATES);
        if (!response.ok) {
            throw new Error('Failed to load plates');
        }

        const plates = await response.json();
        displayPlates(plates);
    } catch (error) {
        showError('Failed to load plates: ' + error.message);
    }
}

function displayPlates(plates) {
    const tableBody = document.getElementById('plates-table-body');
    if (!tableBody) return;

    tableBody.innerHTML = '';
    plates.forEach(plate => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${plate.plate_number}</td>
            <td>${plate.plate_type}</td>
            <td>${plate.confidence}</td>
            <td>${new Date(plate.timestamp).toLocaleString()}</td>
            <td>
                <button onclick="deletePlate(${plate.id})" class="delete-btn">Delete</button>
            </td>
        `;
        tableBody.appendChild(row);
    });
}

async function deletePlate(id) {
    if (!confirm('Are you sure you want to delete this plate?')) return;

    try {
        const response = await fetch(`${API_ENDPOINTS.PLATES}/${id}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error('Failed to delete plate');
        }

        showSuccess('Plate deleted successfully');
        loadPlates(); // Reload the table
    } catch (error) {
        showError('Failed to delete plate: ' + error.message);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Load plates if on database page
    if (document.getElementById('plates-table-body')) {
        loadPlates();
    }

    // Initialize webcam buttons if on webcam page
    if (document.getElementById('webcam-video')) {
        const startButton = document.getElementById('start-webcam');
        const stopButton = document.getElementById('stop-webcam');
        
        if (startButton && stopButton) {
            startButton.addEventListener('click', () => {
                startWebcam();
                startButton.disabled = true;
                stopButton.disabled = false;
            });
            
            stopButton.addEventListener('click', () => {
                stopWebcam();
                startButton.disabled = false;
                stopButton.disabled = true;
            });
        }
    }
});
