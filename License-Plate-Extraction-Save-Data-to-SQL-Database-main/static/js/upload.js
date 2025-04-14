document.addEventListener('DOMContentLoaded', () => {
    const uploadBox = document.getElementById('upload-box');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const fileName = fileInfo.querySelector('.file-name');
    const removeFileBtn = fileInfo.querySelector('.remove-file');
    const processBtn = document.getElementById('process-btn');
    const resultsContainer = document.getElementById('results');
    let selectedFile = null;

    // Drag and drop functionality
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('dragover');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    // Click to upload
    uploadBox.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    // Remove file
    removeFileBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        fileInfo.style.display = 'none';
        resultsContainer.innerHTML = '';
    });

    // Process file
    processBtn.addEventListener('click', () => {
        if (selectedFile) {
            processFile(selectedFile);
        }
    });

    function handleFileSelection(file) {
        selectedFile = file;
        fileName.textContent = file.name;
        fileInfo.style.display = 'block';
        resultsContainer.innerHTML = '';
    }

    function processFile(file) {
        // Show loading state
        resultsContainer.innerHTML = `
            <div class="loading-spinner">
                <div class="spinner"></div>
                <p>Processing ${file.name}...</p>
            </div>
        `;

        const formData = new FormData();
        formData.append('file', file);

        fetch('/api/process', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            displayResults(data);
        })
        .catch(error => {
            resultsContainer.innerHTML = `
                <div class="alert alert-error">
                    <p>Error: ${error.message}</p>
                </div>
            `;
        });
    }

    function displayResults(data) {
        let html = '';

        if (data.license_plates && data.license_plates.length > 0) {
            html += `
                <div class="detection-results">
                    <h3>Detected License Plates</h3>
                    <div class="plates-grid">
                        ${data.license_plates.map(plate => `
                            <div class="plate-card">
                                <div class="plate-number">${plate.number}</div>
                                <div class="plate-details">
                                    <span class="plate-type">${plate.type}</span>
                                    <span class="plate-confidence">${(plate.confidence * 100).toFixed(1)}%</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        } else {
            html += `
                <div class="alert alert-info">
                    <p>No license plates detected in the image.</p>
                </div>
            `;
        }

        if (data.image) {
            html += `
                <div class="processed-image">
                    <h3>Processed Image</h3>
                    <img src="${data.image}" alt="Processed Image">
                </div>
            `;
        }

        resultsContainer.innerHTML = html;
    }
}); 