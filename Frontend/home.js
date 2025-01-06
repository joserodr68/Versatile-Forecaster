document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('csvFile');
    const uploadBtn = document.getElementById('uploadBtn');
    const submitBtn = document.getElementById('submitBtn');
    const fileName = document.getElementById('fileName');
    const uploadStatus = document.getElementById('uploadStatus');

    // Trigger file input when upload button is clicked
    uploadBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle file selection
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            // Check if file extension is .csv
            if (file.name.toLowerCase().endsWith('.csv')) {
                fileName.textContent = `Selected file: ${file.name}`;
                submitBtn.disabled = false;
                uploadStatus.textContent = '';
                uploadStatus.style.color = 'black';
            } else {
                fileName.textContent = 'Please select a CSV file';
                submitBtn.disabled = true;
                uploadStatus.textContent = 'Error: Only CSV files are allowed';
                uploadStatus.style.color = 'red';
                fileInput.value = ''; // Clear the file input
            }
        } else {
            fileName.textContent = 'No file chosen';
            submitBtn.disabled = true;
        }
    });

    // Handle file upload
    submitBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) {
            uploadStatus.textContent = 'Please select a file first';
            uploadStatus.style.color = 'red';
            return;
        }

        submitBtn.disabled = true; // Disable button during upload
        const formData = new FormData();
        formData.append('file', file);

        try {
            uploadStatus.textContent = 'Uploading...';
            uploadStatus.style.color = '#3498db'; // Match your theme blue color
            
            const response = await fetch('http://localhost:8000/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                uploadStatus.textContent = 'Upload successful! Redirecting to analysis...';
                uploadStatus.style.color = '#2ecc71'; // Match your theme green color
                
                // Store the filename in sessionStorage
                sessionStorage.setItem('uploadedFile', file.name);
                
                // Redirect to analysis page after a short delay
                setTimeout(() => {
                    window.location.href = 'analysis.html';
                }, 2000);
            } else {
                throw new Error(data.detail || 'Upload failed');
            }
        } catch (error) {
            uploadStatus.textContent = `Error: ${error.message || 'Upload failed. Please try again.'}`;
            uploadStatus.style.color = '#e74c3c'; // Match your theme red color
            submitBtn.disabled = false; // Re-enable button on error
            console.error('Upload error:', error);
        }
    });

    // Clear file selection when navigating back to the page
    window.addEventListener('pageshow', (event) => {
        if (event.persisted) {
            fileInput.value = '';
            fileName.textContent = 'No file chosen';
            submitBtn.disabled = true;
            uploadStatus.textContent = '';
        }
    });
});