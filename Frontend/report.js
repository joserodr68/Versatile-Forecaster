document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded - Report Page');
    
    // Get the filename from sessionStorage
    const filename = sessionStorage.getItem('uploadedFile');
    console.log('Filename from session:', filename);
    
    if (!filename) {
        console.log('No filename found, redirecting to home');
        window.location.href = 'home.html';
        return;
    }

    // Update current file display
    document.getElementById('currentFile').textContent = `Current file: ${filename}`;

    // Get DOM elements
    const generateButton = document.getElementById('generateReport');
    const saveButton = document.getElementById('saveReport');
    const reportContent = document.getElementById('reportContent');
    const forecastPlot = document.getElementById('forecastPlot');

    // Load the forecast plot
    forecastPlot.src = './downloads/forecast.png';
    forecastPlot.onerror = () => {
        console.error('Error loading forecast plot');
        forecastPlot.alt = 'Error loading forecast plot';
    };

    // Function to handle loading state
    function setLoadingState(isLoading) {
        generateButton.disabled = isLoading;
        generateButton.textContent = isLoading ? 'Generating...' : 'Generate Report';
        if (isLoading) {
            reportContent.classList.add('loading');
        } else {
            reportContent.classList.remove('loading');
        }
    }

    // Function to handle errors
    function handleError(error) {
        console.error('Error:', error);
        reportContent.innerHTML = `
            <div class="error-message">
                Error generating report: ${error.message}. Please try again.
            </div>
        `;
        saveButton.disabled = true;
    }

    // Function to save report
    async function saveReport() {
        try {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const saveFilename = `report_${filename.split('.')[0]}_${timestamp}.md`;
            
            // Get the markdown content
            const markdownContent = reportContent.getAttribute('data-markdown');
            
            // Create a blob with the markdown content
            const blob = new Blob([markdownContent], { type: 'text/markdown' });
            
            // Save the file using the API
            const response = await fetch('http://localhost:8000/save-file', {
                method: 'POST',
                headers: {
                    'Content-Type': 'text/markdown',
                    'X-File-Name': saveFilename
                },
                body: blob
            });

            if (!response.ok) {
                throw new Error('Failed to save report');
            }

            console.log('Report saved successfully as:', saveFilename);
            alert('Report saved successfully!');
        } catch (error) {
            console.error('Error saving report:', error);
            alert('Error saving report: ' + error.message);
        }
    }

    // Function to generate report
    async function generateReport() {
        try {
            setLoadingState(true);
            
            // Make API request
            const response = await fetch(`http://localhost:8000/generate-report/${filename}`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (!data.report) {
                throw new Error('No report data received from server');
            }

            // Store the original markdown
            reportContent.setAttribute('data-markdown', data.report);
            
            // Convert markdown to HTML and display
            reportContent.innerHTML = marked.parse(data.report);
            
            // Enable save button
            saveButton.disabled = false;

        } catch (error) {
            handleError(error);
        } finally {
            setLoadingState(false);
        }
    }

    // Event listeners
    generateButton.addEventListener('click', generateReport);
    saveButton.addEventListener('click', saveReport);

    // Initialize the page
    setLoadingState(false);
});