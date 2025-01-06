document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded - Forecast Page');
    
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
    const modelSelect = document.getElementById('modelSelect');
    const periodsInput = document.getElementById('periodsInput');
    const generateButton = document.getElementById('generateForecast');
    const forecastPlot = document.getElementById('forecastPlot');

    // Function to save the plot automatically
    async function autoSavePlot() {
        try {
            console.log('Starting automatic plot save...');
            
            // Create downloads directory first
            await fetch('http://localhost:8000/create-directory', {
                method: 'POST'
            });

            // Get the plot as an image
            const plotElement = document.getElementById('forecastPlot');
            const plotImage = await Plotly.toImage(plotElement, {
                format: 'png',
                width: 1200,
                height: 800
            });

            // Convert base64 to blob
            const response = await fetch(plotImage);
            const blob = await response.blob();

            // Generate timestamp for unique filename
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const saveFilename = `forecast_${filename.split('.')[0]}_${timestamp}.png`;

            // Save the image using the API
            const saveResponse = await fetch('http://localhost:8000/save-file', {
                method: 'POST',
                headers: {
                    'Content-Type': 'image/png',
                    'X-File-Name': saveFilename
                },
                body: blob
            });

            if (!saveResponse.ok) {
                throw new Error('Failed to save plot');
            }

            console.log('Plot saved successfully as:', saveFilename);
        } catch (error) {
            console.error('Error saving plot:', error);
        }
    }

    // Function to create forecast plot
    function createForecastPlot(data) {
        console.log('Creating forecast plot with data:', JSON.stringify(data, null, 2));
        
        // Validate data before creating plot
        if (!data.val_dates || !data.val_values || 
            !data.validation_predictions || 
            !data.forecast_dates || !data.forecast_predictions) {
            console.error('Missing required data for plot');
            return;
        }

        // Ensure all arrays have the expected length
        console.log('Data lengths:', {
            val: data.val_dates.length,
            val_pred: data.validation_predictions.length,
            forecast: data.forecast_predictions.length
        });

        // Validation data (actual values)
        const valTrace = {
            x: data.val_dates,
            y: data.val_values,
            type: 'scatter',
            mode: 'lines',
            name: 'Actual Values',
            line: { color: '#2ecc71', width: 2.5 }
        };

        // Validation predictions
        const valPredTrace = {
            x: data.val_dates,
            y: data.validation_predictions,
            type: 'scatter',
            mode: 'lines',
            name: `${modelSelect.value.toUpperCase()} Predictions`,
            line: { 
                color: '#e74c3c',
                width: 2,
                dash: 'dash'
            }
        };

        // Future forecast
        const forecastTrace = {
            x: data.forecast_dates,
            y: data.forecast_predictions,
            type: 'scatter',
            mode: 'lines',
            name: `${modelSelect.value.toUpperCase()} Forecast`,
            line: { 
                color: '#9b59b6',
                width: 2,
                dash: 'dash'
            }
        };

        const layout = {
            title: {
                text: ` `,
                font: { size: 24 },
                pad: { t: 20 }
            },
            xaxis: { 
                title: 'Date',
                showgrid: true,
                gridcolor: 'rgba(0,0,0,0.1)',
                tickangle: 45,
                tickfont: { size: 12 }
            },
            yaxis: { 
                title: 'Value',
                showgrid: true,
                gridcolor: 'rgba(0,0,0,0.1)',
                tickfont: { size: 12 }
            },
            showlegend: true,
            legend: {
                x: 0.5,
                y: 1.15,
                xanchor: 'center',
                orientation: 'h',
                font: { size: 12 }
            },
            margin: { t: 100, l: 80, r: 40, b: 80 },
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            hovermode: 'x unified',
            height: 525  // Set to 75% of previous height
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        };

        try {
            Plotly.newPlot('forecastPlot', [valTrace, valPredTrace, forecastTrace], layout, config)
                .then(() => {
                    // Automatically save the plot after successful creation
                    autoSavePlot();
                })
                .catch(error => {
                    console.error('Error in plot creation or saving:', error);
                });
        } catch (error) {
            console.error('Error creating plot:', error);
        }
    }

    // Function to update metrics table
    function updateMetrics(metrics) {
        console.log('Updating metrics:', metrics);
        if (!metrics) {
            console.error('No metrics data provided');
            return;
        }
        document.getElementById('mae').textContent = metrics.mae.toFixed(4);
        document.getElementById('rmse').textContent = metrics.rmse.toFixed(4);
        document.getElementById('r2').textContent = metrics.r2.toFixed(4);
        document.getElementById('mape').textContent = metrics.mape.toFixed(4) + '%';
    }

    // Function to handle loading state
    function setLoadingState(isLoading) {
        generateButton.disabled = isLoading;
        generateButton.textContent = isLoading ? 'Generating...' : 'Generate Forecast';
        
        if (isLoading) {
            // Only clear metrics during loading
            document.getElementById('mae').textContent = '-';
            document.getElementById('rmse').textContent = '-';
            document.getElementById('r2').textContent = '-';
            document.getElementById('mape').textContent = '-';
        }
    }

    // Function to handle errors
    function handleError(error) {
        console.error('Error:', error);
        // Remove any existing error messages first
        const existingError = document.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }

        const errorMessage = document.createElement('div');
        errorMessage.className = 'error-message';
        errorMessage.textContent = `Error generating forecast: ${error.message}. Please try again.`;
        document.querySelector('.forecast-header').appendChild(errorMessage);
        
        // Clear loading state
        setLoadingState(false);
    }

    // Function to generate forecast
    async function generateForecast() {
        try {
            setLoadingState(true);
            
            const model_type = modelSelect.value;
            const forecast_periods = parseInt(periodsInput.value);
            
            console.log(`Generating forecast using ${model_type} for ${forecast_periods} periods`);

            // Remove any existing error messages
            const existingError = document.querySelector('.error-message');
            if (existingError) {
                existingError.remove();
            }

            // Make API request with query parameters
            const url = new URL(`http://localhost:8000/forecast/${filename}`);
            url.searchParams.append('model_type', model_type);
            url.searchParams.append('forecast_periods', forecast_periods);

            console.log('Sending request to:', url.toString());

            const response = await fetch(url, {
                method: 'POST'
            });

            if (!response.ok) {
                let errorMessage = `HTTP error! status: ${response.status}`;
                try {
                    const errorData = await response.text();
                    errorMessage += `, message: ${errorData}`;
                } catch (e) {
                    console.error('Error parsing error response:', e);
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();
            console.log('Forecast data received:', data);

            if (!data) {
                throw new Error('No data received from server');
            }

            // Create visualization
            createForecastPlot(data);
            
            // Update metrics
            if (data.metrics) {
                updateMetrics(data.metrics);
            } else {
                console.error('No metrics data received');
                throw new Error('No metrics data received from server');
            }

        } catch (error) {
            handleError(error);
        } finally {
            setLoadingState(false);
        }
    }

    // Event listener for generate button
    generateButton.addEventListener('click', generateForecast);

    // Event listeners for input validation
    periodsInput.addEventListener('input', () => {
        const value = parseInt(periodsInput.value);
        if (value < 1) periodsInput.value = 1;
        if (value > 365) periodsInput.value = 365;
    });

    // Initialize the page without any initial message
    setLoadingState(false);
});