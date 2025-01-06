document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded');
    
    // Get the filename from sessionStorage
    const filename = sessionStorage.getItem('uploadedFile');
    console.log('Filename from session:', filename);
    
    // Update current file display immediately
    const currentFileElement = document.getElementById('currentFile');
    
    if (!filename) {
        console.log('No filename found, redirecting to home');
        currentFileElement.textContent = 'No file selected. Redirecting to home...';
        currentFileElement.style.color = '#e74c3c';
        setTimeout(() => {
            window.location.href = 'home.html';
        }, 2000);
        return;
    }

    // Update current file display
    currentFileElement.textContent = `Current file: ${filename}`;
    currentFileElement.style.color = '#2c3e50';

    // Show loading state for all plots
    const plotContainers = document.querySelectorAll('.plot-container');
    plotContainers.forEach(container => {
        container.classList.add('loading');
    });

    // Function to create a time series plot
    function createTimeSeriesPlot(timestamps, values) {
        console.log('Creating time series plot');
        try {
            const trace = {
                x: timestamps,
                y: values,
                type: 'scatter',
                mode: 'lines',
                name: 'Time Series',
                line: { color: '#3498db' }
            };

            const layout = {
                title: ' ',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Value' },
                showlegend: false,
                margin: { t: 30, l: 60, r: 30, b: 60 },
                width: 750,  // Fixed width
                height: 550, // Fixed height
                autosize: false
            };

            const config = { 
                responsive: false,
                displayModeBar: true,
                displaylogo: false,
                staticPlot: false
            };

            Plotly.newPlot('timeSeriesPlot', [trace], layout, config)
                .then(() => {
                    const element = document.getElementById('timeSeriesPlot');
                    element.classList.remove('loading');
                });
        } catch (error) {
            console.error('Error creating time series plot:', error);
            const element = document.getElementById('timeSeriesPlot');
            element.classList.remove('loading');
            element.innerHTML = 'Error creating plot';
        }
    }

    // Function to create ACF/PACF plots
    function createCorrelationPlot(lags, values, confInt, plotId, title) {
        console.log(`Creating correlation plot: ${title}`);
        try {
            const trace = {
                x: Array.from({ length: values.length }, (_, i) => i),
                y: values,
                type: 'bar',
                name: title,
                marker: { color: '#3498db' }
            };

            const upperBound = Array(values.length).fill(confInt);
            const lowerBound = Array(values.length).fill(-confInt);

            const upperTrace = {
                x: Array.from({ length: values.length }, (_, i) => i),
                y: upperBound,
                type: 'scatter',
                mode: 'lines',
                name: '95% Confidence Interval',
                line: { color: 'red', dash: 'dash' }
            };

            const lowerTrace = {
                x: Array.from({ length: values.length }, (_, i) => i),
                y: lowerBound,
                type: 'scatter',
                mode: 'lines',
                showlegend: false,
                line: { color: 'red', dash: 'dash' }
            };

            const layout = {
                title: title,
                xaxis: { title: 'Lag' },
                yaxis: { title: 'Correlation' },
                showlegend: false,
                margin: { t: 30, l: 60, r: 30, b: 60 },
                width: 750,  // Fixed width
                height: 550, // Fixed height
                autosize: false
            };

            const config = { 
                responsive: false,
                displayModeBar: true,
                displaylogo: false,
                staticPlot: false
            };

            Plotly.newPlot(plotId, [trace, upperTrace, lowerTrace], layout, config)
                .then(() => {
                    const element = document.getElementById(plotId);
                    element.classList.remove('loading');
                });
        } catch (error) {
            console.error(`Error creating ${title} plot:`, error);
            const element = document.getElementById(plotId);
            element.classList.remove('loading');
            element.innerHTML = 'Error creating plot';
        }
    }

    // Function to create residuals plot
    function createResidualPlot(actual, predicted) {
        console.log('Creating residuals plot');
        try {
            const residuals = actual.map((value, index) => value - predicted[index]);
            
            const trace = {
                y: residuals,
                type: 'scatter',
                mode: 'markers',
                marker: {
                    color: '#3498db',
                    size: 5
                },
                name: 'Residuals'
            };

            const layout = {
                title: ' ',
                yaxis: { title: 'Residual Value' },
                xaxis: { title: 'Observation' },
                showlegend: false,
                margin: { t: 30, l: 60, r: 30, b: 60 },
                width: 750,  // Fixed width
                height: 550, // Fixed height
                autosize: false
            };

            const config = { 
                responsive: false,
                displayModeBar: true,
                displaylogo: false,
                staticPlot: false
            };

            Plotly.newPlot('arimaResiduals', [trace], layout, config)
                .then(() => {
                    const element = document.getElementById('arimaResiduals');
                    element.classList.remove('loading');
                });
        } catch (error) {
            console.error('Error creating residuals plot:', error);
            const element = document.getElementById('arimaResiduals');
            element.classList.remove('loading');
            element.innerHTML = 'Error creating plot';
        }
    }

    // Function to create GARCH volatility plot
    function createGarchPlot(timestamps, volatility) {
        console.log('Creating GARCH plot');
        try {
            const trace = {
                x: timestamps,
                y: volatility,
                type: 'scatter',
                mode: 'lines',
                name: 'Conditional Volatility',
                line: { color: '#e74c3c' }
            };

            const layout = {
                title: ' ',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Volatility' },
                showlegend: false,
                margin: { t: 30, l: 60, r: 30, b: 60 },
                width: 750,  // Fixed width
                height: 550, // Fixed height
                autosize: false
            };

            const config = { 
                responsive: false,
                displayModeBar: true,
                displaylogo: false,
                staticPlot: false
            };

            Plotly.newPlot('garchPlot', [trace], layout, config)
                .then(() => {
                    const element = document.getElementById('garchPlot');
                    element.classList.remove('loading');
                });
        } catch (error) {
            console.error('Error creating GARCH plot:', error);
            const element = document.getElementById('garchPlot');
            element.classList.remove('loading');
            element.innerHTML = 'Error creating plot';
        }
    }

    // Function to update statistical summary
    function updateStatsSummary(stats) {
        console.log('Updating statistics summary');
        try {
            document.getElementById('mean').textContent = Number(stats.mean).toFixed(4);
            document.getElementById('std').textContent = Number(stats.std).toFixed(4);
            document.getElementById('min').textContent = Number(stats.min).toFixed(4);
            document.getElementById('max').textContent = Number(stats.max).toFixed(4);
            document.getElementById('length').textContent = stats.length;

            if (stats.stationarity_test) {
                const test = stats.stationarity_test;
                document.getElementById('adf-stat').textContent = Number(test.test_statistic).toFixed(4);
                document.getElementById('adf-pvalue').textContent = Number(test.p_value).toFixed(4);
                
                const isStationary = String(test.is_stationary) === 'true';
                const stationaryElement = document.getElementById('is-stationary');
                stationaryElement.textContent = isStationary ? 'Yes' : 'No';
                stationaryElement.style.color = isStationary ? '#27ae60' : '#e74c3c';

                if (test.critical_values) {
                    let criticalValuesHtml = '<tr><td colspan="2"><strong>Critical Values:</strong></td></tr>';
                    Object.entries(test.critical_values).forEach(([key, value]) => {
                        criticalValuesHtml += `<tr><td>${key}:</td><td>${Number(value).toFixed(4)}</td></tr>`;
                    });
                    document.getElementById('critical-values').innerHTML = criticalValuesHtml;
                }
            }
        } catch (error) {
            console.error('Error updating statistics:', error);
            handleError(error);
        }
    }

    // Function to display ARIMA parameters
    function displayArimaParams(arima) {
        console.log('Displaying ARIMA parameters');
        try {
            let html = '<div class="model-params">';
            
            if (arima.order) {
                html += `<p><strong>ARIMA Order (p,d,q):</strong> (${arima.order.join(',')})</p>`;
            }
            if (arima.aic) {
                html += `<p><strong>AIC:</strong> ${Number(arima.aic).toFixed(4)}</p>`;
            }
            if (arima.info) {
                // Format the model information for better readability
                const formattedInfo = arima.info.split('\n').join('<br>');
                html += `<p><strong>Model Information:</strong><br>${formattedInfo}</p>`;
            }
            html += '</div>';
            
            document.getElementById('arimaParams').innerHTML = html;
        } catch (error) {
            console.error('Error displaying ARIMA parameters:', error);
            document.getElementById('arimaParams').innerHTML = 'Error displaying ARIMA parameters';
        }
    }

    // Function to handle errors
    function handleError(error) {
        console.error('Error:', error);
        const errorMessage = document.createElement('div');
        errorMessage.className = 'error-message';
        errorMessage.textContent = `Error loading analysis: ${error.message}. Please try again.`;
        document.querySelector('.analysis-header').appendChild(errorMessage);
        
        plotContainers.forEach(container => {
            container.classList.remove('loading');
            container.innerHTML = 'Error loading plot';
        });
    }

    // Function to load and display all analysis
    async function loadAnalysis() {
        try {
            console.log('Starting analysis for file:', filename);
            const response = await fetch(`http://localhost:8000/analyze/${filename}`);
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            const data = await response.json();
            console.log('Analysis data received:', data);

            if (!data) {
                throw new Error('No data received from server');
            }

            // Create all plots
            if (data.timestamps && data.values) {
                await createTimeSeriesPlot(data.timestamps, data.values);
            }
            
            if (data.acf_pacf) {
                await createCorrelationPlot(
                    data.acf_pacf.acf.length,
                    data.acf_pacf.acf,
                    data.acf_pacf.conf_int,
                    'acfPlot',
                    ' '
                );
                await createCorrelationPlot(
                    data.acf_pacf.pacf.length,
                    data.acf_pacf.pacf,
                    data.acf_pacf.conf_int,
                    'pacfPlot',
                    ' '
                );
            }

            if (data.arima && data.values) {
                await createResidualPlot(data.values, data.arima.predictions);
                displayArimaParams(data.arima);
            }

            if (data.garch && data.garch.volatility && data.timestamps) {
                await createGarchPlot(data.timestamps, data.garch.volatility);
            }

            // Update summary statistics
            if (data.statistics) {
                updateStatsSummary(data.statistics);
            }

        } catch (error) {
            handleError(error);
        }
    }

    // Load the analysis
    console.log('Initiating analysis load...');
    loadAnalysis();
});