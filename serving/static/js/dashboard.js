// SOFARE-AI Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

function initializeDashboard() {
    // Navigation handling
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view');

    navItems.forEach(item => {
        item.addEventListener('click', function() {
            const viewName = this.getAttribute('data-view');

            // Update navigation
            navItems.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');

            // Update views
            views.forEach(view => view.classList.remove('active'));
            document.getElementById(`${viewName}-view`).classList.add('active');
        });
    });

    // Timeframe buttons
    const timeframeBtns = document.querySelectorAll('.timeframe-btn');
    timeframeBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const container = this.closest('.widget-header');
            container.querySelectorAll('.timeframe-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            // Here you would update the chart data
            showToast('Chart timeframe updated', 'info');
        });
    });

    // Initialize real-time updates
    startRealTimeUpdates();

    // Initialize charts (placeholder)
    initializeCharts();
    initializeAccuracyChart();

    // Show welcome message
    setTimeout(() => {
        showToast('SOFARE-AI Dashboard loaded successfully', 'success');
    }, 1000);
}

function startRealTimeUpdates() {
    // Update last update time
    setInterval(() => {
        document.getElementById('lastUpdate').textContent = 'Just now';
    }, 1000);

    // Simulate real-time data updates
    setInterval(() => {
        updateLiveData();
        updateChartData(); // Update chart with new data
    }, 5000);

    // Simulate signal updates
    setInterval(() => {
        updateSignals();
    }, 10000);
}

function updateLiveData() {
    // Fetch real macro data
    fetch('/api/macro')
        .then(response => response.json())
        .then(macroData => {
            if (!macroData.error) {
                updateMacroIndicators(macroData);
            }
        })
        .catch(error => {
            console.error('Failed to fetch macro data:', error);
        });

    // Fetch prediction from trained model
    fetch('/predict')
        .then(response => response.json())
        .then(predictionData => {
            if (predictionData.prediction !== undefined) {
                updatePredictionDisplay(predictionData);
            }
        })
        .catch(error => {
            console.error('Failed to fetch prediction:', error);
        });

    // Update performance metrics
    updatePerformanceMetrics();
}

function updateMacroIndicators(macroData) {
    // Helper to safely format numbers
    const safeFormat = (val, decimals = 2) => {
        if (val === null || val === undefined) return 'N/A';
        return val.toFixed(decimals);
    };

    const safeLocale = (val) => {
        if (val === null || val === undefined) return 'N/A';
        return val.toLocaleString();
    };

    // Update Fed Funds Rate
    const fedRateElement = document.querySelector('.macro-item:nth-child(1) .macro-value span:first-child');
    if (fedRateElement) {
        fedRateElement.textContent = safeFormat(macroData.fed_funds_rate, 1);
    }

    // Update Gold Price
    const goldElement = document.querySelector('.macro-item:nth-child(2) .macro-value span:first-child');
    if (goldElement) {
        goldElement.textContent = safeLocale(macroData.gold_price);
    }

    // Update DXY
    const dxyElement = document.querySelector('.macro-item:nth-child(3) .macro-value span:first-child');
    if (dxyElement) {
        dxyElement.textContent = safeFormat(macroData.dxy, 2);
    }

    // Update S&P 500
    const sp500Element = document.querySelector('.macro-item:nth-child(4) .macro-value span:first-child');
    if (sp500Element) {
        sp500Element.textContent = safeLocale(macroData.sp500);
    }

    // Update VIX
    const vixElement = document.querySelector('.macro-item:nth-child(5) .macro-value span:first-child');
    if (vixElement) {
        vixElement.textContent = safeFormat(macroData.vix, 1);
    }

    // Update Oil Price
    const oilElement = document.querySelector('.macro-item:nth-child(6) .macro-value span:first-child');
    if (oilElement) {
        oilElement.textContent = safeFormat(macroData.oil_price, 2);
    }
}

function updatePredictionDisplay(predictionData) {
    // Update prediction card
    const predictionElement = document.getElementById('prediction-value');
    const directionElement = document.getElementById('prediction-direction');
    const confidenceElement = document.getElementById('prediction-confidence');
    const modelVersionElement = document.getElementById('model-version');

    if (predictionElement) {
        const returnPercent = (predictionData.prediction * 100).toFixed(2);
        predictionElement.textContent = `${returnPercent}%`;
        
        // Color based on prediction
        if (predictionData.prediction > 0.005) {
            predictionElement.style.color = 'var(--success-color)';
        } else if (predictionData.prediction < -0.005) {
            predictionElement.style.color = 'var(--error-color)';
        } else {
            predictionElement.style.color = 'var(--warning-color)';
        }
    }

    if (directionElement) {
        if (predictionData.prediction > 0.005) {
            directionElement.textContent = 'UP';
            directionElement.style.color = 'var(--success-color)';
        } else if (predictionData.prediction < -0.005) {
            directionElement.textContent = 'DOWN';
            directionElement.style.color = 'var(--error-color)';
        } else {
            directionElement.textContent = 'HOLD';
            directionElement.style.color = 'var(--warning-color)';
        }
    }

    if (confidenceElement) {
        // Simulate confidence based on prediction magnitude
        const confidence = Math.min(Math.abs(predictionData.prediction) * 1000, 95).toFixed(0);
        confidenceElement.textContent = `${confidence}%`;
    }

    if (modelVersionElement) {
        modelVersionElement.textContent = predictionData.model_version || 'N/A';
    }

    // Update the main signal widget
    updateSignalWidget(predictionData);
}

function updateSignalWidget(predictionData) {
    const signalElement = document.getElementById('mainSignal');
    if (!signalElement) return;

    const signalIcon = signalElement.querySelector('.signal-icon i');
    const signalLabel = signalElement.querySelector('.signal-label');
    const signalConfidence = signalElement.querySelector('.signal-confidence');
    const predictedChange = signalElement.querySelector('.metric-value');

    if (predictionData.prediction > 0.005) {
        signalIcon.className = 'fas fa-arrow-up';
        signalElement.querySelector('.signal-icon').style.backgroundColor = 'var(--success-color)';
        signalLabel.textContent = 'BUY';
        signalLabel.style.color = 'var(--success-color)';
    } else if (predictionData.prediction < -0.005) {
        signalIcon.className = 'fas fa-arrow-down';
        signalElement.querySelector('.signal-icon').style.backgroundColor = 'var(--error-color)';
        signalLabel.textContent = 'SELL';
        signalLabel.style.color = 'var(--error-color)';
    } else {
        signalIcon.className = 'fas fa-minus';
        signalElement.querySelector('.signal-icon').style.backgroundColor = 'var(--warning-color)';
        signalLabel.textContent = 'HOLD';
        signalLabel.style.color = 'var(--warning-color)';
    }

    if (signalConfidence) {
        const confidence = Math.min(Math.abs(predictionData.prediction) * 1000, 95).toFixed(0);
        signalConfidence.textContent = `Confidence: ${confidence}%`;
    }

    if (predictedChange) {
        const returnPercent = (predictionData.prediction * 100).toFixed(2);
        predictedChange.textContent = `${returnPercent}%`;
        predictedChange.className = 'metric-value ' + (predictionData.prediction > 0.005 ? 'positive' : predictionData.prediction < -0.005 ? 'negative' : 'neutral');
    }
}

function updateSignals() {
    const signals = ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL'];
    const confidences = ['65%', '72%', '58%', '81%', '89%'];
    const icons = ['arrow-up', 'arrow-up', 'minus', 'arrow-down', 'arrow-down'];

    const randomIndex = Math.floor(Math.random() * signals.length);
    const signalElement = document.getElementById('mainSignal');
    const iconElement = signalElement.querySelector('.signal-icon i');
    const labelElement = signalElement.querySelector('.signal-label');
    const confidenceElement = signalElement.querySelector('.signal-confidence');

    iconElement.className = `fas fa-${icons[randomIndex]}`;
    labelElement.textContent = signals[randomIndex];
    confidenceElement.textContent = `Confidence: ${confidences[randomIndex]}`;

    // Update signal styling
    const signalContainer = signalElement.querySelector('.signal-icon');
    if (signals[randomIndex].includes('BUY')) {
        signalContainer.style.backgroundColor = 'var(--success-color)';
    } else if (signals[randomIndex].includes('SELL')) {
        signalContainer.style.backgroundColor = 'var(--error-color)';
    } else {
        signalContainer.style.backgroundColor = 'var(--warning-color)';
    }
}

function updatePerformanceMetrics() {
    const metrics = document.querySelectorAll('.metric-card');
    metrics.forEach(metric => {
        const valueElement = metric.querySelector('.metric-value');
        const changeElement = metric.querySelector('.metric-change');

        if (valueElement && changeElement) {
            // Small random changes
            const currentValue = parseFloat(valueElement.textContent);
            const change = (Math.random() - 0.5) * 0.01;
            const newValue = currentValue + change;

            valueElement.textContent = newValue.toFixed(4);

            if (change > 0) {
                changeElement.textContent = `↑ ${(change * 100).toFixed(1)}%`;
                changeElement.className = 'metric-change positive';
            } else {
                changeElement.textContent = `↓ ${Math.abs(change * 100).toFixed(1)}%`;
                changeElement.className = 'metric-change negative';
            }
        }
    });
}

async function fetchData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching data from ${url}:`, error);
        return null;
    }
}

async function initializeCharts() {
    console.log('Starting initializeCharts');
    try {
        const response = await fetchData('/api/ohlcv');
        console.log('Fetched response:', response);
            
            if (!response || response.error || !response.data || !Array.isArray(response.data) || response.data.length === 0) {
                console.warn('No OHLCV data available or invalid format');
                return;
            }

            const ohlcvData = response.data;
        console.log('OHLCV data length:', ohlcvData.length);

        const container = document.getElementById('mainChart');
        console.log('Container found:', container);
            
            // Safely destroy existing chart
            if (window.mainChart && typeof window.mainChart.destroy === 'function') {
                try {
                    window.mainChart.destroy();
                } catch (error) {
                    console.warn('Error destroying chart:', error);
                    window.mainChart = null;
                }
            } else if (window.mainChart) {
                console.warn('window.mainChart exists but destroy is not a function, resetting');
                window.mainChart = null;
            }

            // Verify ApexCharts is loaded
            if (typeof ApexCharts === 'undefined') {
                console.error('ApexCharts is not loaded');
                return;
            }

            let validData = [];
            try {
                console.log('Creating chart');
                // Format data for ApexCharts
                validData = ohlcvData.filter(item => item.timestamp && !isNaN(new Date(item.timestamp).getTime()) && typeof item.close === 'number');
                const formattedData = validData.map(item => ({
                    x: new Date(item.timestamp),
                    y: [item.open, item.high, item.low, item.close]
                }));

                // ApexCharts options
                const options = {
                    series: [{
                        name: 'candle',
                        type: 'candlestick',
                        data: formattedData
                    }],
                    chart: {
                        type: 'candlestick',
                        height: 400,
                        background: '#0a0a0a'
                    },
                    title: {
                        text: 'BTC/USDT Candlestick Chart',
                        align: 'left',
                        style: {
                            color: '#d1d4dc'
                        }
                    },
                    xaxis: {
                        type: 'datetime',
                        labels: {
                            style: {
                                colors: '#d1d4dc'
                            }
                        }
                    },
                    yaxis: {
                        labels: {
                            style: {
                                colors: '#d1d4dc'
                            }
                        }
                    },
                    plotOptions: {
                        candlestick: {
                            colors: {
                                upward: '#089981',
                                downward: '#f23645'
                            }
                        }
                    },
                    tooltip: {
                        theme: 'dark'
                    },
                    grid: {
                        borderColor: '#2a2e39'
                    }
                };

                // Create the chart
                window.mainChart = new ApexCharts(container, options);
                window.mainChart.render();
                console.log('Chart initialized successfully');

                // Make responsive
                const resizeChart = () => {
                    if (window.mainChart && container.offsetWidth > 0) {
                        window.mainChart.updateOptions({
                            chart: {
                                width: container.offsetWidth,
                                height: 400
                            }
                        });
                    }
                };
                window.addEventListener('resize', resizeChart);
                // Initial resize
                setTimeout(resizeChart, 100);

            } catch (error) {
                console.error('Error creating chart:', error);
            }

            // Update current price display
            const latestData = validData[validData.length - 1];
            if (latestData) {
                updateCurrentPrice(latestData.close);
            }

            // ApexCharts has built-in zoom

        } catch (error) {
            console.error('Error initializing charts:', error);
        }
    }

    function addZoomControls() {
        const chartContainer = document.querySelector('.chart-container');
        if (!chartContainer || document.querySelector('.zoom-controls')) return;

        // Create zoom control buttons
        const zoomControls = document.createElement('div');
        zoomControls.className = 'zoom-controls';
        zoomControls.innerHTML = `
            <button class="zoom-btn" id="zoomIn" title="Zoom In">
                <i class="fas fa-plus"></i>
            </button>
            <button class="zoom-btn" id="zoomOut" title="Zoom Out">
                <i class="fas fa-minus"></i>
            </button>
            <button class="zoom-btn" id="resetZoom" title="Reset Zoom">
                <i class="fas fa-undo"></i>
            </button>
            <button class="zoom-btn" id="panLeft" title="Pan Left">
                <i class="fas fa-chevron-left"></i>
            </button>
            <button class="zoom-btn" id="panRight" title="Pan Right">
                <i class="fas fa-chevron-right"></i>
            </button>
        `;

        chartContainer.appendChild(zoomControls);

        // Add event listeners
        document.getElementById('zoomIn').addEventListener('click', () => {
            if (window.mainChart && typeof window.mainChart.zoom === 'function') {
                window.mainChart.zoom(1.2);
            }
        });

        document.getElementById('zoomOut').addEventListener('click', () => {
            if (window.mainChart && typeof window.mainChart.zoom === 'function') {
                window.mainChart.zoom(0.8);
            }
        });

        document.getElementById('resetZoom').addEventListener('click', () => {
            if (window.mainChart && typeof window.mainChart.resetZoom === 'function') {
                window.mainChart.resetZoom();
            }
        });

        document.getElementById('panLeft').addEventListener('click', () => {
            if (window.mainChart && typeof window.mainChart.pan === 'function') {
                const xScale = window.mainChart.scales.x;
                const range = xScale.max - xScale.min;
                const panAmount = range * 0.1;
                window.mainChart.pan({x: -panAmount}, undefined, 'default');
            }
        });

        document.getElementById('panRight').addEventListener('click', () => {
            if (window.mainChart && typeof window.mainChart.pan === 'function') {
                const xScale = window.mainChart.scales.x;
                const range = xScale.max - xScale.min;
                const panAmount = range * 0.1;
                window.mainChart.pan({x: panAmount}, undefined, 'default');
            }
        });
    }

function updateCurrentPrice(price) {
    if (!price || isNaN(price)) return;
    
    // Update the price display in the header or chart area
    const priceDisplay = document.querySelector('.chart-legend');
    if (priceDisplay) {
        const currentPriceElement = priceDisplay.querySelector('.current-price');
        if (!currentPriceElement) {
            const priceDiv = document.createElement('div');
            priceDiv.className = 'legend-item current-price';
            priceDiv.innerHTML = `
                <div class="legend-color" style="background: linear-gradient(45deg, #00d4aa, #007aff);"></div>
                <span style="font-weight: bold; font-size: 1.1rem;">$${price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span>
            `;
            priceDisplay.appendChild(priceDiv);
        } else {
            currentPriceElement.querySelector('span').textContent = `$${price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        }
    }
}

function initializeAccuracyChart() {
    const ctx = document.getElementById('accuracyChart');
    if (!ctx) return;

    // Generate sample accuracy data over time
    const labels = [];
    const accuracyData = [];
    const baselineData = [];

    const now = new Date();
    for (let i = 29; i >= 0; i--) {
        const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000); // Last 30 days
        labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));

        // Generate realistic accuracy data with some variation
        const baseAccuracy = 75 + Math.sin(i * 0.2) * 5 + (Math.random() - 0.5) * 3;
        accuracyData.push(Math.max(65, Math.min(85, baseAccuracy)));

        // Baseline accuracy (constant 70%)
        baselineData.push(70);
    }

    window.accuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'SOFARE Accuracy',
                    data: accuracyData,
                    borderColor: 'rgba(0, 212, 170, 1)',
                    backgroundColor: 'rgba(0, 212, 170, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointBackgroundColor: 'rgba(0, 212, 170, 1)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                },
                {
                    label: 'Baseline',
                    data: baselineData,
                    borderColor: 'rgba(128, 128, 128, 1)',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        color: '#ccc'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 26, 26, 0.95)',
                    titleColor: '#fff',
                    bodyColor: '#ccc',
                    borderColor: 'rgba(0, 212, 170, 0.3)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        title: function(context) {
                            return context[0].label;
                        },
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(64, 64, 64, 0.3)',
                        borderColor: 'rgba(64, 64, 64, 0.5)'
                    },
                    ticks: {
                        color: '#ccc',
                        maxTicksLimit: 7
                    }
                },
                y: {
                    min: 60,
                    max: 90,
                    grid: {
                        color: 'rgba(64, 64, 64, 0.3)',
                        borderColor: 'rgba(64, 64, 64, 0.5)'
                    },
                    ticks: {
                        color: '#ccc',
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            elements: {
                point: {
                    hoverBorderWidth: 3
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        }
    });
}

function updateChartData() {
    if (!window.priceChart) return;

    // Fetch latest OHLCV data
    fetch('/api/ohlcv?limit=10')
        .then(response => response.json())
        .then(result => {
            if (result.error || !result.data || result.data.length === 0) return;

            const newData = result.data.map(item => ({
                x: new Date(item.timestamp),
                o: item.open,
                h: item.high,
                l: item.low,
                c: item.close,
                v: item.volume
            }));

            // Get the latest timestamp from current chart
            const currentData = window.priceChart.data.datasets[0].data;
            const lastTimestamp = currentData.length > 0 ? currentData[currentData.length - 1].x : 0;

            // Filter new data that's after the last timestamp
            const filteredNewData = newData.filter(item => item.x > lastTimestamp);

            if (filteredNewData.length > 0) {
                // Add new data points to both datasets
                window.priceChart.data.datasets[0].data.push(...filteredNewData);
                window.priceChart.data.datasets[1].data.push(...filteredNewData.map(d => ({ x: d.x, y: d.v })));

                // Keep only last 200 data points
                if (window.priceChart.data.datasets[0].data.length > 200) {
                    const excess = window.priceChart.data.datasets[0].data.length - 200;
                    window.priceChart.data.datasets[0].data.splice(0, excess);
                    window.priceChart.data.datasets[1].data.splice(0, excess);
                }

                // Update chart without animation for real-time feel
                window.priceChart.update('none');

                // Update current price
                const latestData = filteredNewData[filteredNewData.length - 1];
                if (latestData) {
                    updateCurrentPrice(latestData.c);
                }
            }
        })
        .catch(error => {
            console.error('Failed to update chart data:', error);
        });
}

function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toastContainer');

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const iconMap = {
        success: 'check-circle',
        warning: 'exclamation-triangle',
        error: 'times-circle',
        info: 'info-circle'
    };

    toast.innerHTML = `
        <i class="fas fa-${iconMap[type]}"></i>
        <div class="toast-message">${message}</div>
        <i class="fas fa-times toast-close"></i>
    `;

    toastContainer.appendChild(toast);

    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 5000);

    // Manual close
    toast.querySelector('.toast-close').addEventListener('click', function() {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    });
}

// Simulate API calls (in real implementation, these would call your FastAPI backend)
function fetchLatestPrediction() {
    // This would call /predict endpoint
    return {
        prediction: 87450,
        confidence: 0.87,
        signal: 'BUY'
    };
}

function fetchSystemHealth() {
    // This would call health endpoints
    return {
        ingestion: 'healthy',
        training: 'healthy',
        serving: 'healthy',
        latency: 45
    };
}

// Candlestick Prediction Chart
let predictionChart = null;

function initializePredictionChart() {
    // Register Chart.js Financial components
    if (typeof Chart !== 'undefined' && window['chartjs-chart-financial']) {
        const { CandlestickController, CandlestickElement } = window['chartjs-chart-financial'];
        Chart.register(CandlestickController, CandlestickElement);
    }

    const ctx = document.getElementById('predictionChart');
    if (!ctx) return;

    // Destroy existing chart
    if (predictionChart) {
        predictionChart.destroy();
    }

    predictionChart = new Chart(ctx, {
        type: 'candlestick',
        data: {
            datasets: [{
                label: 'Candlestick Data',
                data: [],
                backgroundColors: {
                    up: 'rgba(8, 152, 129, 0.8)',
                    down: 'rgba(242, 54, 69, 0.8)',
                    unchanged: 'rgba(128, 128, 128, 0.8)'
                },
                borderColors: {
                    up: 'rgb(8, 152, 129)',
                    down: 'rgb(242, 54, 69)',
                    unchanged: 'rgb(128, 128, 128)'
                },
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'timeseries',
                    time: {
                        unit: 'minute',
                        displayFormats: {
                            minute: 'HH:mm',
                            hour: 'MMM dd HH:mm'
                        }
                    },
                    ticks: {
                        maxRotation: 0,
                        autoSkip: true,
                        autoSkipPadding: 50
                    }
                },
                y: {
                    type: 'linear',
                    position: 'right',
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    intersect: false,
                    mode: 'index',
                    callbacks: {
                        title: function(context) {
                            const date = new Date(context[0].parsed.x);
                            return date.toLocaleString();
                        },
                        label: function(context) {
                            const point = context.parsed;
                            const type = point.prediction_type || 'historical';
                            return [
                                `Type: ${type}`,
                                `Open: $${point.o.toFixed(2)}`,
                                `High: $${point.h.toFixed(2)}`,
                                `Low: $${point.l.toFixed(2)}`,
                                `Close: $${point.c.toFixed(2)}`,
                                `Volume: ${point.v.toLocaleString()}`
                            ];
                        }
                    }
                },
                legend: {
                    display: false
                }
            }
        }
    });
}

async function generateCandlestickPredictions() {
    const forecastSteps = parseInt(document.getElementById('forecast-steps').value);

    try {
        showToast(`Generating ${forecastSteps} candlestick predictions...`, 'info');

        const response = await fetch(`/predict/candlestick?forecast_steps=${forecastSteps}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Prediction failed');
        }

        // Update status panel
        document.getElementById('pred-model-version').textContent = result.model_version;
        document.getElementById('pred-drift-status').textContent = result.drift_detected ? 'Detected' : 'Normal';
        document.getElementById('pred-last-update').textContent = new Date().toLocaleTimeString();
        document.getElementById('pred-steps').textContent = forecastSteps;

        // Prepare data for chart
        const historicalData = result.current_data.map(item => ({
            x: new Date(item.timestamp).getTime(),
            o: item.open,
            h: item.high,
            l: item.low,
            c: item.close,
            v: item.volume,
            prediction_type: 'historical'
        }));

        const predictedCandles = result.predicted_candles.map(item => ({
            x: new Date(item.timestamp).getTime(),
            o: item.open,
            h: item.high,
            l: item.low,
            c: item.close,
            v: item.volume,
            prediction_type: 'forecasted'
        }));

        // Update chart
        predictionChart.data.datasets[0].data = [...historicalData, ...predictedCandles];
        predictionChart.update();

        // Update next candle details
        if (predictedCandles.length > 0) {
            const next = predictedCandles[0];
            document.getElementById('next-open').textContent = `$${next.o.toFixed(2)}`;
            document.getElementById('next-high').textContent = `$${next.h.toFixed(2)}`;
            document.getElementById('next-low').textContent = `$${next.l.toFixed(2)}`;
            document.getElementById('next-close').textContent = `$${next.c.toFixed(2)}`;
            document.getElementById('next-volume').textContent = next.v.toLocaleString();
        }

        showToast(`${forecastSteps} candlestick predictions generated successfully`, 'success');

    } catch (error) {
        showToast(`Error generating predictions: ${error.message}`, 'error');
        console.error('Error generating predictions:', error);
    }
}

async function refreshPredictionData() {
    try {
        showToast('Refreshing prediction data...', 'info');
        await generateCandlestickPredictions();
    } catch (error) {
        showToast('Error refreshing data', 'error');
    }
}

// Initialize prediction chart when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // ... existing code ...

    // Initialize prediction chart
    setTimeout(() => {
        initializePredictionChart();
    }, 100);

    // Add event listeners for prediction controls
    const generateBtn = document.getElementById('generate-predictions');
    const refreshBtn = document.getElementById('refresh-predictions');

    if (generateBtn) {
        generateBtn.addEventListener('click', generateCandlestickPredictions);
    }

    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshPredictionData);
    }
});

function fetchDriftMetrics() {
    // This would call drift monitoring endpoints
    return {
        volume: { psi: 0.023, status: 'safe' },
        gold: { psi: 0.089, status: 'warning' },
        rsi: { psi: 0.015, status: 'safe' }
    };
}

// Export functions for potential API integration
window.SofareDashboard = {
    showToast,
    updateLiveData,
    fetchLatestPrediction,
    fetchSystemHealth,
    fetchDriftMetrics,
    generateCandlestickPredictions,
    initializePredictionChart
};