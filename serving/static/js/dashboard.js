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

    // Timeframe buttons - Dynamic Loading Implementation
    const timeframeBtns = document.querySelectorAll('.timeframe-btn');
    timeframeBtns.forEach(btn => {
        btn.addEventListener('click', async function() {
            const container = this.closest('.widget-header');
            container.querySelectorAll('.timeframe-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            const timeframe = this.getAttribute('data-tf');
            currentTimeframe = timeframe; // Update current timeframe
            await loadTimeframeData(timeframe);
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



// Helper function to update chart with timeframe data
function updateChartWithTimeframeData(ohlcvData, interval) {
    // Format data for ApexCharts
    const validData = ohlcvData.filter(item => 
        item.timestamp && typeof item.close === 'number'
    );
    
    const candleData = validData.map(item => ({
        x: new Date(item.timestamp).getTime(),
        y: [item.open, item.high, item.low, item.close]
    }));
    
    const volumeData = validData.map(item => ({
        x: new Date(item.timestamp).getTime(),
        y: item.volume,
        fillColor: item.close >= item.open ? '#089981' : '#f23645'
    }));
    
    // Update chart series
    window.mainChart.updateSeries([{
        name: `BTC/USDT (${interval})`,
        data: candleData
    }, {
        name: 'Volume',
        data: volumeData
    }]);
    
    // Update x-axis time format based on interval
    const timeFormats = {
        '1m': { unit: 'minute', displayFormats: { minute: 'HH:mm' } },
        '5m': { unit: 'minute', displayFormats: { minute: 'HH:mm' } },
        '15m': { unit: 'minute', displayFormats: { minute: 'HH:mm' } },
        '30m': { unit: 'minute', displayFormats: { minute: 'HH:mm' } },
        '1h': { unit: 'hour', displayFormats: { hour: 'MMM dd HH:mm' } },
        '4h': { unit: 'hour', displayFormats: { hour: 'MMM dd HH:mm' } },
        '1d': { unit: 'day', displayFormats: { day: 'MMM dd' } }
    };
    
    window.mainChart.updateOptions({
        xaxis: {
            type: 'datetime',
            labels: {
                datetimeUTC: false,
                style: { colors: '#787b86', fontSize: '11px' },
                datetimeFormatter: {
                    year: 'yyyy',
                    month: "MMM 'yy",
                    day: 'dd MMM',
                    hour: 'HH:mm'
                }
            },
            axisBorder: { color: '#2a2e39' },
            axisTicks: { color: '#2a2e39' },
            crosshairs: {
                show: true,
                stroke: { color: '#505050', width: 1, dashArray: 3 }
            }
        }
    });
    
    // Update current price display
    const latestData = validData[validData.length - 1];
    if (latestData) {
        updateCurrentPrice(latestData.close);
    }
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

// Global state for dynamic loading
let oldestTimestamp = null;
let isLoadingHistory = false;
let timeframeData = {}; // Store data per timeframe
let currentTimeframe = '1m'; // Track current timeframe

async function initializeCharts() {
    console.log('Initializing empty chart - data will load on demand');
    try {
        // Set initial timeframe
        currentTimeframe = '1m';

        // Initialize empty timeframe data
        timeframeData = {};

        const container = document.getElementById('mainChart');
        console.log('Container found:', container);

        // Verify ApexCharts is loaded
        if (typeof ApexCharts === 'undefined') {
            console.error('ApexCharts is not loaded');
            return;
        }

        // ApexCharts options with empty data initially
        const options = {
            series: [{
                name: 'BTC/USDT',
                type: 'candlestick',
                data: [] // Start with empty data
            }, {
                name: 'Volume',
                type: 'bar',
                data: [] // Start with empty data
            }],
            chart: {
                type: 'candlestick',
                height: 500,
                background: '#0a0a0a',
                id: 'mainChart',
                events: {
                    // Dynamic Loading Event - only load when user interacts
                    zoomed: function(chartContext, { xaxis }) {
                        if (xaxis.min < timeframeData[currentTimeframe]?.oldestTimestamp + 60000 * 100 && !isLoadingHistory) {
                            loadHistoricalData(chartContext);
                        }
                    },
                    scrolled: function(chartContext, { xaxis }) {
                        if (xaxis.min < timeframeData[currentTimeframe]?.oldestTimestamp + 60000 * 100 && !isLoadingHistory) {
                            loadHistoricalData(chartContext);
                        }
                    }
                },
                toolbar: {
                    show: true,
                    tools: {
                        download: true,
                        selection: true,
                        zoom: true,
                        zoomin: true,
                        zoomout: true,
                        pan: true,
                        reset: true
                    },
                    autoSelected: 'zoom'
                },
                animations: {
                    enabled: true,
                    speed: 500
                },
                zoom: {
                    enabled: true,
                    type: 'xy',
                    autoScaleYaxis: true
                }
            },
            title: {
                text: 'BTC/USDT - Click timeframe button or select custom range to load data',
                align: 'left',
                style: {
                    fontSize: '16px',
                    fontWeight: 'bold',
                    color: '#d1d4dc'
                }
            },
            noData: {
                text: 'Click a timeframe button to load data',
                align: 'center',
                verticalAlign: 'middle',
                style: {
                    color: '#787b86',
                    fontSize: '18px'
                }
            },
            xaxis: {
                type: 'datetime',
                labels: {
                    datetimeUTC: false,
                    style: {
                        colors: '#787b86',
                        fontSize: '11px'
                    },
                    datetimeFormatter: {
                        year: 'yyyy',
                        month: "MMM 'yy",
                        day: 'dd MMM',
                        hour: 'HH:mm'
                    }
                },
                axisBorder: {
                    color: '#2a2e39'
                },
                axisTicks: {
                    color: '#2a2e39'
                },
                crosshairs: {
                    show: true,
                    stroke: {
                        color: '#505050',
                        width: 1,
                        dashArray: 3
                    }
                }
            },
            yaxis: [{
                seriesName: 'BTC/USDT',
                labels: {
                    style: {
                        colors: '#787b86',
                        fontSize: '11px'
                    },
                    formatter: function(val) {
                        return '$' + val.toLocaleString('en-US', {
                            minimumFractionDigits: 0,
                            maximumFractionDigits: 0
                        });
                    },
                    offsetX: -10
                },
                opposite: true,
                tooltip: {
                    enabled: true
                },
                crosshairs: {
                    show: true,
                    stroke: {
                        color: '#505050',
                        width: 1,
                        dashArray: 3
                    }
                }
            }, {
                seriesName: 'Volume',
                opposite: false,
                show: false
            }],
            plotOptions: {
                candlestick: {
                    colors: {
                        upward: '#089981',
                        downward: '#f23645'
                    },
                    wick: {
                        useFillColor: true
                    }
                },
                bar: {
                    columnWidth: '80%'
                }
            },
            tooltip: {
                enabled: true,
                shared: true,
                theme: 'dark',
                custom: function({ seriesIndex, dataPointIndex, w }) {
                    if (!w.config.series[0].data[dataPointIndex]) return 'No data loaded';
                    const candle = w.config.series[0].data[dataPointIndex];
                    const volume = w.config.series[1].data[dataPointIndex];
                    const o = candle.y[0];
                    const h = candle.y[1];
                    const l = candle.y[2];
                    const c = candle.y[3];
                    const change = ((c - o) / o * 100).toFixed(2);
                    const changeColor = c >= o ? '#089981' : '#f23645';
                    const date = new Date(candle.x).toLocaleString();

                    return `
                        <div style="background: #1a1a2e; border: 1px solid #2a2e39; border-radius: 4px; padding: 10px; font-size: 12px;">
                            <div style="color: #787b86; margin-bottom: 6px;">${date}</div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                                <span style="color: #787b86;">O:</span><span style="color: #d1d4dc;">$${o.toLocaleString()}</span>
                                <span style="color: #787b86;">H:</span><span style="color: #089981;">$${h.toLocaleString()}</span>
                                <span style="color: #787b86;">L:</span><span style="color: #f23645;">$${l.toLocaleString()}</span>
                                <span style="color: #787b86;">C:</span><span style="color: #d1d4dc;">$${c.toLocaleString()}</span>
                            </div>
                            <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid #2a2e39;">
                                <span style="color: #787b86;">Change:</span>
                                <span style="color: ${changeColor};">${change >= 0 ? '+' : ''}${change}%</span>
                            </div>
                            <div style="color: #787b86; margin-top: 4px;">
                                Vol: ${volume.y.toLocaleString()}
                            </div>
                        </div>
                    `;
                }
            },
            grid: {
                borderColor: '#2a2e39',
                strokeDashArray: 3,
                xaxis: {
                    lines: {
                        show: false
                    }
                },
                yaxis: {
                    lines: {
                        show: true
                    }
                }
            },
            legend: {
                show: false
            },
            stroke: {
                width: [1, 0]
            }
        };

        // Create the chart with empty data
        window.mainChart = new ApexCharts(container, options);
        window.mainChart.render();
        console.log('Empty chart initialized successfully - waiting for user interaction');

        // Make responsive
        const resizeChart = () => {
            if (window.mainChart && container.offsetWidth > 0) {
                window.mainChart.updateOptions({
                    chart: {
                        width: container.offsetWidth,
                        height: 500
                    }
                });
            }
        };
        window.addEventListener('resize', resizeChart);
        setTimeout(resizeChart, 100);

    } catch (error) {
        console.error('Error initializing empty chart:', error);
    }
}

async function loadHistoricalData(chartContext) {
    if (isLoadingHistory || !timeframeData[currentTimeframe]) return;
    
    isLoadingHistory = true;
    showToast('Loading historical data...', 'info');
    
    try {
        const currentData = timeframeData[currentTimeframe];
        const oldestTime = currentData.oldestTimestamp;
        
        // Calculate how much historical data to load (based on current visible range)
        const chartRange = chartContext.w.config.xaxis.range;
        let loadRange;
        
        if (chartRange && chartRange.max && chartRange.min) {
            // Load data for 2x the current visible range before the oldest point
            const visibleRange = chartRange.max - chartRange.min;
            loadRange = visibleRange * 2;
        } else {
            // Fallback: load 500 candles worth
            const multiplier = {
                '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '4h': 240, '1d': 1440
            };
            loadRange = 500 * multiplier[currentTimeframe] * 60 * 1000; // in milliseconds
        }
        
        const newStartTime = oldestTime - loadRange;
        const interval = currentTimeframe === 'custom' ? '1m' : currentTimeframe;
        
        // Fetch historical data for the specific range
        const response = await fetchData(`/api/ohlcv?start=${newStartTime}&end=${oldestTime}&interval=${interval}`);
        
        if (response && response.data && response.data.length > 0) {
            const newHistory = response.data;
            
            // Filter out any overlaps with existing data
            const existingTimestamps = new Set(currentData.data.map(item => new Date(item.timestamp).getTime()));
            const uniqueHistory = newHistory.filter(item => 
                !existingTimestamps.has(new Date(item.timestamp).getTime()) &&
                new Date(item.timestamp).getTime() < oldestTime
            );
            
            if (uniqueHistory.length === 0) {
                isLoadingHistory = false;
                return;
            }
            
            // Update timeframe data
            currentData.data.unshift(...uniqueHistory);
            currentData.oldestTimestamp = newStartTime;
            
            // Format for ApexCharts
            const newCandles = uniqueHistory.map(item => ({
                x: new Date(item.timestamp).getTime(),
                y: [item.open, item.high, item.low, item.close]
            }));
            
            const newVolume = uniqueHistory.map(item => ({
                x: new Date(item.timestamp).getTime(),
                y: item.volume,
                fillColor: item.close >= item.open ? '#089981' : '#f23645'
            }));
            
            // Get current series and prepend new data
            const currentSeries = chartContext.w.config.series;
            const currentCandles = currentSeries[0].data;
            const currentVolume = currentSeries[1].data;
            
            const updatedCandles = [...newCandles, ...currentCandles];
            const updatedVolume = [...newVolume, ...currentVolume];
            
            // Update chart
            chartContext.updateSeries([{
                data: updatedCandles
            }, {
                data: updatedVolume
            }]);
            
            showToast(`Loaded ${uniqueHistory.length} historical ${currentTimeframe === 'custom' ? '1m' : currentTimeframe} candles`, 'success');
        }
    } catch (error) {
        console.error('Error loading history:', error);
        showToast('Failed to load historical data', 'error');
    } finally {
        isLoadingHistory = false;
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

    // Check if Chart.js is available
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js not loaded, skipping accuracy chart');
        return;
    }

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

    try {
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
                        type: 'category',
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
    } catch (error) {
        console.error('Error creating accuracy chart:', error);
    }
}

function updateChartData() {
    if (!window.mainChart || !timeframeData[currentTimeframe]) return;

    // Fetch latest OHLCV data for current timeframe
    const lastTimestamp = timeframeData[currentTimeframe].newestTimestamp;
    const interval = currentTimeframe === 'custom' ? '1m' : currentTimeframe;
    const response = fetchData(`/api/ohlcv?start=${lastTimestamp}&interval=${interval}`)
        .then(result => {
            if (result.error || !result.data || result.data.length === 0) return;

            const newData = result.data.map(item => ({
                x: new Date(item.timestamp).getTime(),
                o: item.open,
                h: item.high,
                l: item.low,
                c: item.close,
                v: item.volume
            }));

            // Filter new data that's after the last timestamp
            const filteredNewData = newData.filter(item => item.x > lastTimestamp);

            if (filteredNewData.length > 0) {
                // Update timeframe data
                timeframeData[currentTimeframe].data.push(...result.data.filter(item => 
                    new Date(item.timestamp).getTime() > lastTimestamp
                ));
                timeframeData[currentTimeframe].newestTimestamp = Math.max(
                    ...filteredNewData.map(item => item.x.getTime())
                );

                // Add new data points to chart series
                const newCandles = filteredNewData.map(d => ({
                    x: d.x.getTime(),
                    y: [d.o, d.h, d.l, d.c]
                }));
                
                const newVolume = filteredNewData.map(d => ({
                    x: d.x.getTime(),
                    y: d.v,
                    fillColor: d.c >= d.o ? '#089981' : '#f23645'
                }));

                // Get current series and append new data
                const currentSeries = window.mainChart.w.config.series;
                const updatedCandles = [...currentSeries[0].data, ...newCandles];
                const updatedVolume = [...currentSeries[1].data, ...newVolume];

                // Keep only last 1000 data points to prevent memory issues
                if (updatedCandles.length > 1000) {
                    const excess = updatedCandles.length - 1000;
                    updatedCandles.splice(0, excess);
                    updatedVolume.splice(0, excess);
                    timeframeData[currentTimeframe].oldestTimestamp = updatedCandles[0].x;
                }

                // Update chart without animation for real-time feel
                window.mainChart.updateSeries([{
                    data: updatedCandles
                }, {
                    data: updatedVolume
                }], false);

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
                label: 'Historical Data',
                data: [],
                color: {
                    up: '#089981',
                    down: '#f23645',
                    unchanged: '#787b86',
                },
                borderColor: {
                    up: '#089981',
                    down: '#f23645',
                    unchanged: '#787b86',
                },
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#d1d4dc'
                    }
                },
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(26, 26, 26, 0.9)',
                    titleColor: '#d1d4dc',
                    bodyColor: '#d1d4dc',
                    borderColor: '#2a2e39',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            if (!point) return '';
                            return [
                                `O: ${point.o.toFixed(2)}`,
                                `H: ${point.h.toFixed(2)}`,
                                `L: ${point.l.toFixed(2)}`,
                                `C: ${point.c.toFixed(2)}`
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'minute',
                        displayFormats: {
                            minute: 'HH:mm'
                        },
                        tooltipFormat: 'MMM d, HH:mm'
                    },
                    grid: {
                        color: '#2a2e39',
                        borderColor: '#2a2e39'
                    },
                    ticks: {
                        color: '#787b86',
                        source: 'auto'
                    },
                    adapters: {
                        date: {
                            locale: 'en'
                        }
                    }
                },
                y: {
                    position: 'right',
                    grid: {
                        color: '#2a2e39',
                        borderColor: '#2a2e39'
                    },
                    ticks: {
                        color: '#787b86',
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
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
            x: new Date(item.timestamp).valueOf(),
            o: item.open,
            h: item.high,
            l: item.low,
            c: item.close
        }));

        const predictedCandles = result.predicted_candles.map(item => ({
            x: new Date(item.timestamp).valueOf(),
            o: item.open,
            h: item.high,
            l: item.low,
            c: item.close
        }));

        // Update chart
        // We use two datasets to distinguish historical vs predicted if we want, 
        // but chartjs-chart-financial usually expects one dataset for continuous candlestick.
        // To visualize predictions differently, we can use a second dataset or just append.
        // Let's use two datasets for clarity.
        
        predictionChart.data.datasets = [
            {
                label: 'Historical',
                data: historicalData,
                color: {
                    up: '#089981',
                    down: '#f23645',
                    unchanged: '#787b86',
                },
                borderColor: {
                    up: '#089981',
                    down: '#f23645',
                    unchanged: '#787b86',
                },
                borderWidth: 1
            },
            {
                label: 'Predicted',
                data: predictedCandles,
                color: {
                    up: '#00b4d8', // Different color for predictions (Blue-ish)
                    down: '#0077b6',
                    unchanged: '#90e0ef',
                },
                borderColor: {
                    up: '#00b4d8',
                    down: '#0077b6',
                    unchanged: '#90e0ef',
                },
                borderWidth: 1
            }
        ];
        
        predictionChart.update();

        // Update next candle details
        if (predictedCandles.length > 0) {
            const next = result.predicted_candles[0]; // Use original result for volume access
            document.getElementById('next-open').textContent = `$${next.open.toFixed(2)}`;
            document.getElementById('next-high').textContent = `$${next.high.toFixed(2)}`;
            document.getElementById('next-low').textContent = `$${next.low.toFixed(2)}`;
            document.getElementById('next-close').textContent = `$${next.close.toFixed(2)}`;
            document.getElementById('next-volume').textContent = next.volume.toLocaleString();
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