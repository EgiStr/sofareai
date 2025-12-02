# Quick Start

Get your first prediction in under 5 minutes!

## Prerequisites

Ensure you've completed the [Installation](installation.md) steps.

## Step 1: Start the Stack

```bash
# Navigate to project directory
cd sofareai

# Start all services
make up

# Wait for services to be ready (about 30 seconds)
docker compose logs -f --tail=20
```

!!! success "Services Ready"
    You should see logs indicating all services are running:
    ```
    sofare_mlflow      | [INFO] Listening at: http://0.0.0.0:5000
    sofare_serving     | INFO:     Application startup complete.
    sofare_ingestion   | [INFO] Connected to Binance WebSocket
    ```

## Step 2: Verify Services

### Check API Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok"}
```

### View MLflow UI

Open [http://localhost:5000](http://localhost:5000) in your browser.

![MLflow Dashboard](../assets/mlflow-dashboard.png)

## Step 3: Make a Prediction

### Using cURL

```bash
curl -X POST http://localhost:8000/predict
```

### Using Python

```python
import requests

response = requests.post("http://localhost:8000/predict")
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Model Version: {result['model_version']}")
print(f"Drift Detected: {result['drift_detected']}")
```

### Expected Response

```json
{
  "prediction": 0.0023,
  "model_version": "latest_shared",
  "drift_detected": false
}
```

!!! info "Prediction Interpretation"
    - `prediction`: Predicted log return for the next period
    - Positive value = expected price increase
    - Negative value = expected price decrease

## Step 4: View Candlestick Predictions

For more detailed forecasting:

```bash
curl -X POST "http://localhost:8000/predict/candlestick?forecast_steps=5"
```

Response includes:
- Last 20 historical candles
- 5 predicted future candles with OHLCV values

## Step 5: Explore the Dashboard

Visit the dashboard at [http://localhost:8000/static/index.html](http://localhost:8000/static/index.html)

Features:
- Real-time price chart
- Macro indicator panel
- Prediction overlay

## Step 6: Monitor Training

The training service runs continuously. Monitor it with:

```bash
# View training logs
docker compose logs -f training
```

Key events to watch:
```
[INFO] Starting training iteration
[INFO] Loaded 10000 samples
[INFO] Training completed - Accuracy: 0.63, Loss: 0.45
[INFO] Model saved to MLflow run_id: abc123
```

## Common Operations

### View All Data

```bash
# OHLCV data
curl http://localhost:8000/api/ohlcv?limit=10

# Macro indicators
curl http://localhost:8000/api/macro
```

### Stop Services

```bash
make down
```

### View Logs

```bash
# All services
make logs

# Specific service
docker compose logs -f serving
```

### Restart a Service

```bash
docker compose restart training
```

## Next Steps

<div class="grid cards" markdown>

-   :material-cog:{ .lg .middle } **Configure**

    ---

    Customize data sources, model parameters, and more

    [:octicons-arrow-right-24: Configuration](configuration.md)

-   :material-chart-box:{ .lg .middle } **Architecture**

    ---

    Deep dive into system design

    [:octicons-arrow-right-24: Architecture](../architecture/index.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Full API documentation

    [:octicons-arrow-right-24: API Docs](../api/index.md)

</div>

## Troubleshooting

!!! warning "No Predictions Available"
    
    If you get a 503 error, the model hasn't been trained yet.
    
    ```bash
    # Check if training is running
    docker compose logs training
    
    # Wait for first training iteration (about 5-10 minutes)
    ```

!!! warning "Insufficient Data"
    
    The ingestion service needs to collect enough data before training can start.
    
    ```bash
    # Check data collection
    wc -l data/ohlcv.csv
    
    # You need at least 60 + 50 = 110 rows
    ```
