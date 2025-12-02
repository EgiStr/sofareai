# Configuration

This guide covers all configuration options for SOFARE-AI.

## Environment Variables

Create a `.env` file in the project root:

```env
# ===========================================
# Data Source Configuration
# ===========================================

# Primary data source: binance, yahoo, or alpha_vantage
DATA_SOURCE=binance

# Binance API (optional, for authenticated endpoints)
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

# Alpha Vantage API (for macro data fallback)
ALPHA_VANTAGE_API_KEY=your_api_key

# ===========================================
# MLflow Configuration
# ===========================================

MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=SOFARE-AI-Phase3

# ===========================================
# Model Configuration
# ===========================================

# Sequence length for time series
SEQUENCE_LENGTH=60

# Training batch size
BATCH_SIZE=32

# Hidden layer size
HIDDEN_SIZE=128

# Embedding dimension
EMBED_DIM=128

# ===========================================
# Training Configuration
# ===========================================

# Retrain interval in seconds (default: 1 hour)
RETRAIN_INTERVAL=3600

# Minimum samples required for training
MIN_TRAINING_SAMPLES=1000

# Learning rate
LEARNING_RATE=0.001

# Number of epochs
NUM_EPOCHS=50

# ===========================================
# Drift Detection
# ===========================================

# KS test p-value threshold
DRIFT_KS_THRESHOLD=0.05

# PSI threshold (0.1 = moderate, 0.2 = significant)
DRIFT_PSI_THRESHOLD=0.2

# Enable automatic retraining on drift
AUTO_RETRAIN_ON_DRIFT=true

# ===========================================
# API Configuration
# ===========================================

# FastAPI host and port
API_HOST=0.0.0.0
API_PORT=8000

# CORS origins (comma-separated)
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

## Docker Compose Configuration

### Service Configuration

```yaml title="docker-compose.yml"
services:
  ingestion:
    build: ./ingestion
    environment:
      - DATA_SOURCE=${DATA_SOURCE:-binance}
    volumes:
      - ./data:/app/data
    env_file: .env

  training:
    build: ./training
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - BATCH_SIZE=${BATCH_SIZE:-32}
    volumes:
      - ./data:/app/data
      - ./shared_model:/app/shared_model
    depends_on:
      - mlflow

  serving:
    build: ./serving
    ports:
      - "${API_PORT:-8000}:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ./data:/app/data
      - ./shared_model:/app/shared_model

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.2
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow_data:/mlflow
```

### Resource Limits

Add resource constraints for production:

```yaml title="docker-compose.prod.yml"
services:
  training:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
  
  serving:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
      replicas: 3
```

## Model Configuration

### SofareM3 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `micro_input_size` | 19 | Number of OHLCV + TA features |
| `macro_input_size` | 3 | Number of macro features |
| `safe_input_size` | 4 | Number of safe haven features |
| `hidden_size` | 128 | Hidden layer dimension |
| `embed_dim` | 128 | Embedding dimension |
| `num_heads` | 8 | Attention heads |
| `dropout` | 0.1 | Dropout rate |

### Feature Configuration

Modify features in `training/src/features.py`:

```python title="training/src/features.py"
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical analysis indicators."""
    
    # Momentum Indicators
    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
    
    # Trend Indicators
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    
    # Volatility Indicators
    bb = BollingerBands(close=df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    
    # Add custom indicators here
    
    return df
```

## Data Source Configuration

### Binance WebSocket

```python title="ingestion/src/binance_client.py"
BINANCE_CONFIG = {
    "symbol": "BTCUSDT",
    "interval": "1m",
    "stream_type": "kline",
    "reconnect_delay": 5,
    "max_reconnect_attempts": 10
}
```

### Yahoo Finance Fallback

```python title="ingestion/src/macro_client.py"
YAHOO_SYMBOLS = {
    "gold": "GC=F",
    "dxy": "DX-Y.NYB",
    "sp500": "^GSPC",
    "vix": "^VIX",
    "nasdaq": "^IXIC",
    "oil": "CL=F"
}
```

## Logging Configuration

```python title="logging.conf"
[loggers]
keys=root,sofareai

[handlers]
keys=console,file

[formatters]
keys=standard

[logger_root]
level=INFO
handlers=console

[logger_sofareai]
level=DEBUG
handlers=console,file
qualname=sofareai
propagate=0

[handler_console]
class=StreamHandler
level=INFO
formatter=standard
args=(sys.stdout,)

[handler_file]
class=FileHandler
level=DEBUG
formatter=standard
args=('/app/logs/sofareai.log', 'a')

[formatter_standard]
format=%(asctime)s [%(levelname)s] %(name)s: %(message)s
datefmt=%Y-%m-%d %H:%M:%S
```

## Security Configuration

### API Authentication

For production, add authentication:

```python title="serving/src/auth.py"
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### CORS Settings

```python title="serving/src/app.py"
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Configuration Best Practices

!!! tip "Environment-Specific Configs"
    
    Use different `.env` files for each environment:
    
    - `.env.development`
    - `.env.staging`
    - `.env.production`
    
    ```bash
    # Load specific environment
    docker compose --env-file .env.production up
    ```

!!! warning "Secrets Management"
    
    Never commit API keys to version control. Use:
    
    - Docker secrets
    - HashiCorp Vault
    - AWS Secrets Manager
    - Environment variables from CI/CD

## Next Steps

- [Architecture Overview](../architecture/index.md)
- [Deployment Guide](../operations/deployment.md)
