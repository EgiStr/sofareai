# SOFARE-AI Copilot Instructions

## Architecture Overview
SOFARE-AI is a microservices-based ML pipeline for cryptocurrency price prediction using multi-modal data: OHLCV from Binance/Yahoo Finance, macro indicators (Fed Funds Rate), and Safe Haven assets (Gold, DXY, S&P500, VIX, NASDAQ, Oil). Services: ingestion (real-time data), training (PyTorch with rolling retraining), serving (FastAPI API), MLflow (model tracking).

## Project Structure
```
sofareai/
├── data/               # Shared data storage (CSV) - minute-level granularity
├── ingestion/          # Data ingestion service with multi-source fallback
├── serving/            # Model serving API
├── training/           # Model training service
├── tests/              # Unit tests
├── notebooks/          # Jupyter notebooks
├── mlflow_data/        # MLflow tracking data
├── docker-compose.yml  # Orchestration
└── Makefile            # Automation
```

## Key Components
- **Ingestion**: Multi-source data pipeline (Binance WebSocket → Yahoo Finance → Alpha Vantage → CCXT exchanges), expands daily macro data to minute-level granularity (1440 records/day historical, 60/hour real-time).
- **Training**: Loads merged multi-modal minute-level data, uses FULL DATASET training with proper temporal train/val/test split (70/15/15), adds 19+ TA indicators, trains SofareM3 (Transformer/TCN encoders + attention fusion + multi-task head), logs to MLflow.
- **Serving**: Loads latest model from MLflow, processes recent data for predictions.
- **Data Flow**: Ingestion → shared CSVs (minute granularity) → Training → MLflow → Serving.

## Development Workflows
- **Management**: Use `make` commands for standard operations.
  - `make build`: Build all services.
  - `make up`: Start the stack in detached mode.
  - `make logs`: View logs for all services.
  - `make down`: Stop and remove containers.
  - `make test`: Run unit tests.
  - `make clean`: Remove artifacts and temporary files.
- **Training**: Runs continuously in container, retrains on rolling windows, checks drift with KS test.
- **Debugging**: Logs to stdout, MLflow UI at localhost:5000, API at localhost:8000/docs.

## Critical Patterns & Conventions

### Data Architecture
- **Shared CSV Communication**: Services communicate via CSV files in `/data`, not APIs. Training reads `ohlcv.csv` + `macro.csv`, serving reads same files.
- **Minute-Level Standardization**: All data converted to minute timestamps. Daily macro data expanded to 1440 minute records using `pd.merge_asof(direction='backward')`.
- **Multi-Source Fallback**: Data collection tries Binance → Yahoo Finance → Alpha Vantage → CCXT exchanges. Deduplicate by timestamp, filter duplicates.

### Model Architecture
- **SofareM3 Structure**: Separate encoders per modality (micro/OHLCV+TA, macro/economic, safe/market assets) + attention fusion + multi-task head.
- **Multi-Task Learning**: Classification (up/down) + regression (return prediction) with combined CrossEntropy + Huber loss.
- **Feature Engineering**: Always use `ta` library for 19 indicators: RSI, Stochastic, MACD, SMA/EMA, Bollinger Bands, ATR, OBV, plus custom log_return, hl_range, rolling_vol_20.

### Training Strategy
- **Full Dataset Training**: Set `USE_FULL_DATASET=True`, never subsample. Use temporal 70/15/15 train/val/test split.
- **Temporal Ordering**: Never shuffle time-series data. Maintain chronological order for validation.
- **Drift Detection**: KS test on feature distributions before retraining. Retrain on rolling windows.

### API & Serving
- **Timestamp Format**: Always return milliseconds since epoch (`row['timestamp'].timestamp() * 1000`) for ApexCharts compatibility.
- **Model Loading**: Load from shared volume `/app/shared_model` (not MLflow registry). Config in `model_config.json`, weights in `model_weights.pth`.
- **Scalers**: Load from `scalers.pkl` with fallback to fit on current data if missing.

### Docker & Deployment
- **Host Networking**: Ingestion uses `network_mode: "host"` for WebSocket connections.
- **Shared Volumes**: Data exchange via mounted volumes, models via shared_model volume.
- **Service Dependencies**: MLflow starts first, then training (depends_on mlflow), then serving (depends_on mlflow + training).

## Integration Points
- **External APIs**: FRED (macro), Yahoo Finance (safe haven), Binance WebSocket (crypto), Alpha Vantage/CCXT (fallbacks).
- **MLflow**: Experiment tracking only - models stored separately in shared volume.
- **Charting**: ApexCharts requires millisecond timestamps, not ISO strings.
- **WebSocket**: Real-time data ingestion with callback pattern.

## Code Patterns & Examples

### Multi-Task Loss Function
```python
def multi_task_loss(cls_pred, reg_pred, cls_target, reg_target):
    cls_loss = F.cross_entropy(cls_pred, cls_target)
    reg_loss = F.huber_loss(reg_pred.squeeze(), reg_target, delta=1.0)
    total_loss = 0.5 * cls_loss + 0.5 * reg_loss  # Equal weighting
    return total_loss, cls_loss, reg_loss
```

### Temporal Data Split (CRITICAL)
```python
# NEVER shuffle time-series data
def temporal_split(data, train_ratio=0.7, val_ratio=0.15):
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * (train_ratio + val_ratio))

    train_data = data[:n_train]      # Chronological first 70%
    val_data = data[n_train:n_val]   # Next 15%
    test_data = data[n_val:]         # Final 15% (future data)

    return train_data, val_data, test_data
```

### Feature Engineering with TA Library
```python
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

def add_technical_indicators(df):
    # RSI
    rsi_indicator = RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi_indicator.rsi()

    # MACD
    macd_indicator = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_indicator.macd()
    df["macd_signal"] = macd_indicator.macd_signal()
    df["macd_diff"] = macd_indicator.macd_diff()

    # Bollinger Bands
    bb_indicator = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb_indicator.bollinger_hband()
    df["bb_lower"] = bb_indicator.bollinger_lband()
    df["bb_middle"] = bb_indicator.bollinger_mavg()

    return df
```

### Timestamp Conversion for ApexCharts
```python
# Convert pandas timestamp to milliseconds for ApexCharts
def convert_to_milliseconds(timestamp_series):
    return (timestamp_series.astype('int64') // 10**6).astype(str)

# Usage in API response
response_data = {
    "timestamp": int(row['timestamp'].timestamp() * 1000),  # milliseconds
    "prediction": prediction_value
}
```

### Model Loading Pattern
```python
def load_model_from_registry():
    """Load model from shared volume, not MLflow registry"""
    model_path = "/app/shared_model/model_weights.pth"
    config_path = "/app/shared_model/model_config.json"

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize model
    model = SofareM3(**config)

    # Load weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model
```

### Data Quality Validation
```python
def validate_data_quality(df):
    """Comprehensive data validation before training"""
    # Null value detection
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values detected: {null_counts}")

    # Duplicate timestamp removal
    df = df.drop_duplicates(subset=['timestamp'], keep='last')

    # Temporal ordering validation
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Statistical outlier detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = df[z_scores > 3]
        if len(outliers) > 0:
            logger.warning(f"Outliers in {col}: {len(outliers)} records")

    return df
```

## Examples
- Add new indicator: Extend `add_technical_indicators()` in `training/src/features.py` using `ta` library.
- New data source: Add to `MacroClient.fetch_data()` in `ingestion/src/macro_client.py`, expand to minute-level in `fetch_historical_macro()`.
- Model extension: Modify `SofareM3` in `training/src/model.py`, update fusion/encoders as needed.
- API endpoint: Add to `serving/src/app.py` FastAPI app, load model with `mlflow.pytorch.load_model()`.
- Multi-task training: Use `nn.CrossEntropyLoss()` + `nn.HuberLoss(delta=1.0)` combined loss.
- Minute expansion: For daily data, create 1440 records per day with `day_start + timedelta(minutes=minute_offset)`.
- Data source fallback: Implement in `fetch_comprehensive_historical()` - try Binance → CCXT → Yahoo Finance → Alpha Vantage, deduplicate by timestamp.
- Validation: Run `make test` to execute unit tests, check MLflow UI at localhost:5000 for model metrics.
- Monitoring: View container logs with `make logs`, check data integrity in `data/` directory.