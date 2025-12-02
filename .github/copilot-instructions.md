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