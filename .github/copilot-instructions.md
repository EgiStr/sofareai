# SOFARE-AI Copilot Instructions

## Architecture Overview
SOFARE-AI is a microservices-based ML pipeline for cryptocurrency price prediction using multi-modal data: OHLCV from Binance, macro indicators (Fed Funds Rate), and Safe Haven assets (Gold, DXY, S&P500, VIX, NASDAQ, Oil). Services: ingestion (real-time data), training (PyTorch with rolling retraining), serving (FastAPI API), MLflow (model tracking).

## Project Structure
```
sofareai/
├── data/               # Shared data storage (CSV)
├── ingestion/          # Data ingestion service
├── serving/            # Model serving API
├── training/           # Model training service
├── tests/              # Unit tests
├── notebooks/          # Jupyter notebooks
├── mlflow_data/        # MLflow tracking data
├── docker-compose.yml  # Orchestration
└── Makefile            # Automation
```

## Key Components
- **Ingestion**: WebSocket client for Binance streams, FRED/yfinance for macro/safe haven data, saves to shared CSV in `/app/data`.
- **Training**: Loads merged multi-modal data, adds 19+ TA indicators, trains SofareM3 (Transformer/TCN encoders + attention fusion + multi-task head), logs to MLflow.
- **Serving**: Loads latest model from MLflow, processes recent data for predictions.
- **Data Flow**: Ingestion → shared CSVs → Training → MLflow → Serving.

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

## Conventions
- **Data Merging**: Use `pd.merge_asof` with `direction='backward'` for time-series alignment, forward-fill macro data.
- **Modalities**: Separate micro (OHLCV+TA), macro (fed_funds_rate), safe (gold_price, dxy, sp500, vix, nasdaq, oil_price).
- **Features**: Always add TA indicators via `ta` library in `features.py` pattern, 19+ indicators required.
- **Model**: SofareM3 with separate encoders per modality, attention fusion, multi-task head (CE + Huber loss).
- **Training**: Multi-task learning with classification (up/down) + regression (return), combined loss.
- **Drift**: KS test on feature distributions before retraining.
- **MLflow**: Experiment "SOFARE-AI-Phase3", log artifacts, use run_id for model versioning.
- **Docker**: Host network for ingestion WebSocket, shared volumes for data exchange.
- **Dependencies**: Torch installed separately in Dockerfile for CPU builds.

## Examples
- Add new indicator: Extend `add_technical_indicators()` in `training/src/features.py` using `ta` library.
- New data source: Add to `MacroClient.fetch_data()` in `ingestion/src/macro_client.py`, update storage columns.
- Model extension: Modify `SofareM3` in `training/src/model.py`, update fusion/encoders as needed.
- API endpoint: Add to `serving/src/app.py` FastAPI app, load model with `mlflow.pytorch.load_model()`.
- Multi-task training: Use `nn.CrossEntropyLoss()` + `nn.HuberLoss(delta=1.0)` combined loss.