# SOFARE-AI Copilot Instructions

## Architecture Overview
SOFARE-AI is a microservices-based ML pipeline for cryptocurrency price prediction using OHLCV data from Binance and macro indicators (Fed Funds Rate, Gold, DXY). Services: ingestion (real-time data), training (PyTorch LSTM with rolling retraining), serving (FastAPI API), MLflow (model tracking).

## Key Components
- **Ingestion**: WebSocket client for Binance streams, FRED/yfinance for macro data, saves to shared CSV in `/app/data`.
- **Training**: Loads merged OHLCV/macro data, adds TA indicators (RSI, MACD), trains MultiModalLSTM (LSTM + dense fusion), logs to MLflow.
- **Serving**: Loads latest model from MLflow, processes recent data for predictions.
- **Data Flow**: Ingestion → shared CSVs → Training → MLflow → Serving.

## Development Workflows
- **Build/Run**: `docker-compose build && docker-compose up` (ingestion uses host network for WebSocket).
- **Training**: Runs continuously in container, retrains on rolling 2000-candle windows, checks drift with KS test.
- **Testing**: Unit tests in `tests/`, run via `python -m unittest` inside containers or locally with paths adjusted.
- **Debugging**: Logs to stdout, MLflow UI at localhost:5000, API at localhost:8000.

## Conventions
- **Data Merging**: Use `pd.merge_asof` with `direction='backward'` for time-series alignment, forward-fill macro data.
- **Features**: Always add TA indicators via `ta` library in `features.py` pattern.
- **Model**: MultiModalLSTM fuses time-series (LSTM) and macro (dense) branches.
- **Drift**: KS test on feature distributions before retraining.
- **Dependencies**: Torch installed separately in Dockerfile for CPU builds.

## Examples
- Add new indicator: Extend `add_technical_indicators()` in `training/src/features.py` using `ta` library.
- New macro source: Add to `MacroClient.fetch_data()` in `ingestion/src/macro_client.py`, save via `storage.save_macro()`.
- API endpoint: Add to `serving/src/app.py` FastAPI app, load model with `mlflow.pytorch.load_model()`.