from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import pandas as pd
import numpy as np
import time
import subprocess
import io
import logging
from src.inference import InferenceService

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = "/app/data/ohlcv.csv"
MACRO_PATH = "/app/data/macro.csv"

# Global Service Instance
inference_service = InferenceService(data_path=DATA_PATH, macro_path=MACRO_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Loads the model once on startup.
    """
    try:
        logger.info("Starting up... Loading model.")
        inference_service.load_model()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # We don't raise here to allow the app to start even if model fails (e.g. for health checks)
        # But prediction endpoints will fail.
    yield
    logger.info("Shutting down...")

app = FastAPI(title="SOFARE-AI Prediction Engine", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="/app/static"), name="static")

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str = "latest_shared"
    drift_detected: bool = False

class CandlestickPrediction(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    prediction_type: str = "forecasted"

class CandlestickResponse(BaseModel):
    current_data: list[CandlestickPrediction]
    predicted_candles: list[CandlestickPrediction]
    model_version: str = "latest_shared"
    drift_detected: bool = False

# --- Utility for Data Fetching (kept for /api/ohlcv) ---
def read_last_lines(filepath, n_lines):
    """
    Efficiently read the last n lines of a large CSV file using the 'tail' command.
    """
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    try:
        header = subprocess.check_output(['head', '-n', '1', filepath]).decode('utf-8').strip()
        tail = subprocess.check_output(['tail', '-n', str(n_lines), filepath]).decode('utf-8')
        
        if not tail: return pd.DataFrame()
        
        if tail.startswith(header):
            content = tail
        else:
            content = header + '\n' + tail
            
        return pd.read_csv(io.StringIO(content))
    except Exception as e:
        logger.error(f"Error reading tail: {e}")
        return pd.read_csv(filepath).tail(n_lines)

# --- Endpoints ---

@app.api_route("/predict", methods=["GET", "POST"], response_model=PredictionResponse)
async def predict():
    try:
        prediction, drift = inference_service.predict_next()
        
        if prediction is None:
            raise HTTPException(status_code=503, detail="Not enough data for inference")
            
        return {
            "prediction": float(prediction),
            "model_version": "latest_shared",
            "drift_detected": drift
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/candlestick", response_model=CandlestickResponse)
async def predict_candlestick(forecast_steps: int = 5):
    try:
        # 1. Get Historical Context (Last 20 candles)
        # We use the utility function here for raw data display
        df_hist = read_last_lines(DATA_PATH, 20)
        
        current_candles = []
        if not df_hist.empty:
            # Ensure timestamp format
            if 'timestamp' in df_hist.columns and pd.api.types.is_numeric_dtype(df_hist['timestamp']):
                df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], unit='ms')
            else:
                df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
                
            for _, row in df_hist.iterrows():
                current_candles.append(CandlestickPrediction(
                    timestamp=row['timestamp'].isoformat(),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume']),
                    prediction_type="historical"
                ))

        # 2. Generate Predictions
        predictions_data = inference_service.predict_candlesticks(steps=forecast_steps)
        
        predicted_candles = []
        for p in predictions_data:
            predicted_candles.append(CandlestickPrediction(
                timestamp=p['timestamp'].isoformat(),
                open=float(p['open']),
                high=float(p['high']),
                low=float(p['low']),
                close=float(p['close']),
                volume=float(p['volume']),
                prediction_type="forecasted"
            ))

        return CandlestickResponse(
            current_data=current_candles,
            predicted_candles=predicted_candles,
            model_version="latest_shared",
            drift_detected=False
        )
    except Exception as e:
        logger.error(f"Candlestick prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    model_status = "loaded" if inference_service.model is not None else "not_loaded"
    return {"status": "ok", "model_status": model_status}

@app.get("/api/ohlcv")
def get_ohlcv_data(limit: int = 100, interval: str = "1m", start: int = None, end: int = None):
    """Get recent OHLCV data for charting"""
    if not os.path.exists(DATA_PATH):
        return {"error": "No data available"}

    try:
        df = pd.DataFrame()
        
        # Dynamic Loading Strategy
        if start is not None or end is not None:
            current_time = int(time.time() * 1000)
            thirty_days_ago = current_time - (30 * 24 * 60 * 60 * 1000)
            
            is_recent_data = False
            if start is not None and start >= thirty_days_ago:
                is_recent_data = True
            elif end is not None and end >= thirty_days_ago:
                is_recent_data = True
            elif start is None and end is None:
                is_recent_data = True
            
            if is_recent_data:
                if start is not None and end is not None:
                    time_range_ms = end - start
                    estimated_records = int(time_range_ms / (60 * 1000)) + 1000
                else:
                    multiplier = 1
                    if interval == '5m': multiplier = 5
                    elif interval == '15m': multiplier = 15
                    elif interval == '30m': multiplier = 30
                    elif interval == '1h': multiplier = 60
                    elif interval == '4h': multiplier = 240
                    elif interval == '1d': multiplier = 1440
                    estimated_records = limit * multiplier * 2
                
                estimated_records = min(estimated_records, 100000)
                df = read_last_lines(DATA_PATH, estimated_records)
                
                if start is not None:
                    df = df[df.iloc[:, 0] >= start]
                if end is not None:
                    df = df[df.iloc[:, 0] <= end]
            else:
                chunks = []
                chunk_size = 50000
                for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size):
                    chunk_ts = pd.to_numeric(chunk.iloc[:, 0], errors='coerce')
                    mask = pd.Series(True, index=chunk.index)
                    if start is not None: mask &= (chunk_ts >= start)
                    if end is not None: mask &= (chunk_ts <= end)
                    if mask.any(): chunks.append(chunk[mask])
                    if end is not None and chunk_ts.max() > end: break
                if chunks: df = pd.concat(chunks)
        else:
            limit = min(limit, 5000)
            multiplier = 1
            if interval == '5m': multiplier = 5
            elif interval == '15m': multiplier = 15
            elif interval == '30m': multiplier = 30
            elif interval == '1h': multiplier = 60
            elif interval == '4h': multiplier = 240
            elif interval == '1d': multiplier = 1440
            required_lines = limit * multiplier
            df = read_last_lines(DATA_PATH, required_lines + 100)

        if len(df) == 0:
            return {"error": "No data available"}

        df['timestamp'] = pd.to_datetime(df.iloc[:, 0], unit='ms')
        df = df.sort_values('timestamp')
        
        df['open'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        df['high'] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
        df['low'] = pd.to_numeric(df.iloc[:, 3], errors='coerce')
        df['close'] = pd.to_numeric(df.iloc[:, 4], errors='coerce')
        df['volume'] = pd.to_numeric(df.iloc[:, 5], errors='coerce')
        
        df = df.dropna()
        
        interval_map = {
            '1m': '1min', '5m': '5min', '15m': '15min',
            '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1D'
        }
        
        if interval in interval_map and interval != '1m':
            df = df.set_index('timestamp')
            df = df.resample(interval_map[interval]).agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna().reset_index()

        if start is None and end is None:
            df = df.tail(limit)

        data = []
        for _, row in df.iterrows():
            data.append({
                'timestamp': int(row['timestamp'].timestamp() * 1000),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })

        return {"data": data, "interval": interval, "count": len(data)}

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.get("/api/macro")
def get_macro_data():
    """Get latest macro indicators"""
    if not os.path.exists(MACRO_PATH):
        return {"error": "No macro data available"}

    try:
        df = pd.read_csv(MACRO_PATH)
        if len(df) == 0:
            return {"error": "No macro data available"}

        latest = df.iloc[-1]
        def safe_float(value):
            if pd.isna(value) or np.isnan(value): return None
            return float(value)
        
        return {
            "fed_funds_rate": safe_float(latest['fed_funds_rate']),
            "gold_price": safe_float(latest['gold_price']),
            "dxy": safe_float(latest['dxy']),
            "sp500": safe_float(latest['sp500']),
            "vix": safe_float(latest['vix']),
            "nasdaq": safe_float(latest['nasdaq']),
            "oil_price": safe_float(latest['oil_price'])
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def dashboard():
    return FileResponse("/app/static/index.html")
