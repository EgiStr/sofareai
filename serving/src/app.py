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
from sofare_common import SessionLocal, OHLCV
from datetime import datetime

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
# DATA_PATH and MACRO_PATH are deprecated in favor of DB

# Global Service Instance
inference_service = InferenceService()

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
def read_last_lines(n_lines):
    """
    Read last n lines from DB.
    """
    db = SessionLocal()
    try:
        query = db.query(OHLCV).filter(OHLCV.symbol == 'BTCUSDT').order_by(OHLCV.timestamp.desc()).limit(n_lines)
        records = query.all()
        records.reverse()
        
        if not records:
            return pd.DataFrame()
            
        data = [{
            'timestamp': int(r.timestamp.timestamp() * 1000),
            'open': r.open,
            'high': r.high,
            'low': r.low,
            'close': r.close,
            'volume': r.volume
        } for r in records]
        
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error reading from DB: {e}")
        return pd.DataFrame()
    finally:
        db.close()

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
        df_hist = read_last_lines(20)
        
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
    db = SessionLocal()
    try:
        query = db.query(OHLCV).filter(OHLCV.symbol == 'BTCUSDT')
        
        if start is not None:
            start_dt = datetime.fromtimestamp(start / 1000.0)
            query = query.filter(OHLCV.timestamp >= start_dt)
        
        if end is not None:
            end_dt = datetime.fromtimestamp(end / 1000.0)
            query = query.filter(OHLCV.timestamp <= end_dt)
            
        if start is None and end is None:
             query = query.order_by(OHLCV.timestamp.desc()).limit(limit)
        else:
             query = query.order_by(OHLCV.timestamp.asc())
             query = query.limit(10000)

        records = query.all()
        
        if start is None and end is None:
            records.reverse()
        
        if not records:
            return {"error": "No data available"}
            
        data = []
        for r in records:
            data.append({
                'timestamp': int(r.timestamp.timestamp() * 1000),
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume
            })
            
        return {"data": data, "interval": interval, "count": len(data)}

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.get("/api/macro")
def get_macro_data():
    """Get latest macro indicators"""
    db = SessionLocal()
    try:
        # Get latest timestamp
        last_ts_record = db.query(MacroIndicator.timestamp).order_by(MacroIndicator.timestamp.desc()).first()
        if not last_ts_record:
             return {"error": "No macro data available"}
        
        last_ts = last_ts_record[0]
        
        # Get all indicators for this timestamp
        records = db.query(MacroIndicator).filter(MacroIndicator.timestamp == last_ts).all()
        
        result = {}
        for r in records:
            result[r.name] = r.value
            
        return result
    except Exception as e:
        logger.error(f"Error fetching macro: {e}")
        return {"error": str(e)}
    finally:
        db.close()

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def dashboard():
    return FileResponse("/app/static/index.html")
