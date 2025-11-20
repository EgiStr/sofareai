from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pytorch
import torch
import pandas as pd
import numpy as np
import os
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler

app = FastAPI(title="SOFARE-AI Prediction Engine")

# Configuration
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DATA_PATH = "/app/data/ohlcv.csv"
MACRO_PATH = "/app/data/macro.csv"
SEQUENCE_LENGTH = 60

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    drift_detected: bool = False

def get_latest_data():
    """
    Fetches the latest sequence of data for inference.
    In a real production system, this would come from a Feature Store.
    Here we read from the shared CSV.
    """
    if not os.path.exists(DATA_PATH):
        return None, None

    try:
        df = pd.read_csv(DATA_PATH)
        if len(df) < SEQUENCE_LENGTH + 50: # Buffer for indicators
            return None, None
            
        # Load Macro
        if os.path.exists(MACRO_PATH):
            macro_df = pd.read_csv(MACRO_PATH)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp'])
            df = pd.merge_asof(df.sort_values('timestamp'), macro_df.sort_values('timestamp'), on='timestamp', direction='backward')
            df[['fed_funds_rate', 'gold_price', 'dxy']] = df[['fed_funds_rate', 'gold_price', 'dxy']].fillna(method='ffill').fillna(0)
        else:
            df['fed_funds_rate'] = 0
            df['gold_price'] = 0
            df['dxy'] = 0

        # Feature Engineering (On the fly)
        # Ensure numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        rsi = RSIIndicator(close=df["close"], window=14)
        df["rsi"] = rsi.rsi()
        macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
        
        df = df.dropna()
        
        # Get last sequence
        if len(df) < SEQUENCE_LENGTH:
            return None, None
            
        last_sequence = df.iloc[-SEQUENCE_LENGTH:]
        
        feature_cols = ['close', 'volume', 'rsi', 'macd', 'macd_signal', 'macd_diff']
        macro_cols = ['fed_funds_rate', 'gold_price', 'dxy']
        
        x_seq = last_sequence[feature_cols].values
        x_macro = last_sequence[macro_cols].values[-1] # Take last macro state
        
        return x_seq, x_macro
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None

@app.post("/predict", response_model=PredictionResponse)
async def predict():
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    # Load Model
    # In production, we would load specific version or "Production" stage
    # Here we try to load the latest run from Phase 3 experiment
    try:
        # Find latest run
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("SOFARE-AI-Phase3")
        if not experiment:
            raise HTTPException(status_code=503, detail="Model not ready (Experiment not found)")
            
        runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
        if not runs:
            raise HTTPException(status_code=503, detail="Model not ready (No runs)")
            
        run_id = runs[0].info.run_id
        model_uri = f"runs:/{run_id}/model"
        
        # Load model (this is slow, should be cached in global var in real app)
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Get Data
    x_seq, x_macro = get_latest_data()
    if x_seq is None:
        raise HTTPException(status_code=503, detail="Not enough data for inference")

    # Preprocessing (Scaling) - Ideally load scaler from MLflow too
    # For MVP, we fit scaler on the single sequence (Approximation) or use fixed range
    # Using MinMax on the sequence itself is risky if range changes. 
    # Better: Load scaler artifact. For MVP: Simple normalization
    scaler = MinMaxScaler()
    x_seq_scaled = scaler.fit_transform(x_seq)
    
    # Reshape for model
    x_seq_tensor = torch.FloatTensor(x_seq_scaled).unsqueeze(0) # Batch size 1
    
    # Macro scaling (approx)
    x_macro_tensor = torch.FloatTensor(x_macro).unsqueeze(0)

    # Inference
    with torch.no_grad():
        prediction_scaled = model(x_seq_tensor, x_macro_tensor).item()
    
    # Inverse transform (approx)
    # We scaled input 'close' (col 0). We need to inverse transform the output.
    # Since we fit scaler on x_seq (6 cols), we need to be careful.
    # Let's assume prediction is in range of input close.
    # Reconstruct a dummy row to inverse transform
    dummy_row = np.zeros((1, 6))
    dummy_row[0, 0] = prediction_scaled
    prediction = scaler.inverse_transform(dummy_row)[0, 0]

    return {
        "prediction": float(prediction),
        "model_version": run_id,
        "drift_detected": False # Placeholder, drift is checked in training loop
    }

@app.get("/health")
def health():
    return {"status": "ok"}
