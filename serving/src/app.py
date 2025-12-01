from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import mlflow
import mlflow.pytorch
import torch
import pandas as pd
import numpy as np
import os
import json
import pickle
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sklearn.preprocessing import MinMaxScaler
from src.model import SofareM3

app = FastAPI(title="SOFARE-AI Prediction Engine")

# Mount static files
app.mount("/static", StaticFiles(directory="/app/static"), name="static")

# Configuration
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DATA_PATH = "/app/data/ohlcv.csv"
MACRO_PATH = "/app/data/macro.csv"
SEQUENCE_LENGTH = 60

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
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
            df[['fed_funds_rate', 'gold_price', 'dxy', 'sp500', 'vix', 'nasdaq', 'oil_price']] = df[['fed_funds_rate', 'gold_price', 'dxy', 'sp500', 'vix', 'nasdaq', 'oil_price']].fillna(method='ffill').fillna(0)
        else:
            df['fed_funds_rate'] = 0
            df['gold_price'] = 0
            df['dxy'] = 0
            df['sp500'] = 0
            df['vix'] = 0
            df['nasdaq'] = 0
            df['oil_price'] = 0

        # Feature Engineering (match training features)
        # Ensure numeric types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Momentum Indicators
        rsi_indicator = RSIIndicator(close=df["close"], window=14)
        df["rsi"] = rsi_indicator.rsi()

        stoch_indicator = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3)
        df["stoch_k"] = stoch_indicator.stoch()
        df["stoch_d"] = stoch_indicator.stoch_signal()

        # Trend Indicators
        macd_indicator = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd_indicator.macd()
        df["macd_signal"] = macd_indicator.macd_signal()
        df["macd_diff"] = macd_indicator.macd_diff()

        df["sma_20"] = SMAIndicator(close=df["close"], window=20).sma_indicator()
        df["ema_12"] = EMAIndicator(close=df["close"], window=12).ema_indicator()
        df["ema_26"] = EMAIndicator(close=df["close"], window=26).ema_indicator()

        # Volatility Indicators
        bb_indicator = BollingerBands(close=df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb_indicator.bollinger_hband()
        df["bb_lower"] = bb_indicator.bollinger_lband()
        df["bb_middle"] = bb_indicator.bollinger_mavg()

        atr_indicator = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
        df["atr"] = atr_indicator.average_true_range()

        # Volume Indicators
        obv_indicator = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
        df["obv"] = obv_indicator.on_balance_volume()

        # Additional Features
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["hl_range"] = df["high"] - df["low"]
        df["rolling_vol_20"] = df["log_return"].rolling(window=20).std()

        df = df.dropna()
        
        # Get last sequence
        if len(df) < SEQUENCE_LENGTH:
            return None, None
            
        last_sequence = df.iloc[-SEQUENCE_LENGTH:]
        
        feature_cols = ['close', 'volume', 'rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_diff', 
                       'sma_20', 'ema_12', 'ema_26', 'bb_upper', 'bb_lower', 'bb_middle', 'atr', 'obv', 
                       'log_return', 'hl_range', 'rolling_vol_20']
        macro_cols = ['fed_funds_rate', 'gold_price', 'dxy']
        safe_cols = ['sp500', 'vix', 'nasdaq', 'oil_price']
        
        x_seq = last_sequence[feature_cols].values
        x_macro = last_sequence[macro_cols].values[-1] # Take last macro state
        x_safe = last_sequence[safe_cols].values[-1] # Take last safe state
        
        return x_seq, x_macro, x_safe
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None

@app.post("/predict", response_model=PredictionResponse)
async def predict():
    # Get Data first
    x_seq, x_macro, x_safe = get_latest_data()
    if x_seq is None:
        raise HTTPException(status_code=503, detail="Not enough data for inference")
    
    # Load model from shared volume (MLflow used for experiment tracking)
    shared_model_path = "/app/shared_model"
    try:
        if not os.path.exists(shared_model_path):
            raise HTTPException(status_code=503, detail="Model not ready (Shared model not found)")
        
        # Load model config
        with open(os.path.join(shared_model_path, "model_config.json"), "r") as f:
            config = json.load(f)
        
        # Create model
        model = SofareM3(
            micro_input_size=config["micro_input_size"],
            macro_input_size=config["macro_input_size"],
            safe_input_size=config["safe_input_size"],
            hidden_size=config["hidden_size"],
            embed_dim=config["embed_dim"]
        )
        
        # Load weights
        model.load_state_dict(torch.load(os.path.join(shared_model_path, "model_weights.pth")))
        model.eval()
        
        model_version = "latest_shared"
        
        # Load scalers from shared volume (with fallback)
        try:
            scalers_path = os.path.join(shared_model_path, "scalers.pkl")
            with open(scalers_path, "rb") as f:
                scalers = pickle.load(f)
            scaler_seq = scalers["seq"]
            scaler_macro = scalers["macro"] 
            scaler_safe = scalers["safe"]
            scaler_target = scalers["target"]
        except Exception as e:
            # Fallback to approximation if scalers not available
            print(f"Warning: Could not load scalers from shared volume ({e}), using approximations")
            scaler_seq = MinMaxScaler()
            scaler_macro = MinMaxScaler()
            scaler_safe = MinMaxScaler()
            scaler_target = MinMaxScaler()
            # Fit on current data as approximation
            scaler_seq.fit(x_seq)
            scaler_macro.fit(x_macro.reshape(1, -1))
            scaler_safe.fit(x_safe.reshape(1, -1))
            # For target scaler, use a simple approximation
            scaler_target.fit([[0], [1]])  # Dummy fit for inverse transform
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Preprocessing with loaded scalers
    x_seq_scaled = scaler_seq.transform(x_seq)
    
    # Reshape for model
    x_seq_tensor = torch.FloatTensor(x_seq_scaled).unsqueeze(0) # Batch size 1
    
    # Macro scaling
    x_macro_scaled = scaler_macro.transform(x_macro.reshape(1, -1))
    x_macro_tensor = torch.FloatTensor(x_macro_scaled)
    
    # Safe scaling
    x_safe_scaled = scaler_safe.transform(x_safe.reshape(1, -1))
    x_safe_tensor = torch.FloatTensor(x_safe_scaled)

    # Inference
    with torch.no_grad():
        cls_pred, reg_pred = model(x_seq_tensor, x_macro_tensor, x_safe_tensor)
        prediction_scaled = reg_pred.item()
    
    # Inverse transform using target scaler
    prediction = scaler_target.inverse_transform([[prediction_scaled]])[0, 0]

    return {
        "prediction": float(prediction),
        "model_version": model_version,
        "drift_detected": False # Placeholder, drift is checked in training loop
    }

@app.post("/predict/candlestick", response_model=CandlestickResponse)
async def predict_candlestick(forecast_steps: int = 5):
    """
    Generate candlestick predictions for the next N steps.
    Uses the model's regression output to create realistic OHLCV predictions.
    """
    # Get Data first
    x_seq, x_macro, x_safe = get_latest_data()
    if x_seq is None:
        raise HTTPException(status_code=503, detail="Not enough data for inference")
    
    # Load model from shared volume (MLflow used for experiment tracking)
    shared_model_path = "/app/shared_model"
    try:
        if not os.path.exists(shared_model_path):
            raise HTTPException(status_code=503, detail="Model not ready (Shared model not found)")
        
        # Load model config
        with open(os.path.join(shared_model_path, "model_config.json"), "r") as f:
            config = json.load(f)
        
        # Create model
        model = SofareM3(
            micro_input_size=config["micro_input_size"],
            macro_input_size=config["macro_input_size"],
            safe_input_size=config["safe_input_size"],
            hidden_size=config["hidden_size"],
            embed_dim=config["embed_dim"]
        )
        
        # Load weights
        model.load_state_dict(torch.load(os.path.join(shared_model_path, "model_weights.pth")))
        model.eval()
        
        model_version = "latest_shared"
        
        # Load scalers from shared volume (with fallback)
        try:
            scalers_path = os.path.join(shared_model_path, "scalers.pkl")
            with open(scalers_path, "rb") as f:
                scalers = pickle.load(f)
            scaler_seq = scalers["seq"]
            scaler_macro = scalers["macro"] 
            scaler_safe = scalers["safe"]
            scaler_target = scalers["target"]
        except Exception as e:
            # Fallback to approximation if scalers not available
            print(f"Warning: Could not load scalers from shared volume ({e}), using approximations")
            scaler_seq = MinMaxScaler()
            scaler_macro = MinMaxScaler()
            scaler_safe = MinMaxScaler()
            scaler_target = MinMaxScaler()
            # Fit on current data as approximation
            scaler_seq.fit(x_seq)
            scaler_macro.fit(x_macro.reshape(1, -1))
            scaler_safe.fit(x_safe.reshape(1, -1))
            # For target scaler, use a simple approximation
            scaler_target.fit([[0], [1]])  # Dummy fit for inverse transform
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Get current data for context
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df.iloc[:, 0], unit='ms')
    
    # Get last 20 actual candles for display
    recent_data = df.tail(20)
    current_candles = []
    for _, row in recent_data.iterrows():
        current_candles.append(CandlestickPrediction(
            timestamp=row['timestamp'].isoformat(),
            open=float(row.iloc[1]),
            high=float(row.iloc[2]),
            low=float(row.iloc[3]),
            close=float(row.iloc[4]),
            volume=float(row.iloc[5]),
            prediction_type="historical"
        ))
    
    # Generate predictions
    predicted_candles = []
    current_close = float(recent_data.iloc[-1]['close'])
    current_volume = float(recent_data.iloc[-1]['volume'])
    
    # Use recent volatility for realistic candle generation
    recent_returns = recent_data['close'].pct_change().dropna()
    volatility = recent_returns.std()
    
    for i in range(forecast_steps):
        # Preprocessing with loaded scalers
        x_seq_scaled = scaler_seq.transform(x_seq)
        
        # Reshape for model
        x_seq_tensor = torch.FloatTensor(x_seq_scaled).unsqueeze(0) # Batch size 1
        
        # Macro scaling
        x_macro_scaled = scaler_macro.transform(x_macro.reshape(1, -1))
        x_macro_tensor = torch.FloatTensor(x_macro_scaled)
        
        # Safe scaling
        x_safe_scaled = scaler_safe.transform(x_safe.reshape(1, -1))
        x_safe_tensor = torch.FloatTensor(x_safe_scaled)

        # Inference
        with torch.no_grad():
            cls_pred, reg_pred = model(x_seq_tensor, x_macro_tensor, x_safe_tensor)
            prediction_scaled = reg_pred.item()
        
        # Inverse transform using target scaler to get log return prediction
        predicted_log_return = scaler_target.inverse_transform([[prediction_scaled]])[0, 0]
        
        # Convert log return to price prediction
        predicted_close = current_close * (1 + predicted_log_return)
        
        # Generate realistic OHLCV from the predicted close price
        # Use current close as reference for open price
        predicted_open = current_close
        
        # Generate high/low based on volatility and predicted close
        price_range = abs(predicted_close - predicted_open)
        volatility_factor = max(volatility, 0.005)  # Minimum 0.5% volatility
        
        # High is typically above the max of open/close
        high_offset = np.random.uniform(0.001, volatility_factor) * abs(predicted_close)
        predicted_high = max(predicted_open, predicted_close) + high_offset
        
        # Low is typically below the min of open/close  
        low_offset = np.random.uniform(0.001, volatility_factor) * abs(predicted_close)
        predicted_low = min(predicted_open, predicted_close) - low_offset
        
        # Volume prediction (simplified - could be enhanced with separate model)
        volume_factor = np.random.uniform(0.8, 1.2)  # 80-120% of current volume
        predicted_volume = current_volume * volume_factor
        
        # Create timestamp for next candle (assuming 1-minute intervals)
        next_timestamp = recent_data.iloc[-1]['timestamp'] + pd.Timedelta(minutes=i+1)
        
        predicted_candles.append(CandlestickPrediction(
            timestamp=next_timestamp.isoformat(),
            open=float(predicted_open),
            high=float(predicted_high),
            low=float(predicted_low),
            close=float(predicted_close),
            volume=float(predicted_volume),
            prediction_type="forecasted"
        ))
        
        # Update current_close for next iteration
        current_close = predicted_close
        current_volume = predicted_volume

    return CandlestickResponse(
        current_data=current_candles,
        predicted_candles=predicted_candles,
        model_version=model_version,
        drift_detected=False
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/ohlcv")
def get_ohlcv_data(limit: int = 100):
    """Get recent OHLCV data for charting"""
    if not os.path.exists(DATA_PATH):
        return {"error": "No data available"}

    try:
        df = pd.read_csv(DATA_PATH)
        if len(df) == 0:
            return {"error": "No data available"}

        # Take last 'limit' records
        df = df.tail(limit)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df.iloc[:, 0], unit='ms')

        # Format for frontend (OHLC format)
        data = []
        for _, row in df.iterrows():
            data.append({
                'timestamp': row['timestamp'].isoformat(),
                'open': float(row.iloc[1]),
                'high': float(row.iloc[2]),
                'low': float(row.iloc[3]),
                'close': float(row.iloc[4]),
                'volume': float(row.iloc[5])
            })

        return {"data": data}

    except Exception as e:
        return {"error": str(e)}

@app.get("/api/macro")
def get_macro_data():
    """Get latest macro indicators"""
    if not os.path.exists(MACRO_PATH):
        return {"error": "No macro data available"}

    try:
        df = pd.read_csv(MACRO_PATH)
        if len(df) == 0:
            return {"error": "No macro data available"}

        # Get latest values
        latest = df.iloc[-1]
        
        # Handle NaN values properly
        def safe_float(value):
            if pd.isna(value) or np.isnan(value):
                return None
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
    """Serve the main dashboard"""
    return {"message": "SOFARE-AI Dashboard", "docs": "/docs", "dashboard": "/static/index.html"}
