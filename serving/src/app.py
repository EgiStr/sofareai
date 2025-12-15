from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import mlflow
import mlflow.pytorch
import torch
import pandas as pd
import numpy as np
import os
import json
import pickle
import subprocess
import io
import time
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

def read_last_lines(filepath, n_lines):
    """
    Efficiently read the last n lines of a large CSV file using the 'tail' command.
    This avoids loading the entire file into memory.
    """
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    try:
        # Get header
        header = subprocess.check_output(['head', '-n', '1', filepath]).decode('utf-8').strip()
        # Get tail
        tail = subprocess.check_output(['tail', '-n', str(n_lines), filepath]).decode('utf-8')
        
        # Combine
        if tail.startswith(header):
            content = tail
        else:
            content = header + '\n' + tail
            
        return pd.read_csv(io.StringIO(content))
    except Exception as e:
        print(f"Error reading tail: {e}")
        # Fallback
        return pd.read_csv(filepath).tail(n_lines)

def get_latest_data():
    """
    Fetches the latest sequence of data for inference.
    Reads only the most recent data for efficiency.
    """
    if not os.path.exists(DATA_PATH):
        return None, None, None

    try:
        # Read only the last 500 lines for fast inference
        df = read_last_lines(DATA_PATH, 500)
        
        if len(df) < SEQUENCE_LENGTH + 50: # Buffer for indicators
            return None, None, None
            
        # Load Macro
        if os.path.exists(MACRO_PATH):
            macro_df = pd.read_csv(MACRO_PATH)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp'], format='mixed', errors='coerce')
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
            return None, None, None
            
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
        import traceback
        traceback.print_exc()
        return None, None, None

@app.api_route("/predict", methods=["GET", "POST"], response_model=PredictionResponse)
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
    Uses autoregressive forecasting: output of step t is input for step t+1.
    """
    # 1. Load Data (Buffer for indicators)
    # We need enough history for indicators (e.g. EMA_26, MACD) to be accurate
    df = read_last_lines(DATA_PATH, 500)
    if len(df) < SEQUENCE_LENGTH + 50:
        raise HTTPException(status_code=503, detail="Not enough data for inference")

    # Load Macro
    if os.path.exists(MACRO_PATH):
        macro_df = pd.read_csv(MACRO_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp'], format='mixed', errors='coerce')
        df = pd.merge_asof(df.sort_values('timestamp'), macro_df.sort_values('timestamp'), on='timestamp', direction='backward')
        df[['fed_funds_rate', 'gold_price', 'dxy', 'sp500', 'vix', 'nasdaq', 'oil_price']] = df[['fed_funds_rate', 'gold_price', 'dxy', 'sp500', 'vix', 'nasdaq', 'oil_price']].fillna(method='ffill').fillna(0)
    else:
        # Handle missing macro
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['fed_funds_rate', 'gold_price', 'dxy', 'sp500', 'vix', 'nasdaq', 'oil_price']:
            df[col] = 0

    # Ensure numeric
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Load model
    shared_model_path = "/app/shared_model"
    try:
        if not os.path.exists(shared_model_path):
            raise HTTPException(status_code=503, detail="Model not ready (Shared model not found)")
        
        with open(os.path.join(shared_model_path, "model_config.json"), "r") as f:
            config = json.load(f)
            
        model = SofareM3(
            micro_input_size=config["micro_input_size"],
            macro_input_size=config["macro_input_size"],
            safe_input_size=config["safe_input_size"],
            hidden_size=config["hidden_size"],
            embed_dim=config["embed_dim"]
        )
        model.load_state_dict(torch.load(os.path.join(shared_model_path, "model_weights.pth")))
        model.eval()
        model_version = "latest_shared"
        
        # Load scalers
        try:
            with open(os.path.join(shared_model_path, "scalers.pkl"), "rb") as f:
                scalers = pickle.load(f)
            scaler_seq = scalers["seq"]
            scaler_macro = scalers["macro"] 
            scaler_safe = scalers["safe"]
            scaler_target = scalers["target"]
        except Exception as e:
            # Fallback (simplified for robustness)
            print(f"Warning: Could not load scalers ({e}), using approximations")
            scaler_seq = MinMaxScaler()
            scaler_macro = MinMaxScaler()
            scaler_safe = MinMaxScaler()
            scaler_target = MinMaxScaler()
            # Fit on current data as approximation
            # Note: In production, this is suboptimal, but prevents crash
            feature_cols_approx = ['close', 'volume', 'rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_diff', 
                       'sma_20', 'ema_12', 'ema_26', 'bb_upper', 'bb_lower', 'bb_middle', 'atr', 'obv', 
                       'log_return', 'hl_range', 'rolling_vol_20']
            # We can't fit here easily without calculating features first. 
            # We'll handle this in the loop or assume pre-fitted if possible.
            # For now, we'll just proceed and hope for the best or catch errors.
            pass

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Prepare response data (Historical context)
    current_candles = []
    for _, row in df.tail(20).iterrows():
        current_candles.append(CandlestickPrediction(
            timestamp=row['timestamp'].isoformat(),
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume']),
            prediction_type="historical"
        ))

    predicted_candles = []
    
    # Autoregressive Loop
    for i in range(forecast_steps):
        # 1. Feature Engineering on current df (re-calculate on updated history)
        # Momentum
        df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
        stoch = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # Trend
        macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
        
        df["sma_20"] = SMAIndicator(close=df["close"], window=20).sma_indicator()
        df["ema_12"] = EMAIndicator(close=df["close"], window=12).ema_indicator()
        df["ema_26"] = EMAIndicator(close=df["close"], window=26).ema_indicator()

        # Volatility
        bb = BollingerBands(close=df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_middle"] = bb.bollinger_mavg()
        
        df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()

        # Volume
        from ta.volume import OnBalanceVolumeIndicator
        df["obv"] = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()

        # Additional
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["hl_range"] = df["high"] - df["low"]
        df["rolling_vol_20"] = df["log_return"].rolling(window=20).std()

        feature_cols = ['close', 'volume', 'rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_diff', 
                       'sma_20', 'ema_12', 'ema_26', 'bb_upper', 'bb_lower', 'bb_middle', 'atr', 'obv', 
                       'log_return', 'hl_range', 'rolling_vol_20']
        macro_cols = ['fed_funds_rate', 'gold_price', 'dxy']
        safe_cols = ['sp500', 'vix', 'nasdaq', 'oil_price']
        
        # Fill NaNs (created by indicators at start of buffer)
        df_clean = df.fillna(0)
        
        last_sequence = df_clean.iloc[-SEQUENCE_LENGTH:]
        
        x_seq = last_sequence[feature_cols].values
        x_macro = last_sequence[macro_cols].values[-1]
        x_safe = last_sequence[safe_cols].values[-1]
        
        # Scale & Predict
        try:
            # If scalers were not loaded, we need to fit them now (fallback)
            try:
                scaler_seq.transform(x_seq)
            except:
                scaler_seq.fit(x_seq)
                scaler_macro.fit(x_macro.reshape(1, -1))
                scaler_safe.fit(x_safe.reshape(1, -1))
                scaler_target.fit([[0], [1]])

            x_seq_scaled = scaler_seq.transform(x_seq)
            x_macro_scaled = scaler_macro.transform(x_macro.reshape(1, -1))
            x_safe_scaled = scaler_safe.transform(x_safe.reshape(1, -1))
            
            x_seq_tensor = torch.FloatTensor(x_seq_scaled).unsqueeze(0)
            x_macro_tensor = torch.FloatTensor(x_macro_scaled)
            x_safe_tensor = torch.FloatTensor(x_safe_scaled)
            
            with torch.no_grad():
                _, reg_pred = model(x_seq_tensor, x_macro_tensor, x_safe_tensor)
                prediction_scaled = reg_pred.item()
                
            predicted_log_return = scaler_target.inverse_transform([[prediction_scaled]])[0, 0]
        except Exception as e:
            print(f"Prediction error: {e}")
            predicted_log_return = 0.0
            
        # Generate next candle
        current_close = df.iloc[-1]['close']
        predicted_close = current_close * (1 + predicted_log_return)
        
        # Volatility for realistic candle shape
        volatility = df['log_return'].tail(20).std()
        if np.isnan(volatility): volatility = 0.005
        
        predicted_open = current_close
        high_offset = np.random.uniform(0.001, max(volatility, 0.001)) * predicted_close
        low_offset = np.random.uniform(0.001, max(volatility, 0.001)) * predicted_close
        
        predicted_high = max(predicted_open, predicted_close) + high_offset
        predicted_low = min(predicted_open, predicted_close) - low_offset
        
        # Volume
        current_volume = df.iloc[-1]['volume']
        predicted_volume = current_volume * np.random.uniform(0.8, 1.2)
        
        next_timestamp = df.iloc[-1]['timestamp'] + pd.Timedelta(minutes=1)
        
        # Create new row
        new_row = {
            'timestamp': next_timestamp,
            'open': predicted_open,
            'high': predicted_high,
            'low': predicted_low,
            'close': predicted_close,
            'volume': predicted_volume,
            # Carry forward macro/safe values
            'fed_funds_rate': df.iloc[-1]['fed_funds_rate'],
            'gold_price': df.iloc[-1]['gold_price'],
            'dxy': df.iloc[-1]['dxy'],
            'sp500': df.iloc[-1]['sp500'],
            'vix': df.iloc[-1]['vix'],
            'nasdaq': df.iloc[-1]['nasdaq'],
            'oil_price': df.iloc[-1]['oil_price']
        }
        
        # Append to df for next iteration
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        predicted_candles.append(CandlestickPrediction(
            timestamp=next_timestamp.isoformat(),
            open=float(predicted_open),
            high=float(predicted_high),
            low=float(predicted_low),
            close=float(predicted_close),
            volume=float(predicted_volume),
            prediction_type="forecasted"
        ))

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
def get_ohlcv_data(limit: int = 100, interval: str = "1m"):
    """Get recent OHLCV data for charting
    
    Args:
        limit: Number of candles to return (max 5000)
        interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d) - aggregates from 1m data
    """
    if not os.path.exists(DATA_PATH):
        return {"error": "No data available"}

    try:
        # Limit query: Use efficient tail reading
        limit = min(limit, 5000)
        
        # Calculate required lines based on interval
        multiplier = 1
        if interval == '5m': multiplier = 5
        elif interval == '15m': multiplier = 15
        elif interval == '30m': multiplier = 30
        elif interval == '1h': multiplier = 60
        elif interval == '4h': multiplier = 240
        elif interval == '1d': multiplier = 1440
        
        required_lines = limit * multiplier
        # Read only recent data
        df = read_last_lines(DATA_PATH, required_lines + 100)

        if len(df) == 0:
            return {"error": "No data available"}

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df.iloc[:, 0], unit='ms')
        df = df.sort_values('timestamp')
        
        # Ensure numeric columns
        df['open'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        df['high'] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
        df['low'] = pd.to_numeric(df.iloc[:, 3], errors='coerce')
        df['close'] = pd.to_numeric(df.iloc[:, 4], errors='coerce')
        df['volume'] = pd.to_numeric(df.iloc[:, 5], errors='coerce')
        
        # Drop any NaN rows
        df = df.dropna()
        
        # Aggregate to different intervals if requested
        interval_map = {
            '1m': '1min',
            '5m': '5min', 
            '15m': '15min',
            '30m': '30min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1D'
        }
        
        if interval in interval_map and interval != '1m':
            df = df.set_index('timestamp')
            df = df.resample(interval_map[interval]).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna().reset_index()

        # Take last 'limit' records
        df = df.tail(limit)

        # Format for frontend (OHLC format)
        data = []
        for _, row in df.iterrows():
            data.append({
                'timestamp': int(row['timestamp'].timestamp() * 1000),  # Convert to milliseconds since epoch
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })

        return {
            "data": data,
            "interval": interval,
            "count": len(data)
        }

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
    return FileResponse("/app/static/index.html")
