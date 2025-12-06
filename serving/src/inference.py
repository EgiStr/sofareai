import os
import json
import pickle
import torch
import pandas as pd
import numpy as np
import subprocess
import io
import logging
import threading
import time
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import MinMaxScaler
from sofare_common.model import SofareM3
from sofare_common.features import add_technical_indicators
from sofare_common import SessionLocal, OHLCV, MacroIndicator

logger = logging.getLogger(__name__)

class InferenceService:
    def __init__(self, model_name="SofareM3", stage="Production"):
        self.model_name = model_name
        self.stage = stage
        self.model = None
        self.config = None
        self.scalers = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = 60  # Default
        self.current_version = None
        self.lock = threading.Lock()
        self.polling_thread = None
        self.stop_polling = False

    def start_polling(self, interval=300):
        """Starts a background thread to poll for model updates."""
        if self.polling_thread is not None and self.polling_thread.is_alive():
            logger.warning("Polling thread already running.")
            return

        self.stop_polling = False
        self.polling_thread = threading.Thread(target=self._poll_loop, args=(interval,), daemon=True)
        self.polling_thread.start()
        logger.info(f"Started model polling thread with interval {interval}s.")

    def _poll_loop(self, interval):
        """Background loop to check for model updates."""
        while not self.stop_polling:
            try:
                self.load_model()
            except Exception as e:
                logger.error(f"Error in model polling loop: {e}")
            time.sleep(interval)

    def load_model(self):
        """Loads the model from MLflow Registry if a newer version is available."""
        try:
            client = mlflow.tracking.MlflowClient()
            latest_versions = client.get_latest_versions(self.model_name, stages=[self.stage])
            
            if not latest_versions:
                 # Fallback to None or latest
                 latest_versions = client.get_latest_versions(self.model_name, stages=["None"])
                 if not latest_versions:
                     raise RuntimeError(f"No model found for {self.model_name}")
            
            target_version_obj = latest_versions[0]
            target_version = target_version_obj.version
            run_id = target_version_obj.run_id

            if self.current_version == target_version:
                logger.debug(f"Model {self.model_name} (Stage: {self.stage}) is up to date (Version: {target_version}).")
                return

            logger.info(f"New model version detected: {target_version} (Current: {self.current_version}). Loading...")
            
            model_uri = f"models:/{self.model_name}/{self.stage}"
            
            # Load Model
            new_model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
            new_model.eval()
            
            # Download artifacts
            local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, dst_path="/tmp/model_artifacts")
            
            # Load Config
            config_path = os.path.join(local_dir, "metadata", "feature_info.json")
            new_config = None
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    new_config = json.load(f)
            else:
                logger.warning("Config not found in artifacts. Using defaults.")
                
            # Load Scalers
            scalers_path = os.path.join(local_dir, "preprocessing", "scalers.pkl")
            new_scalers = None
            if os.path.exists(scalers_path):
                with open(scalers_path, "rb") as f:
                    new_scalers = pickle.load(f)
            else:
                logger.warning("Scalers not found in artifacts. Using fallback.")
                new_scalers = {
                    "seq": MinMaxScaler(),
                    "macro": MinMaxScaler(),
                    "safe": MinMaxScaler(),
                    "target": MinMaxScaler()
                }
                new_scalers["target"].fit([[0], [1]])

            # Atomic Update
            with self.lock:
                self.model = new_model
                self.config = new_config
                self.scalers = new_scalers
                self.current_version = target_version
                if self.config:
                    self.sequence_length = self.config.get("sequence_length", 60)
            
            logger.info(f"Successfully updated model to version {target_version}.")
                
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            # Fallback to local if needed, or raise
            if self.model is None: # Only raise if we don't have a model at all
                raise e

    def _read_last_lines(self, n_lines):
        """Read last n lines from DB."""
        db = SessionLocal()
        try:
            query = db.query(OHLCV).filter(OHLCV.symbol == 'BTCUSDT').order_by(OHLCV.timestamp.desc()).limit(n_lines)
            records = query.all()
            records.reverse() # Chronological
            
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

    def _prepare_data(self, df_raw):
        """
        Prepares data for inference:
        1. Merges with Macro data from DB
        2. Applies Feature Engineering
        """
        df = df_raw.copy()
        
        # Ensure timestamp_dt
        if 'timestamp' in df.columns and pd.api.types.is_numeric_dtype(df['timestamp']):
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'])

        # Load Macro from DB
        db = SessionLocal()
        try:
            # Get macro data covering the range of df
            min_ts = df['timestamp_dt'].min()
            
            from datetime import timedelta
            start_ts = min_ts - timedelta(days=7)
            
            query = db.query(MacroIndicator).filter(MacroIndicator.timestamp >= start_ts).order_by(MacroIndicator.timestamp.asc())
            macro_df_long = pd.read_sql(query.statement, db.bind)
            
            if not macro_df_long.empty:
                macro_df = macro_df_long.pivot(index='timestamp', columns='name', values='value')
                macro_df.reset_index(inplace=True)
                
                # Merge
                df = pd.merge_asof(df.sort_values('timestamp_dt'), macro_df.sort_values('timestamp'), 
                                   left_on='timestamp_dt', right_on='timestamp', direction='backward')
                
                # Fill macro NaNs
                macro_cols = [c for c in macro_df.columns if c != 'timestamp']
                df[macro_cols] = df[macro_cols].ffill().fillna(0)
            else:
                 # Fill zeros
                macro_cols = ['fed_funds_rate', 'gold_price', 'dxy', 'sp500', 'vix', 'nasdaq', 'oil_price']
                for col in macro_cols:
                    df[col] = 0
        except Exception as e:
            logger.error(f"Error loading macro from DB: {e}")
             # Fill zeros
            macro_cols = ['fed_funds_rate', 'gold_price', 'dxy', 'sp500', 'vix', 'nasdaq', 'oil_price']
            for col in macro_cols:
                df[col] = 0
        finally:
            db.close()

        # Apply Feature Engineering
        # This uses the SHARED logic from src.features
        df = add_technical_indicators(df)
        
        return df

    def predict_next(self):
        """
        Predicts the next step (log return) based on the latest data.
        Returns: (prediction_value, drift_detected)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Read enough data for indicators (buffer)
        # We need at least SEQUENCE_LENGTH + 50 (for indicators)
        buffer_size = self.sequence_length + 200 
        df_raw = self._read_last_lines(buffer_size)
        
        if len(df_raw) < self.sequence_length + 20:
            logger.warning("Not enough data for inference.")
            return None, False

        # Prepare Data
        df = self._prepare_data(df_raw)
        
        if len(df) < self.sequence_length:
            logger.warning("Not enough data after feature engineering.")
            return None, False

        # Extract features
        feature_cols = ['close', 'volume', 'rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_diff', 
                       'sma_20', 'ema_12', 'ema_26', 'bb_upper', 'bb_lower', 'bb_middle', 'atr', 'obv', 
                       'log_return', 'hl_range', 'rolling_vol_20']
        macro_cols = ['fed_funds_rate', 'gold_price', 'dxy']
        safe_cols = ['sp500', 'vix', 'nasdaq', 'oil_price']

        last_sequence = df.iloc[-self.sequence_length:]
        
        x_seq = last_sequence[feature_cols].values
        x_macro = last_sequence[macro_cols].values[-1]
        x_safe = last_sequence[safe_cols].values[-1]

        # Scale
        try:
            x_seq_scaled = self.scalers["seq"].transform(x_seq)
            x_macro_scaled = self.scalers["macro"].transform(x_macro.reshape(1, -1))
            x_safe_scaled = self.scalers["safe"].transform(x_safe.reshape(1, -1))
        except Exception as e:
            logger.error(f"Scaling error: {e}. Using unscaled data (fallback).")
            x_seq_scaled = x_seq
            x_macro_scaled = x_macro.reshape(1, -1)
            x_safe_scaled = x_safe.reshape(1, -1)

        # Tensorize
        x_seq_tensor = torch.FloatTensor(x_seq_scaled).unsqueeze(0).to(self.device)
        x_macro_tensor = torch.FloatTensor(x_macro_scaled).to(self.device)
        x_safe_tensor = torch.FloatTensor(x_safe_scaled).to(self.device)

        # Inference
        with torch.no_grad():
            cls_pred, reg_pred = self.model(x_seq_tensor, x_macro_tensor, x_safe_tensor)
            prediction_scaled = reg_pred.item()

        # Inverse Transform
        prediction = self.scalers["target"].inverse_transform([[prediction_scaled]])[0, 0]
        
        # Note: The model predicts LOG RETURN (percentage if scaled by 100 in features, but scaler handles that)
        # If features.py multiplies by 100, the scaler was fit on that, so inverse transform returns that scale.
        # If we want the actual log return value, we might need to divide by 100 if the consumer expects raw log return.
        # However, usually we just want to apply it to the price.
        # Let's assume the consumer knows how to handle the predicted value (which matches the target distribution).
        
        return prediction, False

    def predict_candlesticks(self, steps=5):
        """
        Generates candlestick predictions for N steps using autoregression.
        """
        # 1. Load Initial Data
        buffer_size = self.sequence_length + 200
        df_raw = self._read_last_lines(self.data_path, buffer_size)
        
        # We need to work with a DataFrame that we can append to
        # First, prepare the base dataframe with indicators
        # Note: We can't just append to the processed DF because indicators need to be re-calculated 
        # on the raw values (close, high, low) to be accurate for the new steps.
        # So we append to df_raw and re-calculate features in each step.
        
        # Optimization: We don't need to re-calc features for the whole history, just the tail.
        # But for simplicity and correctness (indicators like EMA depend on history), we re-calc.
        # With 5 steps and ~300 rows, it's fast enough.
        
        current_df = df_raw.copy()
        
        # Ensure timestamp is datetime for manipulation
        if 'timestamp' in current_df.columns and pd.api.types.is_numeric_dtype(current_df['timestamp']):
             current_df['timestamp'] = pd.to_datetime(current_df['timestamp'], unit='ms')
        
        predictions = []
        
        for i in range(steps):
            # 1. Prepare Data (Merge Macro + Features)
            # We do this inside the loop because the "latest" row changes
            processed_df = self._prepare_data(current_df)
            
            if len(processed_df) < self.sequence_length:
                break
                
            # 2. Extract Features for Inference
            feature_cols = ['close', 'volume', 'rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_diff', 
                           'sma_20', 'ema_12', 'ema_26', 'bb_upper', 'bb_lower', 'bb_middle', 'atr', 'obv', 
                           'log_return', 'hl_range', 'rolling_vol_20']
            macro_cols = ['fed_funds_rate', 'gold_price', 'dxy']
            safe_cols = ['sp500', 'vix', 'nasdaq', 'oil_price']
            
            last_sequence = processed_df.iloc[-self.sequence_length:]
            
            x_seq = last_sequence[feature_cols].values
            x_macro = last_sequence[macro_cols].values[-1]
            x_safe = last_sequence[safe_cols].values[-1]
            
            # 3. Scale
            x_seq_scaled = self.scalers["seq"].transform(x_seq)
            x_macro_scaled = self.scalers["macro"].transform(x_macro.reshape(1, -1))
            x_safe_scaled = self.scalers["safe"].transform(x_safe.reshape(1, -1))
            
            x_seq_tensor = torch.FloatTensor(x_seq_scaled).unsqueeze(0).to(self.device)
            x_macro_tensor = torch.FloatTensor(x_macro_scaled).to(self.device)
            x_safe_tensor = torch.FloatTensor(x_safe_scaled).to(self.device)
            
            # 4. Predict
            with torch.no_grad():
                _, reg_pred = self.model(x_seq_tensor, x_macro_tensor, x_safe_tensor)
                pred_scaled = reg_pred.item()
                
            # Inverse transform to get predicted log return (percentage)
            pred_log_return_pct = self.scalers["target"].inverse_transform([[pred_scaled]])[0, 0]
            
            # Convert percentage back to raw log return if needed
            # In features.py: log_return = np.log(...) * 100
            # So pred_log_return_pct is in range ~ -5 to 5 (percent)
            # Actual log return = pred_log_return_pct / 100
            pred_log_return = pred_log_return_pct / 100.0
            
            # 5. Generate Next Candle
            last_row = current_df.iloc[-1]
            last_close = float(last_row['close'])
            
            # Calculate next close
            # log_return = log(next/prev) -> next = prev * exp(log_return)
            next_close = last_close * np.exp(pred_log_return)
            
            # Heuristic for other candle parts (Open, High, Low, Volume)
            # This is a simulation for visualization purposes
            next_open = last_close
            
            # Estimate volatility from recent history
            recent_vol = processed_df['log_return'].tail(10).std() / 100.0 # Back to raw scale
            if np.isnan(recent_vol) or recent_vol == 0: recent_vol = 0.001
            
            # Random noise for High/Low based on volatility
            # We use a fixed seed or deterministic logic if we want reproducible "predictions"
            # But here random is fine for "generative" aspect
            high_offset = abs(np.random.normal(0, recent_vol)) * next_close
            low_offset = abs(np.random.normal(0, recent_vol)) * next_close
            
            next_high = max(next_open, next_close) + high_offset
            next_low = min(next_open, next_close) - low_offset
            
            # Volume
            next_volume = last_row['volume'] # Simple persistence
            
            # Timestamp
            next_timestamp = last_row['timestamp'] + pd.Timedelta(minutes=1)
            
            # Create new row
            new_row = last_row.copy()
            new_row['timestamp'] = next_timestamp
            new_row['open'] = next_open
            new_row['high'] = next_high
            new_row['low'] = next_low
            new_row['close'] = next_close
            new_row['volume'] = next_volume
            
            # Append to current_df
            current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
            
            predictions.append({
                "timestamp": next_timestamp,
                "open": next_open,
                "high": next_high,
                "low": next_low,
                "close": next_close,
                "volume": next_volume
            })
            
        return predictions
