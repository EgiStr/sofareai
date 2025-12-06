import os
import json
import pickle
import torch
import pandas as pd
import numpy as np
import subprocess
import io
import logging
from sklearn.preprocessing import MinMaxScaler
from sofare_common.model import SofareM3
from sofare_common.features import add_technical_indicators

logger = logging.getLogger(__name__)

class InferenceService:
    def __init__(self, model_dir="/app/shared_model", data_path="/app/data/ohlcv.csv", macro_path="/app/data/macro.csv"):
        self.model_dir = model_dir
        self.data_path = data_path
        self.macro_path = macro_path
        self.model = None
        self.config = None
        self.scalers = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = 60  # Default, will be updated from config if available

    def load_model(self):
        """Loads the model, config, and scalers from disk."""
        logger.info(f"Loading model from {self.model_dir}...")
        
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        # 1. Load Config
        config_path = os.path.join(self.model_dir, "model_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
            
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        # Update sequence length from config if present, else default to 60
        self.sequence_length = self.config.get("sequence_length", 60)

        # 2. Initialize Model
        # Filter config args to match SofareM3 signature
        model_args = {k: v for k, v in self.config.items() if k in [
            "micro_input_size", "macro_input_size", "safe_input_size", 
            "hidden_size", "embed_dim", "dropout"
        ]}
        
        self.model = SofareM3(**model_args)
        self.model.to(self.device)

        # 3. Load Weights
        weights_path = os.path.join(self.model_dir, "model_weights.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
            
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        logger.info("Model loaded successfully.")

        # 4. Load Scalers
        scalers_path = os.path.join(self.model_dir, "scalers.pkl")
        if os.path.exists(scalers_path):
            with open(scalers_path, "rb") as f:
                self.scalers = pickle.load(f)
            logger.info("Scalers loaded successfully.")
        else:
            logger.warning("Scalers not found. Using fallback (unfitted) scalers. PREDICTIONS WILL BE INACCURATE.")
            self.scalers = {
                "seq": MinMaxScaler(),
                "macro": MinMaxScaler(),
                "safe": MinMaxScaler(),
                "target": MinMaxScaler()
            }
            # Fit dummy to prevent errors
            self.scalers["target"].fit([[0], [1]])

    def _read_last_lines(self, filepath, n_lines):
        """Efficiently read the last n lines of a large CSV file."""
        if not os.path.exists(filepath):
            return pd.DataFrame()
        
        try:
            # Get header
            header = subprocess.check_output(['head', '-n', '1', filepath]).decode('utf-8').strip()
            # Get tail
            tail = subprocess.check_output(['tail', '-n', str(n_lines), filepath]).decode('utf-8')
            
            if not tail:
                return pd.DataFrame()

            # Combine
            if tail.startswith(header):
                content = tail
            else:
                content = header + '\n' + tail
                
            return pd.read_csv(io.StringIO(content))
        except Exception as e:
            logger.error(f"Error reading tail: {e}")
            # Fallback
            return pd.read_csv(filepath).tail(n_lines)

    def _prepare_data(self, df_raw):
        """
        Prepares data for inference:
        1. Merges with Macro data
        2. Applies Feature Engineering (using shared logic)
        3. Returns processed dataframe
        """
        df = df_raw.copy()
        
        # Load Macro
        if os.path.exists(self.macro_path):
            macro_df = pd.read_csv(self.macro_path)
            
            # Ensure timestamps are datetime
            # OHLCV is usually ms timestamp
            if 'timestamp' in df.columns and pd.api.types.is_numeric_dtype(df['timestamp']):
                df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
                
            macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp'], format='mixed', errors='coerce')
            
            # Merge
            df = pd.merge_asof(df.sort_values('timestamp_dt'), macro_df.sort_values('timestamp'), 
                               left_on='timestamp_dt', right_on='timestamp', direction='backward')
            
            # Fill macro NaNs
            macro_cols = ['fed_funds_rate', 'gold_price', 'dxy', 'sp500', 'vix', 'nasdaq', 'oil_price']
            for col in macro_cols:
                if col in df.columns:
                    df[col] = df[col].ffill().fillna(0)
        else:
            # Fill zeros if no macro data
            macro_cols = ['fed_funds_rate', 'gold_price', 'dxy', 'sp500', 'vix', 'nasdaq', 'oil_price']
            for col in macro_cols:
                df[col] = 0

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
        df_raw = self._read_last_lines(self.data_path, buffer_size)
        
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
