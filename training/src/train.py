import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
import os
import time
from sklearn.preprocessing import MinMaxScaler
from features import add_technical_indicators
from dataset import TimeSeriesDataset
from model import MultiModalLSTM

from drift import check_drift

# Configuration
DATA_PATH = "/app/data/ohlcv.csv"
MACRO_PATH = "/app/data/macro.csv"
SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 5 # Reduced for rolling training speed
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
ROLLING_WINDOW_SIZE = 2000 # Train on last 2000 candles

def load_data():
    if not os.path.exists(DATA_PATH):
        print("Data file not found. Waiting...")
        return None
    
    try:
        df = pd.read_csv(DATA_PATH)
        if len(df) < 200: 
            return None
            
        # Load Macro Data
        if os.path.exists(MACRO_PATH):
            macro_df = pd.read_csv(MACRO_PATH)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp'])
            
            df = pd.merge_asof(df.sort_values('timestamp'), 
                               macro_df.sort_values('timestamp'), 
                               on='timestamp', 
                               direction='backward')
            
            df[['fed_funds_rate', 'gold_price', 'dxy']] = df[['fed_funds_rate', 'gold_price', 'dxy']].ffill().fillna(0)
        else:
            print("Macro file not found, using zeros.")
            df['fed_funds_rate'] = 0
            df['gold_price'] = 0
            df['dxy'] = 0

        return df
    except Exception as e:
        print(f"Error reading data: {e}")
        return None

def train():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("SOFARE-AI-Phase3")
    
    # Keep track of reference data for drift detection (simple approach: use previous window)
    reference_close_prices = None

    while True:
        print("Checking for data...")
        df = load_data()
        if df is None:
            time.sleep(10)
            continue

        print(f"Total data points: {len(df)}")
        
        # Rolling Window
        if len(df) > ROLLING_WINDOW_SIZE:
            df = df.iloc[-ROLLING_WINDOW_SIZE:]
            print(f"Applied rolling window. Using last {ROLLING_WINDOW_SIZE} points.")

        print("Starting training pipeline...")
        
        # Feature Engineering
        df = add_technical_indicators(df)
        
        # Drift Detection
        current_close_prices = df['close'].values
        drift_detected = False
        drift_info = {}
        
        if reference_close_prices is not None:
            # Check drift against previous window
            # Ensure we compare similar sample sizes if possible, or just distribution
            drift_detected, drift_info = check_drift(reference_close_prices, current_close_prices)
            print(f"Drift Check: {drift_detected}, Details: {drift_info}")
        
        # Update reference for next cycle
        reference_close_prices = current_close_prices

        # Prepare features
        feature_cols = ['close', 'volume', 'rsi', 'macd', 'macd_signal', 'macd_diff']
        macro_cols = ['fed_funds_rate', 'gold_price', 'dxy']
        target_col = 'close'
        
        data_seq = df[feature_cols].values
        data_macro = df[macro_cols].values
        target = df[target_col].values
        
        # Scaling
        scaler_seq = MinMaxScaler()
        scaler_macro = MinMaxScaler()
        scaler_target = MinMaxScaler()
        
        data_seq_scaled = scaler_seq.fit_transform(data_seq)
        data_macro_scaled = scaler_macro.fit_transform(data_macro)
        target_scaled = scaler_target.fit_transform(target.reshape(-1, 1))
        
        # Dataset
        dataset = TimeSeriesDataset(data_seq_scaled, data_macro_scaled, target_scaled, SEQUENCE_LENGTH)
        
        # Split into train/validation (simple 80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Model setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultiModalLSTM(
            input_size=len(feature_cols), 
            macro_size=len(macro_cols),
            hidden_size=HIDDEN_SIZE, 
            num_layers=NUM_LAYERS
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        with mlflow.start_run():
            # Set run description
            mlflow.set_tag("mlflow.runName", f"SOFARE-AI Training - {pd.Timestamp.now()}")
            mlflow.set_tag("model_architecture", "MultiModalLSTM: LSTM for time-series + Dense for macro indicators")
            mlflow.set_tag("pipeline_stage", "training")
            mlflow.set_tag("data_sources", "Binance OHLCV + FRED Macro (Fed Funds, Gold, DXY)")
            
            # Log parameters
            mlflow.log_param("model_type", "MultiModalLSTM")
            mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
            mlflow.log_param("rolling_window", ROLLING_WINDOW_SIZE)
            mlflow.log_param("drift_detected", drift_detected)
            if drift_info:
                mlflow.log_metric("drift_p_value", drift_info.get("p_value", 1.0))
            
            # Log dataset info
            mlflow.log_param("total_data_points", len(df))
            mlflow.log_param("features", "OHLCV + TA (RSI, MACD) + Macro (Fed, Gold, DXY)")
            
            for epoch in range(EPOCHS):
                model.train()
                epoch_loss = 0
                for i, (x_seq, x_macro, y) in enumerate(train_loader):
                    x_seq, x_macro, y = x_seq.to(device), x_macro.to(device), y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(x_seq, x_macro)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_train_loss = epoch_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.6f}")
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                
                # Validation
                model.eval()
                val_loss = 0
                predictions = []
                targets = []
                with torch.no_grad():
                    for x_seq, x_macro, y in val_loader:
                        x_seq, x_macro, y = x_seq.to(device), x_macro.to(device), y.to(device)
                        outputs = model(x_seq, x_macro)
                        loss = criterion(outputs, y)
                        val_loss += loss.item()
                        predictions.extend(outputs.cpu().numpy())
                        targets.extend(y.cpu().numpy())
                
                avg_val_loss = val_loss / len(val_loader)
                print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {avg_val_loss:.6f}")
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                
                # Additional metrics
                predictions = np.array(predictions).flatten()
                targets = np.array(targets).flatten()
                mae = np.mean(np.abs(predictions - targets))
                rmse = np.sqrt(np.mean((predictions - targets)**2))
                r2 = 1 - np.sum((predictions - targets)**2) / np.sum((targets - np.mean(targets))**2) if np.var(targets) > 0 else 0
                
                mlflow.log_metric("mae", mae, step=epoch)
                mlflow.log_metric("rmse", rmse, step=epoch)
                mlflow.log_metric("r2_score", r2, step=epoch)
            
            # Log artifacts
            import pickle
            with open("scalers.pkl", "wb") as f:
                pickle.dump({"seq": scaler_seq, "macro": scaler_macro, "target": scaler_target}, f)
            mlflow.log_artifact("scalers.pkl", "preprocessing")
            
            feature_info = {
                "sequence_features": feature_cols,
                "macro_features": macro_cols,
                "target": target_col,
                "sequence_length": SEQUENCE_LENGTH,
                "rolling_window": ROLLING_WINDOW_SIZE
            }
            with open("feature_info.json", "w") as f:
                import json
                json.dump(feature_info, f)
            mlflow.log_artifact("feature_info.json", "metadata")
            
            # Log model to MLflow
            import os
            model_path = "model"
            if os.path.exists(model_path):
                import shutil
                shutil.rmtree(model_path)
            mlflow.pytorch.save_model(model, model_path, pip_requirements=["torch", "pandas", "numpy", "scikit-learn"])
            mlflow.log_artifact(model_path, "model")
            print("Model saved and logged to MLflow successfully.")
            
            print("Training finished. Model saved.")
        
        print("Sleeping for 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    train()
