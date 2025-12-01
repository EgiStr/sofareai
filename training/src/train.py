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
import pickle
from sklearn.preprocessing import MinMaxScaler
from features import add_technical_indicators
from dataset import TimeSeriesDataset
from model import SofareM3

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
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("SOFARE-AI-Phase3")
        mlflow_enabled = True
    except:
        print("MLflow not available, skipping experiment tracking")
        mlflow_enabled = False
    
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
        feature_cols = ['close', 'volume', 'rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_diff', 
                       'sma_20', 'ema_12', 'ema_26', 'bb_upper', 'bb_lower', 'bb_middle', 'atr', 'obv', 
                       'log_return', 'hl_range', 'rolling_vol_20']
        macro_cols = ['fed_funds_rate', 'gold_price', 'dxy']
        safe_cols = ['sp500', 'vix', 'nasdaq', 'oil_price']
        target_col = 'log_return'  # Use log return as regression target
        
        # Ensure safe columns exist, fill with zeros if not
        for col in safe_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        data_seq = df[feature_cols].values
        data_macro = df[macro_cols].values
        data_safe = df[safe_cols].values
        target = df[target_col].values
        
        # Scaling
        scaler_seq = MinMaxScaler()
        scaler_macro = MinMaxScaler()
        scaler_safe = MinMaxScaler()
        scaler_target = MinMaxScaler()
        
        data_seq_scaled = scaler_seq.fit_transform(data_seq)
        data_macro_scaled = scaler_macro.fit_transform(data_macro)
        data_safe_scaled = scaler_safe.fit_transform(data_safe)
        target_scaled = scaler_target.fit_transform(target.reshape(-1, 1))
        
        # Dataset
        dataset = TimeSeriesDataset(data_seq_scaled, data_macro_scaled, data_safe_scaled, target_scaled.flatten(), SEQUENCE_LENGTH)
        
        # Split into train/validation (simple 80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Model setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SofareM3(
            micro_input_size=len(feature_cols), 
            macro_input_size=len(macro_cols),
            safe_input_size=len(safe_cols),
            hidden_size=HIDDEN_SIZE, 
            embed_dim=128
        ).to(device)
        
        # Multi-task loss
        cls_criterion = nn.CrossEntropyLoss()
        reg_criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        with mlflow.start_run():
            mlflow.set_tag("mlflow.runName", f"SOFARE-M3 Training - {pd.Timestamp.now()}")
            mlflow.set_tag("model_architecture", "SofareM3: Transformer+TCN Encoders + Attention Fusion + Multi-task Head")
            mlflow.set_tag("pipeline_stage", "training")
            mlflow.set_tag("data_sources", "Binance OHLCV + FRED Macro + Safe Haven")
            
            # Log parameters
            mlflow.log_param("model_type", "SofareM3")
            mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
            mlflow.log_param("rolling_window", ROLLING_WINDOW_SIZE)
            mlflow.log_param("drift_detected", drift_detected)
            if drift_info:
                mlflow.log_metric("drift_p_value", drift_info.get("p_value", 1.0))
            
            # Log dataset info
            mlflow.log_param("total_data_points", len(df))
            mlflow.log_param("features", f"Micro: {feature_cols}, Macro: {macro_cols}, Safe: {safe_cols}")
            
            for epoch in range(EPOCHS):
                model.train()
                epoch_cls_loss = 0
                epoch_reg_loss = 0
                for i, (x_seq, x_macro, x_safe, y_cls, y_reg) in enumerate(train_loader):
                    x_seq, x_macro, x_safe, y_cls, y_reg = x_seq.to(device), x_macro.to(device), x_safe.to(device), y_cls.to(device), y_reg.to(device)
                    
                    optimizer.zero_grad()
                    cls_pred, reg_pred = model(x_seq, x_macro, x_safe)
                    cls_loss = cls_criterion(cls_pred, y_cls)
                    reg_loss = reg_criterion(reg_pred.squeeze(), y_reg)
                    loss = 0.5 * cls_loss + 0.5 * reg_loss
                    loss.backward()
                    optimizer.step()
                    
                    epoch_cls_loss += cls_loss.item()
                    epoch_reg_loss += reg_loss.item()
                
                avg_train_cls_loss = epoch_cls_loss / len(train_loader)
                avg_train_reg_loss = epoch_reg_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{EPOCHS}, Train CLS Loss: {avg_train_cls_loss:.6f}, Train REG Loss: {avg_train_reg_loss:.6f}")
                mlflow.log_metric("train_cls_loss", avg_train_cls_loss, step=epoch)
                mlflow.log_metric("train_reg_loss", avg_train_reg_loss, step=epoch)
                
                # Validation
                model.eval()
                val_cls_loss = 0
                val_reg_loss = 0
                predictions = []
                targets_reg = []
                targets_cls = []
                with torch.no_grad():
                    for x_seq, x_macro, x_safe, y_cls, y_reg in val_loader:
                        x_seq, x_macro, x_safe, y_cls, y_reg = x_seq.to(device), x_macro.to(device), x_safe.to(device), y_cls.to(device), y_reg.to(device)
                        cls_pred, reg_pred = model(x_seq, x_macro, x_safe)
                        cls_loss = cls_criterion(cls_pred, y_cls)
                        reg_loss = reg_criterion(reg_pred.squeeze(), y_reg)
                        val_cls_loss += cls_loss.item()
                        val_reg_loss += reg_loss.item()
                        predictions.extend(reg_pred.cpu().numpy().flatten())
                        targets_reg.extend(y_reg.cpu().numpy())
                        targets_cls.extend(y_cls.cpu().numpy())
                
                avg_val_cls_loss = val_cls_loss / len(val_loader)
                avg_val_reg_loss = val_reg_loss / len(val_loader)
                print(f"Epoch {epoch+1}/{EPOCHS}, Val CLS Loss: {avg_val_cls_loss:.6f}, Val REG Loss: {avg_val_reg_loss:.6f}")
                mlflow.log_metric("val_cls_loss", avg_val_cls_loss, step=epoch)
                mlflow.log_metric("val_reg_loss", avg_val_reg_loss, step=epoch)
                
                # Additional metrics
                predictions = np.array(predictions).flatten()
                targets_reg = np.array(targets_reg).flatten()
                mae = np.mean(np.abs(predictions - targets_reg))
                rmse = np.sqrt(np.mean((predictions - targets_reg)**2))
                r2 = 1 - np.sum((predictions - targets_reg)**2) / np.sum((targets_reg - np.mean(targets_reg))**2) if np.var(targets_reg) > 0 else 0
                
                # Classification accuracy (from last batch)
                preds_cls = np.argmax(cls_pred.cpu().numpy(), axis=1)
                acc = np.mean(preds_cls == np.array(targets_cls[-len(preds_cls):]))
                
                mlflow.log_metric("mae", mae, step=epoch)
                mlflow.log_metric("rmse", rmse, step=epoch)
                mlflow.log_metric("r2_score", r2, step=epoch)
                mlflow.log_metric("cls_accuracy", acc, step=epoch)
            
            # Log artifacts
            with open("scalers.pkl", "wb") as f:
                pickle.dump({"seq": scaler_seq, "macro": scaler_macro, "safe": scaler_safe, "target": scaler_target}, f)
            mlflow.log_artifact("scalers.pkl", "preprocessing")
            
            feature_info = {
                "sequence_features": feature_cols,
                "macro_features": macro_cols,
                "safe_features": safe_cols,
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
            mlflow.pytorch.save_model(model, model_path, pip_requirements=["torch", "pandas", "numpy", "scikit-learn", "transformers"])
            print(f"Model saved to {model_path}, contents: {os.listdir(model_path)}")
            mlflow.log_artifact(model_path, "model")
            print("Model artifact logged to MLflow")
            
            # Also save model to shared location for serving
            shared_model_path = "/app/shared_model"
            os.makedirs(shared_model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(shared_model_path, "model_weights.pth"))
            # Save model config
            import json
            config = {
                "micro_input_size": len(feature_cols),
                "macro_input_size": len(macro_cols), 
                "safe_input_size": len(safe_cols),
                "hidden_size": HIDDEN_SIZE,
                "embed_dim": 128
            }
            with open(os.path.join(shared_model_path, "model_config.json"), "w") as f:
                json.dump(config, f)
            
            # Save scalers to shared volume
            with open(os.path.join(shared_model_path, "scalers.pkl"), "wb") as f:
                pickle.dump({"seq": scaler_seq, "macro": scaler_macro, "safe": scaler_safe, "target": scaler_target}, f)
            print("Model saved and logged to MLflow successfully.")
            
            print("Training finished. Model saved.")
        
        print("Sleeping for 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    train()
