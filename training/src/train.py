import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
import os
import time
import pickle
import logging
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from features import add_technical_indicators
from dataset import TimeSeriesDataset
from model import SofareM3

# Comprehensive drift detection and version management
from drift_detector import DriftDetector, DriftSeverity, DriftType
from version_manager import (
    ModelVersionManager, 
    VersionBump, 
    create_release_from_mlflow,
    determine_version_bump
)

# Setup logging with immediate flush for Docker
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'
logger = logging.getLogger(__name__)
print("=== Training Script Started ===", flush=True)

# Configuration
DATA_PATH = "/app/data/ohlcv.csv"
MACRO_PATH = "/app/data/macro.csv"
MODEL_REGISTRY_PATH = "/app/shared_model/registry"
BEST_PARAMS_PATH = "/app/shared_model/best_params.json"

# Default hyperparameters (can be overridden by tuned params)
# Best practices for time-series regression:
# 1. Lower learning rate for stable training
# 2. More epochs with early stopping
# 3. Higher dropout for regularization
# 4. AdamW optimizer with weight decay
DEFAULT_PARAMS = {
    "sequence_length": 60,
    "batch_size": 64,           # Larger batch for stable gradients
    "epochs": 5,               # More epochs with early stopping
    "learning_rate": 0.0005,    # Lower LR for stable training
    "hidden_size": 128,         # Larger hidden size
    "embed_dim": 128,
    "num_heads": 4,
    "num_encoder_layers": 3,    # Deeper network
    "dropout": 0.2,             # Higher dropout for regularization
    "weight_decay": 1e-4,       # Stronger regularization
    "optimizer": "adamw",       # AdamW is better for transformers
    "cls_weight": 0.3,          # Lower cls weight - focus more on regression
    "patience": 10,             # Early stopping patience
    "lr_factor": 0.5,           # LR reduction factor
    "lr_patience": 5            # LR scheduler patience
}

# ===== DATA STRATEGY CONFIGURATION =====
# Best practices for time-series ML training:
# 1. USE_FULL_DATASET: Train on all available data for better generalization
# 2. TRAIN_VAL_TEST_SPLIT: Proper temporal split (no data leakage)
# 3. MIN_DATA_POINTS: Minimum data required for training
# 4. INCREMENTAL_TRAINING: Add new data incrementally

USE_FULL_DATASET = True  # Use all available data instead of rolling window
MIN_DATA_POINTS = 500    # Minimum data points required to start training
MAX_DATA_POINTS = int(os.getenv("MAX_DATA_POINTS", 100000))  # Limit for faster training, set to None for no limit

# Train/Val/Test split ratios (temporal split - no shuffle for time-series!)
TRAIN_RATIO = 0.7   # 70% for training
VAL_RATIO = 0.15    # 15% for validation  
TEST_RATIO = 0.15   # 15% for final testing

# Legacy rolling window (only used if USE_FULL_DATASET = False)
ROLLING_WINDOW_SIZE = 2000


def load_tuned_params() -> dict:
    """Load tuned hyperparameters if available."""
    if os.path.exists(BEST_PARAMS_PATH):
        try:
            with open(BEST_PARAMS_PATH, "r") as f:
                import json
                config = json.load(f)
                params = config.get("best_params", {})
                if params:
                    logger.info(f"Loaded tuned hyperparameters from {BEST_PARAMS_PATH}")
                    # Merge with defaults
                    merged = DEFAULT_PARAMS.copy()
                    merged.update(params)
                    return merged
        except Exception as e:
            logger.warning(f"Failed to load tuned params: {e}")
    
    logger.info("Using default hyperparameters")
    return DEFAULT_PARAMS.copy()


# Load hyperparameters (tuned or default)
PARAMS = load_tuned_params()
SEQUENCE_LENGTH = PARAMS.get("sequence_length", 60)
BATCH_SIZE = PARAMS.get("batch_size", 32)
EPOCHS = PARAMS.get("epochs", 5)
LEARNING_RATE = PARAMS.get("learning_rate", 0.001)
HIDDEN_SIZE = PARAMS.get("hidden_size", 64)
EMBED_DIM = PARAMS.get("embed_dim", 128)
NUM_HEADS = PARAMS.get("num_heads", 4)
NUM_ENCODER_LAYERS = PARAMS.get("num_encoder_layers", 2)
DROPOUT = PARAMS.get("dropout", 0.1)
WEIGHT_DECAY = PARAMS.get("weight_decay", 1e-5)
OPTIMIZER_NAME = PARAMS.get("optimizer", "adam")
CLS_WEIGHT = PARAMS.get("cls_weight", 0.5)

# Early stopping and LR scheduler params
PATIENCE = PARAMS.get("patience", 10)
LR_FACTOR = PARAMS.get("lr_factor", 0.5)
LR_PATIENCE = PARAMS.get("lr_patience", 5)

# Initialize version manager
version_manager = ModelVersionManager(
    model_name="sofarem3",
    registry_path=MODEL_REGISTRY_PATH,
    mlflow_tracking_uri="http://mlflow:5000"
)

# Initialize drift detector
drift_detector = DriftDetector(
    ks_threshold=0.05,
    psi_threshold=0.2,
    wasserstein_threshold=0.1
)

def load_data():
    """
    Load and merge OHLCV + Macro data with best practices:
    1. Data validation and quality checks
    2. Proper timestamp handling
    3. Missing value imputation
    4. Data integrity logging
    """
    if not os.path.exists(DATA_PATH):
        logger.warning("Data file not found. Waiting...")
        return None
    
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Best Practice: Validate minimum data requirement
        if len(df) < MIN_DATA_POINTS:
            logger.warning(f"Insufficient data: {len(df)} < {MIN_DATA_POINTS} minimum required")
            return None
        
        # Best Practice: Log data statistics
        logger.info(f"Loaded OHLCV data: {len(df)} records, date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Best Practice: Check for data quality issues
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.warning(f"Found null values in OHLCV: {null_counts[null_counts > 0].to_dict()}")
            
        # Load Macro Data
        if os.path.exists(MACRO_PATH):
            macro_df = pd.read_csv(MACRO_PATH)
            logger.info(f"Loaded Macro data: {len(macro_df)} records")
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp'], format='mixed', errors='coerce')
            
            # Best Practice: Log merge statistics
            before_merge = len(df)
            df = pd.merge_asof(df.sort_values('timestamp'), 
                               macro_df.sort_values('timestamp'), 
                               on='timestamp', 
                               direction='backward')
            logger.info(f"Merged data: {before_merge} -> {len(df)} records")
            
            # Best Practice: Forward fill without limit for macro data 
            # (macro data changes daily, safe to fill for OHLCV minute data)
            df[['fed_funds_rate', 'gold_price', 'dxy']] = df[['fed_funds_rate', 'gold_price', 'dxy']].ffill().fillna(0)
        else:
            logger.warning("Macro file not found, using zeros.")
            df['fed_funds_rate'] = 0
            df['gold_price'] = 0
            df['dxy'] = 0

        # Best Practice: Remove duplicates by timestamp
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        if before_dedup != len(df):
            logger.info(f"Removed {before_dedup - len(df)} duplicate timestamps")
        
        # Best Practice: Sort by timestamp for proper temporal ordering
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    except Exception as e:
        logger.error(f"Error reading data: {e}")
        return None


def temporal_train_val_test_split(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    Best Practice: Temporal split for time-series data.
    NEVER shuffle time-series data to avoid data leakage!
    
    Returns indices for train, validation, and test sets.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_idx = (0, train_end)
    val_idx = (train_end, val_end)
    test_idx = (val_end, n)
    
    logger.info(f"Temporal Split - Train: {train_idx[0]}:{train_idx[1]} ({train_end} samples)")
    logger.info(f"Temporal Split - Val: {val_idx[0]}:{val_idx[1]} ({val_end - train_end} samples)")
    logger.info(f"Temporal Split - Test: {test_idx[0]}:{test_idx[1]} ({n - val_end} samples)")
    
    return train_idx, val_idx, test_idx


def get_data_statistics(df, feature_cols):
    """
    Best Practice: Log comprehensive data statistics for monitoring.
    All values converted to Python native types for JSON serialization.
    """
    stats = {
        "total_records": int(len(df)),
        "date_range": {
            "start": str(df['timestamp'].min()),
            "end": str(df['timestamp'].max())
        },
        "features": {}
    }
    
    for col in feature_cols[:10]:  # Log stats for key features
        if col in df.columns:
            stats["features"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "null_pct": float(df[col].isnull().mean() * 100)
            }
    
    return stats

def train():
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("SOFARE-AI-Phase3")
        mlflow_enabled = True
    except:
        logger.warning("MLflow not available, skipping experiment tracking")
        mlflow_enabled = False
    
    # Keep track of reference data for drift detection
    reference_data = None
    previous_metrics = None

    while True:
        logger.info("Checking for data...")
        df = load_data()
        if df is None:
            time.sleep(10)
            continue

        logger.info(f"Total data points: {len(df)}")
        
        # ===== DATA STRATEGY: Full Dataset vs Rolling Window =====
        if USE_FULL_DATASET:
            # Best Practice: Use all available data for training
            if MAX_DATA_POINTS and len(df) > MAX_DATA_POINTS:
                df = df.iloc[-MAX_DATA_POINTS:]
                logger.info(f"Capped to last {MAX_DATA_POINTS} points (MAX_DATA_POINTS limit)")
            else:
                logger.info(f"Using FULL DATASET: {len(df)} data points for training")
        else:
            # Legacy: Rolling Window (only last N candles)
            if len(df) > ROLLING_WINDOW_SIZE:
                df = df.iloc[-ROLLING_WINDOW_SIZE:]
                logger.info(f"Applied rolling window. Using last {ROLLING_WINDOW_SIZE} points.")

        logger.info("Starting training pipeline...")
        
        # Feature Engineering
        logger.info(f"Before feature engineering: {len(df)} records")
        df = add_technical_indicators(df)
        logger.info(f"After feature engineering: {len(df)} records")
        
        # Prepare features
        feature_cols = ['close', 'volume', 'rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_diff', 
                       'sma_20', 'ema_12', 'ema_26', 'bb_upper', 'bb_lower', 'bb_middle', 'atr', 'obv', 
                       'log_return', 'hl_range', 'rolling_vol_20']
        macro_cols = ['fed_funds_rate', 'gold_price', 'dxy']
        safe_cols = ['sp500', 'vix', 'nasdaq', 'oil_price']
        target_col = 'log_return'
        
        # ===== BEST PRACTICE: Fill NaN for macro and safe columns =====
        # These are external data that may not be available for all timestamps
        for col in macro_cols + safe_cols:
            if col not in df.columns:
                df[col] = 0.0
            else:
                # Forward fill then backward fill, then fill with 0
                df[col] = df[col].ffill().bfill().fillna(0)
        
        logger.info(f"Data after filling NaN: {len(df)} records with no NaN in features")
        
        # ===== BEST PRACTICE: Log data statistics =====
        data_stats = get_data_statistics(df, feature_cols)
        logger.info(f"Training data statistics: {data_stats['total_records']} records from {data_stats['date_range']['start']} to {data_stats['date_range']['end']}")
        
        # ===== COMPREHENSIVE DRIFT DETECTION =====
        drift_report = None
        should_retrain = True
        
        if reference_data is not None:
            logger.info("Running comprehensive drift detection...")
            
            # Set reference data for drift detector
            drift_detector.set_reference_data(reference_data[feature_cols + macro_cols + safe_cols])
            
            # Run multivariate drift detection with multiple tests
            current_features = df[feature_cols + macro_cols + safe_cols]
            
            # KS Test for all features
            ks_results = drift_detector.detect_univariate_drift(
                current_features, 
                features=feature_cols[:5],  # Check key features
                test_type="ks"
            )
            
            # PSI for distribution shift
            psi_results = drift_detector.detect_univariate_drift(
                current_features,
                features=['close', 'volume', 'rsi'],
                test_type="psi"
            )
            
            # Multivariate drift
            multivariate_result = drift_detector.detect_multivariate_drift(current_features)
            
            # Generate drift report
            drift_report = drift_detector.get_drift_report()
            logger.info(f"Drift Status: {drift_report['status']}")
            
            for recommendation in drift_report.get('recommendations', []):
                logger.info(recommendation)
            
            # Determine if retraining is needed based on drift severity
            if drift_report['status'] in ['CRITICAL', 'WARNING']:
                should_retrain = True
                logger.info("Drift detected - triggering retraining")
            elif drift_report['status'] == 'HEALTHY' and previous_metrics:
                # Check if we should skip training (no drift, model is stable)
                logger.info("No significant drift detected - model may be stable")
                # Still retrain for continuous learning, but could skip in production
        
        # Update reference data for next cycle
        reference_data = df.copy()
        
        # Clear drift history for next cycle
        drift_detector.clear_history()
        
        data_seq = df[feature_cols].values
        data_macro = df[macro_cols].values
        data_safe = df[safe_cols].values
        target = df[target_col].values  # Original returns (for classification labels) - now in percentage
        
        # ===== BEST PRACTICE: RobustScaler for financial data with outliers =====
        # RobustScaler is less sensitive to outliers compared to StandardScaler/MinMaxScaler
        scaler_seq = RobustScaler()
        scaler_macro = RobustScaler()
        scaler_safe = RobustScaler()
        # Use RobustScaler for target - better for financial data with fat tails
        scaler_target = RobustScaler()
        
        data_seq_scaled = scaler_seq.fit_transform(data_seq)
        data_macro_scaled = scaler_macro.fit_transform(data_macro)
        data_safe_scaled = scaler_safe.fit_transform(data_safe)
        target_scaled = scaler_target.fit_transform(target.reshape(-1, 1))
        
        logger.info(f"Target stats - Mean: {target.mean():.6f}, Std: {target.std():.6f}, Min: {target.min():.6f}, Max: {target.max():.6f}")
        
        # ===== BEST PRACTICE: Temporal Train/Val/Test Split =====
        # IMPORTANT: Never shuffle time-series data to avoid data leakage!
        train_idx, val_idx, test_idx = temporal_train_val_test_split(df)
        
        # Create datasets for each split
        # IMPORTANT: Pass original_targets for correct classification labels (up/down based on actual returns)
        full_dataset = TimeSeriesDataset(
            data_seq_scaled, 
            data_macro_scaled, 
            data_safe_scaled, 
            target_scaled.flatten(), 
            SEQUENCE_LENGTH,
            original_targets=target  # Pass original returns for classification
        )
        
        # Calculate actual indices considering sequence length
        effective_length = len(full_dataset)
        train_end = int(effective_length * TRAIN_RATIO)
        val_end = int(effective_length * (TRAIN_RATIO + VAL_RATIO))
        
        # Use Subset for temporal split (no shuffling!)
        from torch.utils.data import Subset
        train_indices = list(range(0, train_end))
        val_indices = list(range(train_end, val_end))
        test_indices = list(range(val_end, effective_length))
        
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)
        
        logger.info(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Best Practice: Shuffle only training data, never validation/test
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Model setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SofareM3(
            micro_input_size=len(feature_cols), 
            macro_input_size=len(macro_cols),
            safe_input_size=len(safe_cols),
            hidden_size=HIDDEN_SIZE, 
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            dropout=DROPOUT
        ).to(device)
        
        # Multi-task loss
        cls_criterion = nn.CrossEntropyLoss()
        reg_criterion = nn.HuberLoss(delta=1.0)
        
        # Optimizer (use tuned params)
        if OPTIMIZER_NAME == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        elif OPTIMIZER_NAME == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9)
        else:  # adam
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # Learning Rate Scheduler - ReduceLROnPlateau is best for regression
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=LR_FACTOR, 
            patience=LR_PATIENCE,
            verbose=True,
            min_lr=1e-7
        )
        
        # Early stopping tracking
        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0
        best_epoch = 0
        
        with mlflow.start_run() as run:
            mlflow.set_tag("mlflow.runName", f"SOFARE-M3 Training - {pd.Timestamp.now()}")
            mlflow.set_tag("model_architecture", "SofareM3: Transformer+TCN Encoders + Attention Fusion + Multi-task Head")
            mlflow.set_tag("pipeline_stage", "training")
            mlflow.set_tag("data_sources", "Binance OHLCV + FRED Macro + Safe Haven")
            mlflow.set_tag("data_strategy", "full_dataset" if USE_FULL_DATASET else "rolling_window")
            
            # Log parameters
            mlflow.log_param("model_type", "SofareM3")
            mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
            mlflow.log_param("use_full_dataset", USE_FULL_DATASET)
            mlflow.log_param("train_ratio", TRAIN_RATIO)
            mlflow.log_param("val_ratio", VAL_RATIO)
            mlflow.log_param("test_ratio", TEST_RATIO)
            
            # Log data statistics
            mlflow.log_param("total_data_points", len(df))
            mlflow.log_param("train_samples", len(train_dataset))
            mlflow.log_param("val_samples", len(val_dataset))
            mlflow.log_param("test_samples", len(test_dataset))
            mlflow.log_param("date_range_start", str(data_stats['date_range']['start']))
            mlflow.log_param("date_range_end", str(data_stats['date_range']['end']))
            
            # Log drift detection results
            if drift_report:
                mlflow.log_param("drift_status", drift_report['status'])
                mlflow.log_metric("drift_features_affected", drift_report['summary'].get('drift_detected', 0))
                mlflow.log_metric("drift_critical_count", drift_report['summary'].get('critical', 0))
                mlflow.log_metric("drift_high_count", drift_report['summary'].get('high', 0))
            
            # Log dataset info
            mlflow.log_param("features", f"Micro: {len(feature_cols)}, Macro: {len(macro_cols)}, Safe: {len(safe_cols)}")
            
            # Log hyperparameters
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("learning_rate", LEARNING_RATE)
            mlflow.log_param("hidden_size", HIDDEN_SIZE)
            mlflow.log_param("embed_dim", EMBED_DIM)
            mlflow.log_param("num_heads", NUM_HEADS)
            mlflow.log_param("num_encoder_layers", NUM_ENCODER_LAYERS)
            mlflow.log_param("dropout", DROPOUT)
            mlflow.log_param("optimizer", OPTIMIZER_NAME)
            mlflow.log_param("weight_decay", WEIGHT_DECAY)
            mlflow.log_param("cls_weight", CLS_WEIGHT)
            mlflow.log_param("lr_scheduler", "ReduceLROnPlateau")
            mlflow.log_param("patience", PATIENCE)
            mlflow.log_param("lr_patience", LR_PATIENCE)
            
            # Training loop with detailed progress logging
            total_batches = len(train_loader)
            logger.info(f"Starting training: {EPOCHS} epochs, {total_batches} batches/epoch, {len(train_dataset)} samples")
            
            for epoch in range(EPOCHS):
                model.train()
                epoch_cls_loss = 0
                epoch_reg_loss = 0
                epoch_start = time.time()
                
                for i, (x_seq, x_macro, x_safe, y_cls, y_reg) in enumerate(train_loader):
                    x_seq, x_macro, x_safe, y_cls, y_reg = x_seq.to(device), x_macro.to(device), x_safe.to(device), y_cls.to(device), y_reg.to(device)
                    
                    optimizer.zero_grad()
                    cls_pred, reg_pred = model(x_seq, x_macro, x_safe)
                    cls_loss = cls_criterion(cls_pred, y_cls)
                    reg_loss = reg_criterion(reg_pred.squeeze(), y_reg)
                    loss = CLS_WEIGHT * cls_loss + (1 - CLS_WEIGHT) * reg_loss
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_cls_loss += cls_loss.item()
                    epoch_reg_loss += reg_loss.item()
                    
                    # Progress logging every 10% of batches
                    if (i + 1) % max(1, total_batches // 10) == 0 or i == 0:
                        progress = (i + 1) / total_batches * 100
                        logger.info(f"  Epoch {epoch+1} Progress: {progress:.0f}% ({i+1}/{total_batches} batches)")
                
                epoch_time = time.time() - epoch_start
                avg_train_cls_loss = epoch_cls_loss / len(train_loader)
                avg_train_reg_loss = epoch_reg_loss / len(train_loader)
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1}/{EPOCHS} [{epoch_time:.1f}s], Train CLS: {avg_train_cls_loss:.6f}, REG: {avg_train_reg_loss:.6f}, LR: {current_lr:.2e}")
                mlflow.log_metric("train_cls_loss", avg_train_cls_loss, step=epoch)
                mlflow.log_metric("train_reg_loss", avg_train_reg_loss, step=epoch)
                mlflow.log_metric("learning_rate", current_lr, step=epoch)
                mlflow.log_metric("epoch_time_seconds", epoch_time, step=epoch)
                
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
                logger.info(f"Epoch {epoch+1}/{EPOCHS}, Val CLS Loss: {avg_val_cls_loss:.6f}, Val REG Loss: {avg_val_reg_loss:.6f}")
                mlflow.log_metric("val_cls_loss", avg_val_cls_loss, step=epoch)
                mlflow.log_metric("val_reg_loss", avg_val_reg_loss, step=epoch)
                
                # ===== TRADING-RELEVANT METRICS =====
                # For crypto prediction, classification metrics are MORE important than R²
                
                # Regression metrics (for reference)
                predictions_scaled = np.array(predictions).flatten().reshape(-1, 1)
                targets_scaled = np.array(targets_reg).flatten().reshape(-1, 1)
                predictions_original = scaler_target.inverse_transform(predictions_scaled).flatten()
                targets_original = scaler_target.inverse_transform(targets_scaled).flatten()
                mae = np.mean(np.abs(predictions_original - targets_original))
                rmse = np.sqrt(np.mean((predictions_original - targets_original)**2))
                
                # Classification metrics - MOST IMPORTANT for trading
                all_cls_preds = []
                all_cls_targets = []
                with torch.no_grad():
                    for x_seq, x_macro, x_safe, y_cls, y_reg in val_loader:
                        x_seq, x_macro, x_safe = x_seq.to(device), x_macro.to(device), x_safe.to(device)
                        cls_pred, _ = model(x_seq, x_macro, x_safe)
                        all_cls_preds.extend(torch.argmax(cls_pred, dim=1).cpu().numpy())
                        all_cls_targets.extend(y_cls.numpy())
                
                all_cls_preds = np.array(all_cls_preds)
                all_cls_targets = np.array(all_cls_targets)
                
                # Calculate comprehensive classification metrics
                accuracy = np.mean(all_cls_preds == all_cls_targets)
                precision = precision_score(all_cls_targets, all_cls_preds, average='binary', zero_division=0)
                recall = recall_score(all_cls_targets, all_cls_preds, average='binary', zero_division=0)
                f1 = f1_score(all_cls_targets, all_cls_preds, average='binary', zero_division=0)
                
                # Directional accuracy (prediksi arah yang benar)
                # Class 1 = UP, Class 0 = DOWN
                up_correct = np.sum((all_cls_preds == 1) & (all_cls_targets == 1))
                down_correct = np.sum((all_cls_preds == 0) & (all_cls_targets == 0))
                total_up = np.sum(all_cls_targets == 1)
                total_down = np.sum(all_cls_targets == 0)
                
                up_accuracy = up_correct / total_up if total_up > 0 else 0
                down_accuracy = down_correct / total_down if total_down > 0 else 0
                
                # Log all metrics
                mlflow.log_metric("val_accuracy", accuracy, step=epoch)
                mlflow.log_metric("val_precision", precision, step=epoch)
                mlflow.log_metric("val_recall", recall, step=epoch)
                mlflow.log_metric("val_f1_score", f1, step=epoch)
                mlflow.log_metric("val_up_accuracy", up_accuracy, step=epoch)
                mlflow.log_metric("val_down_accuracy", down_accuracy, step=epoch)
                mlflow.log_metric("val_mae", mae, step=epoch)
                mlflow.log_metric("val_rmse", rmse, step=epoch)
                
                # Store for later use
                acc = accuracy
                
                # ===== Learning Rate Scheduler Step =====
                combined_val_loss = CLS_WEIGHT * avg_val_cls_loss + (1 - CLS_WEIGHT) * avg_val_reg_loss
                scheduler.step(combined_val_loss)
                
                # ===== Early Stopping Check =====
                if combined_val_loss < best_val_loss:
                    best_val_loss = combined_val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    logger.info(f"  ✓ New best model! Acc: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                    logger.info(f"    UP acc: {up_accuracy:.4f}, DOWN acc: {down_accuracy:.4f}, MAE: {mae:.6f}")
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= PATIENCE:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch}")
                        break
            
            # ===== Restore Best Model =====
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                logger.info(f"Restored best model from epoch {best_epoch}")
            
            # ===== BEST PRACTICE: Final Evaluation on Test Set =====
            logger.info("Running final evaluation on held-out test set...")
            model.eval()
            test_cls_loss = 0
            test_reg_loss = 0
            test_predictions = []
            test_targets_reg = []
            test_targets_cls = []
            test_cls_predictions = []
            
            with torch.no_grad():
                for x_seq, x_macro, x_safe, y_cls, y_reg in test_loader:
                    x_seq, x_macro, x_safe = x_seq.to(device), x_macro.to(device), x_safe.to(device)
                    y_cls, y_reg = y_cls.to(device), y_reg.to(device)
                    
                    cls_pred, reg_pred = model(x_seq, x_macro, x_safe)
                    cls_loss = cls_criterion(cls_pred, y_cls)
                    reg_loss = reg_criterion(reg_pred.squeeze(), y_reg)
                    
                    test_cls_loss += cls_loss.item()
                    test_reg_loss += reg_loss.item()
                    test_predictions.extend(reg_pred.cpu().numpy().flatten())
                    test_targets_reg.extend(y_reg.cpu().numpy())
                    test_targets_cls.extend(y_cls.cpu().numpy())
                    test_cls_predictions.extend(torch.argmax(cls_pred, dim=1).cpu().numpy())
            
            # Calculate test metrics
            if len(test_loader) > 0:
                avg_test_cls_loss = test_cls_loss / len(test_loader)
                avg_test_reg_loss = test_reg_loss / len(test_loader)
                
                # Inverse transform predictions and targets to original scale
                test_predictions_scaled = np.array(test_predictions).flatten().reshape(-1, 1)
                test_targets_scaled = np.array(test_targets_reg).flatten().reshape(-1, 1)
                
                test_predictions_original = scaler_target.inverse_transform(test_predictions_scaled).flatten()
                test_targets_original = scaler_target.inverse_transform(test_targets_scaled).flatten()
                
                test_targets_cls = np.array(test_targets_cls)
                test_cls_predictions = np.array(test_cls_predictions)
                
                # Regression metrics
                test_mae = np.mean(np.abs(test_predictions_original - test_targets_original))
                test_rmse = np.sqrt(np.mean((test_predictions_original - test_targets_original)**2))
                
                # ===== TRADING-RELEVANT CLASSIFICATION METRICS =====
                test_accuracy = np.mean(test_cls_predictions == test_targets_cls)
                test_precision = precision_score(test_targets_cls, test_cls_predictions, average='binary', zero_division=0)
                test_recall = recall_score(test_targets_cls, test_cls_predictions, average='binary', zero_division=0)
                test_f1 = f1_score(test_targets_cls, test_cls_predictions, average='binary', zero_division=0)
                
                # Directional accuracy
                test_up_correct = np.sum((test_cls_predictions == 1) & (test_targets_cls == 1))
                test_down_correct = np.sum((test_cls_predictions == 0) & (test_targets_cls == 0))
                test_total_up = np.sum(test_targets_cls == 1)
                test_total_down = np.sum(test_targets_cls == 0)
                
                test_up_accuracy = test_up_correct / test_total_up if test_total_up > 0 else 0
                test_down_accuracy = test_down_correct / test_total_down if test_total_down > 0 else 0
                
                # Confusion matrix for detailed analysis
                cm = confusion_matrix(test_targets_cls, test_cls_predictions)
                
                logger.info(f"=" * 60)
                logger.info(f"TEST SET RESULTS (Held-out {len(test_targets_cls)} samples)")
                logger.info(f"=" * 60)
                logger.info(f"  Classification Metrics (Trading-Relevant):")
                logger.info(f"    Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
                logger.info(f"    Precision: {test_precision:.4f} (when predict UP, how often correct)")
                logger.info(f"    Recall:    {test_recall:.4f} (of actual UPs, how many caught)")
                logger.info(f"    F1 Score:  {test_f1:.4f}")
                logger.info(f"  Directional Accuracy:")
                logger.info(f"    UP predictions:   {test_up_accuracy:.4f} ({test_up_correct}/{test_total_up})")
                logger.info(f"    DOWN predictions: {test_down_accuracy:.4f} ({test_down_correct}/{test_total_down})")
                logger.info(f"  Regression Metrics:")
                logger.info(f"    MAE:  {test_mae:.6f}")
                logger.info(f"    RMSE: {test_rmse:.6f}")
                logger.info(f"  Confusion Matrix: [[TN={cm[0,0]}, FP={cm[0,1]}], [FN={cm[1,0]}, TP={cm[1,1]}]]")
                logger.info(f"=" * 60)
                
                # Log test metrics to MLflow
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("test_precision", test_precision)
                mlflow.log_metric("test_recall", test_recall)
                mlflow.log_metric("test_f1_score", test_f1)
                mlflow.log_metric("test_up_accuracy", test_up_accuracy)
                mlflow.log_metric("test_down_accuracy", test_down_accuracy)
                mlflow.log_metric("test_mae", test_mae)
                mlflow.log_metric("test_rmse", test_rmse)
                mlflow.log_metric("test_cls_loss", avg_test_cls_loss)
                mlflow.log_metric("test_reg_loss", avg_test_reg_loss)
            
            # Store final metrics for version comparison (use test metrics as final evaluation)
            # Convert numpy types to Python native types for JSON serialization
            current_metrics = {
                # ===== PRIMARY METRICS (Trading-Relevant Classification) =====
                "test_accuracy": float(test_accuracy) if len(test_loader) > 0 else float(accuracy),
                "test_precision": float(test_precision) if len(test_loader) > 0 else 0.0,
                "test_recall": float(test_recall) if len(test_loader) > 0 else 0.0,
                "test_f1_score": float(test_f1) if len(test_loader) > 0 else 0.0,
                "test_up_accuracy": float(test_up_accuracy) if len(test_loader) > 0 else 0.0,
                "test_down_accuracy": float(test_down_accuracy) if len(test_loader) > 0 else 0.0,
                
                # ===== VALIDATION METRICS (for training monitoring) =====
                "val_accuracy": float(accuracy),
                "val_precision": float(precision),
                "val_recall": float(recall),
                "val_f1_score": float(f1),
                "val_cls_loss": float(avg_val_cls_loss),
                "val_reg_loss": float(avg_val_reg_loss),
                
                # ===== REGRESSION METRICS (secondary, for reference) =====
                "test_mae": float(test_mae) if len(test_loader) > 0 else float(mae),
                "test_rmse": float(test_rmse) if len(test_loader) > 0 else float(rmse),
                "val_mae": float(mae),
                "val_rmse": float(rmse),
                
                # ===== DATA INFO =====
                "total_samples": int(len(df)),
                "train_samples": int(len(train_dataset)),
                "val_samples": int(len(val_dataset)),
                "test_samples": int(len(test_dataset))
            }
            
            # Log artifacts
            with open("scalers.pkl", "wb") as f:
                pickle.dump({"seq": scaler_seq, "macro": scaler_macro, "safe": scaler_safe, "target": scaler_target}, f)
            mlflow.log_artifact("scalers.pkl", "preprocessing")
            
            feature_info = {
                "sequence_features": feature_cols,
                "macro_features": macro_cols,
                "safe_features": safe_cols,
                "target": target_col,
                "sequence_length": int(SEQUENCE_LENGTH),
                "data_strategy": "full_dataset" if USE_FULL_DATASET else "rolling_window",
                "train_ratio": float(TRAIN_RATIO),
                "val_ratio": float(VAL_RATIO),
                "test_ratio": float(TEST_RATIO),
                "total_samples": int(len(df)),
                "train_samples": int(len(train_dataset)),
                "val_samples": int(len(val_dataset)),
                "test_samples": int(len(test_dataset)),
                "data_statistics": data_stats
            }
            with open("feature_info.json", "w") as f:
                import json
                json.dump(feature_info, f)
            mlflow.log_artifact("feature_info.json", "metadata")
            
            # Log drift report if available
            if drift_report:
                with open("drift_report.json", "w") as f:
                    json.dump(drift_report, f, indent=2)
                mlflow.log_artifact("drift_report.json", "monitoring")
            
            # Log model to MLflow
            import shutil
            model_path = "model"
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            mlflow.pytorch.save_model(model, model_path, pip_requirements=["torch", "pandas", "numpy", "scikit-learn", "transformers"])
            logger.info(f"Model saved to {model_path}, contents: {os.listdir(model_path)}")
            mlflow.log_artifact(model_path, "model")
            logger.info("Model artifact logged to MLflow")
            
            # ===== SEMANTIC VERSIONING =====
            # Determine version bump based on metrics improvement
            bump_type = VersionBump.PATCH  # Default
            if previous_metrics:
                bump_type = determine_version_bump(
                    current_metrics=previous_metrics,
                    new_metrics=current_metrics,
                    architecture_changed=False,
                    features_changed=False
                )
                logger.info(f"Determined version bump: {bump_type.value}")
            
            # Create new release
            try:
                release = version_manager.create_release(
                    mlflow_run_id=run.info.run_id,
                    bump_type=bump_type,
                    metrics=current_metrics,
                    parameters={
                        "sequence_length": SEQUENCE_LENGTH,
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "learning_rate": LEARNING_RATE,
                        "hidden_size": HIDDEN_SIZE
                    },
                    changelog=f"Auto-trained model with drift status: {drift_report['status'] if drift_report else 'N/A'}"
                )
                logger.info(f"Created model release: {release.model_id} (v{release.version})")
                
                # Promote and release if TEST metrics are good (use held-out test set for decision)
                promotion_accuracy = test_accuracy if len(test_loader) > 0 else acc
                if promotion_accuracy > 0.55:  # Threshold for auto-promotion
                    version_manager.release_version(str(release.version))
                    version_manager.deploy_to_production(str(release.version))
                    version_manager.set_champion(str(release.version))
                    logger.info(f"Model v{release.version} deployed to production as champion (test_accuracy: {promotion_accuracy:.4f})")
                else:
                    logger.info(f"Model v{release.version} not promoted (test_accuracy: {promotion_accuracy:.4f} < 0.55 threshold)")
                
                # Log version info to MLflow
                mlflow.set_tag("model_version", str(release.version))
                mlflow.set_tag("release_status", release.status.value)
                
            except Exception as e:
                logger.error(f"Version management error: {e}")
            
            # Update previous metrics for next cycle
            previous_metrics = current_metrics
            
            # Also save model to shared location for serving
            shared_model_path = "/app/shared_model"
            os.makedirs(shared_model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(shared_model_path, "model_weights.pth"))
            
            # Save model config with version
            config = {
                "micro_input_size": len(feature_cols),
                "macro_input_size": len(macro_cols), 
                "safe_input_size": len(safe_cols),
                "hidden_size": HIDDEN_SIZE,
                "embed_dim": 128,
                "model_version": str(release.version) if 'release' in dir() else "1.0.0",
                "mlflow_run_id": run.info.run_id
            }
            with open(os.path.join(shared_model_path, "model_config.json"), "w") as f:
                json.dump(config, f)
            
            # Save scalers to shared volume
            with open(os.path.join(shared_model_path, "scalers.pkl"), "wb") as f:
                pickle.dump({"seq": scaler_seq, "macro": scaler_macro, "safe": scaler_safe, "target": scaler_target}, f)
            logger.info("Model saved and logged to MLflow successfully.")
            
            logger.info("Training finished. Model saved.")
        
        logger.info("Sleeping for 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    train()
