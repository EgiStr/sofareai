#!/usr/bin/env python3
"""
Hyperparameter Tuning Runner for SOFARE-AI.

This script runs hyperparameter optimization using Optuna with MLflow tracking.
Can be run as a separate job before training or integrated into CI/CD pipeline.

Usage:
    python run_tuning.py --n-trials 50 --timeout 3600
    python run_tuning.py --load-best  # Use best params for training
"""

import argparse
import os
import sys
import logging
import pandas as pd
from datetime import datetime

from hyperparameter_tuning import (
    HyperparameterTuner,
    TuningConfig,
    create_default_tuning_config,
    load_best_params
)
from features import add_technical_indicators

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = os.getenv("DATA_PATH", "/app/data/ohlcv.csv")
MACRO_PATH = os.getenv("MACRO_PATH", "/app/data/macro.csv")
BEST_PARAMS_PATH = os.getenv("BEST_PARAMS_PATH", "/app/shared_model/best_params.json")


def load_and_prepare_data() -> pd.DataFrame:
    """Load and prepare data for tuning."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} data points")
    
    # Load Macro Data
    if os.path.exists(MACRO_PATH):
        macro_df = pd.read_csv(MACRO_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        macro_df['timestamp'] = pd.to_datetime(
            macro_df['timestamp'], format='mixed', errors='coerce'
        )
        
        df = pd.merge_asof(
            df.sort_values('timestamp'),
            macro_df.sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        
        df[['fed_funds_rate', 'gold_price', 'dxy']] = \
            df[['fed_funds_rate', 'gold_price', 'dxy']].ffill().fillna(0)
    else:
        logger.warning("Macro file not found, using zeros")
        df['fed_funds_rate'] = 0
        df['gold_price'] = 0
        df['dxy'] = 0
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Ensure safe haven columns exist
    safe_cols = ['sp500', 'vix', 'nasdaq', 'oil_price']
    for col in safe_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    # Drop NaN values
    df = df.dropna()
    
    logger.info(f"Prepared {len(df)} data points after feature engineering")
    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.8):
    """Split data into train and validation sets."""
    split_idx = int(len(df) * train_ratio)
    train_data = df.iloc[:split_idx].reset_index(drop=True)
    val_data = df.iloc[split_idx:].reset_index(drop=True)
    
    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    return train_data, val_data


def run_tuning(args):
    """Run hyperparameter tuning."""
    logger.info("=" * 60)
    logger.info("SOFARE-AI Hyperparameter Tuning")
    logger.info("=" * 60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    train_data, val_data = split_data(df, train_ratio=0.8)
    
    # Define feature columns
    feature_cols = [
        'close', 'volume', 'rsi', 'stoch_k', 'stoch_d', 
        'macd', 'macd_signal', 'macd_diff',
        'sma_20', 'ema_12', 'ema_26', 
        'bb_upper', 'bb_lower', 'bb_middle',
        'atr', 'obv', 'log_return', 'hl_range', 'rolling_vol_20'
    ]
    macro_cols = ['fed_funds_rate', 'gold_price', 'dxy']
    safe_cols = ['sp500', 'vix', 'nasdaq', 'oil_price']
    
    # Create tuning configuration
    config = TuningConfig(
        study_name=f"sofare-tuning-{datetime.now().strftime('%Y%m%d-%H%M')}",
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        direction="maximize",
        
        # Search space
        sequence_length_range=(30, 90),
        batch_size_options=[32, 64],
        hidden_size_range=(64, 192),
        embed_dim_options=[64, 128],
        num_heads_options=[4, 8],
        num_layers_range=(1, 3),
        dropout_range=(0.1, 0.4),
        learning_rate_range=(1e-4, 1e-2),
        epochs_range=(5, 10),
        weight_decay_range=(1e-5, 1e-3),
        optimizer_options=["adam", "adamw"],
        
        # Pruning
        use_pruning=True,
        pruner_type="hyperband",
        
        # MLflow
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        experiment_name="SOFARE-AI-Tuning"
    )
    
    # Create tuner
    tuner = HyperparameterTuner(
        config=config,
        train_data=train_data,
        val_data=val_data,
        feature_cols=feature_cols,
        macro_cols=macro_cols,
        safe_cols=safe_cols
    )
    
    # Run tuning
    results = tuner.run()
    
    # Print results
    logger.info("=" * 60)
    logger.info("TUNING RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best Accuracy: {results['best_value']:.4f}")
    logger.info(f"Total Trials: {results['n_trials']}")
    logger.info(f"Completed: {results['n_complete']}")
    logger.info(f"Pruned: {results['n_pruned']}")
    logger.info(f"Best Parameters:")
    for key, value in results['best_params'].items():
        logger.info(f"  {key}: {value}")
    
    # Save best parameters
    tuner.save_best_config(BEST_PARAMS_PATH)
    logger.info(f"Best parameters saved to: {BEST_PARAMS_PATH}")
    
    # Get parameter importance
    importance = tuner.get_param_importance()
    if importance:
        logger.info("Parameter Importance:")
        for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {param}: {score:.4f}")
    
    # Save optimization history
    history_path = BEST_PARAMS_PATH.replace('.json', '_history.csv')
    history = tuner.get_optimization_history()
    history.to_csv(history_path, index=False)
    logger.info(f"Optimization history saved to: {history_path}")
    
    return results


def show_best_params(args):
    """Show current best parameters."""
    if not os.path.exists(BEST_PARAMS_PATH):
        logger.error(f"No best params file found at {BEST_PARAMS_PATH}")
        logger.info("Run tuning first: python run_tuning.py --n-trials 50")
        return
    
    params = load_best_params(BEST_PARAMS_PATH)
    
    logger.info("=" * 60)
    logger.info("CURRENT BEST HYPERPARAMETERS")
    logger.info("=" * 60)
    for key, value in params.items():
        logger.info(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="SOFARE-AI Hyperparameter Tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Run hyperparameter tuning")
    tune_parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of trials to run"
    )
    tune_parser.add_argument(
        "--timeout", type=int, default=3600,
        help="Timeout in seconds"
    )
    tune_parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Number of parallel jobs"
    )
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show best parameters")
    
    args = parser.parse_args()
    
    if args.command == "tune":
        run_tuning(args)
    elif args.command == "show":
        show_best_params(args)
    else:
        # Default: run tuning with default args
        args.n_trials = 50
        args.timeout = 3600
        args.n_jobs = 1
        run_tuning(args)


if __name__ == "__main__":
    main()
