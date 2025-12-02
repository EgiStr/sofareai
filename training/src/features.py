import pandas as pd
import numpy as np
import logging
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

logger = logging.getLogger(__name__)

def add_technical_indicators(df):
    """
    Adds comprehensive technical indicators to the dataframe.
    """
    df = df.copy()
    initial_len = len(df)
    logger.info(f"Starting feature engineering with {initial_len} records")
    
    # Ensure numeric types
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Log NaN counts before indicators
    nan_counts = df[numeric_cols].isna().sum()
    if nan_counts.any():
        logger.warning(f"NaN in numeric columns before indicators: {nan_counts.to_dict()}")

    logger.info("Computing momentum indicators (RSI, Stochastic)...")
    # Momentum Indicators
    # RSI
    rsi_indicator = RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi_indicator.rsi()

    # Stochastic Oscillator
    stoch_indicator = StochasticOscillator(high=df["high"], low=df["low"], close=df["close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch_indicator.stoch()
    df["stoch_d"] = stoch_indicator.stoch_signal()

    logger.info("Computing trend indicators (MACD, SMA, EMA)...")
    # Trend Indicators
    # MACD
    macd_indicator = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_indicator.macd()
    df["macd_signal"] = macd_indicator.macd_signal()
    df["macd_diff"] = macd_indicator.macd_diff()

    # Moving Averages
    df["sma_20"] = SMAIndicator(close=df["close"], window=20).sma_indicator()
    df["ema_12"] = EMAIndicator(close=df["close"], window=12).ema_indicator()
    df["ema_26"] = EMAIndicator(close=df["close"], window=26).ema_indicator()

    logger.info("Computing volatility indicators (Bollinger Bands, ATR)...")
    # Volatility Indicators
    # Bollinger Bands
    bb_indicator = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb_indicator.bollinger_hband()
    df["bb_lower"] = bb_indicator.bollinger_lband()
    df["bb_middle"] = bb_indicator.bollinger_mavg()

    # ATR
    atr_indicator = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr"] = atr_indicator.average_true_range()

    logger.info("Computing volume indicators (OBV)...")
    # Volume Indicators
    # OBV
    obv_indicator = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
    df["obv"] = obv_indicator.on_balance_volume()

    # Additional Features
    # Log Return - SCALED by 100 for better model training (now represents % change)
    # This makes the target more meaningful and reduces numerical precision issues
    df["log_return"] = np.log(df["close"] / df["close"].shift(1)) * 100  # Percentage log return
    
    # High-Low Range
    df["hl_range"] = df["high"] - df["low"]
    
    # Rolling Volatility (20-period) - also scaled
    df["rolling_vol_20"] = df["log_return"].rolling(window=20).std()

    # ===== BEST PRACTICE: Only dropna for technical indicator columns =====
    # Don't drop rows based on macro/safe columns - they can be forward filled later
    ta_columns = ['rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_diff', 
                  'sma_20', 'ema_12', 'ema_26', 'bb_upper', 'bb_lower', 'bb_middle',
                  'atr', 'obv', 'log_return', 'hl_range', 'rolling_vol_20']
    
    # Log NaN counts per column before dropping
    nan_per_col = df[ta_columns].isna().sum()
    logger.info(f"NaN counts per TA column: {nan_per_col[nan_per_col > 0].to_dict()}")
    
    before_drop = len(df)
    # Only check NaN in technical indicators, not in the whole dataframe
    df = df.dropna(subset=ta_columns)
    after_drop = len(df)
    
    logger.info(f"Dropped {before_drop - after_drop} rows with NaN in TA columns")
    logger.info(f"Feature engineering complete: {initial_len} -> {after_drop} records ({after_drop/initial_len*100:.2f}% retained)")
    
    return df
