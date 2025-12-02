import time
import logging
import sys
import os
from binance_client import BinanceClient
from storage import DataStorage
from macro_client import MacroClient
from coingecko_client import CoinGeckoClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def fetch_historical_data(client, storage, lookback="1 year ago UTC"):
    """
    Fetch historical data on startup if needed.
    
    Args:
        client: BinanceClient instance
        storage: DataStorage instance
        lookback: How far back to fetch data (default: 1 year)
    """
    logger = logging.getLogger(__name__)
    
    # Check if we need to fetch historical data
    last_timestamp = storage.get_last_timestamp()
    
    if last_timestamp is None:
        logger.info(f"No existing data found. Fetching historical data from {lookback}...")
        
        # Try python-binance first
        candles = client.fetch_historical_klines(lookback=lookback)
        
        # If python-binance failed, try CCXT as fallback
        if not candles:
            logger.warning("python-binance failed, trying CCXT as fallback...")
            candles = client.fetch_historical_klines_ccxt(lookback=lookback)
        
        # If CCXT also failed, try comprehensive data fetching as last resort
        if not candles:
            logger.warning("CCXT also failed (tried multiple exchanges), trying comprehensive data fetching...")
            candles = client.fetch_historical_klines_comprehensive(lookback=lookback)
            if candles:
                # Analyze data granularity
                minute_count = sum(1 for c in candles if c.get('close_time', 0) - c['timestamp'] <= 60000)  # 1 minute or less
                daily_count = len(candles) - minute_count
                
                if minute_count > len(candles) * 0.5:  # More than 50% minute data
                    logger.warning(f"Successfully fetched high-granularity data: {minute_count} minute + {daily_count} daily candles")
                elif minute_count > 0:
                    logger.warning(f"Successfully fetched mixed-granularity data: {minute_count} minute + {daily_count} daily candles")
                else:
                    logger.warning(f"Successfully fetched daily data: {daily_count} candles")
        
        if candles:
            storage.save_candles_bulk(candles)
            logger.info(f"Historical data fetch complete. Total candles: {len(candles)}")
        else:
            logger.warning("Failed to fetch historical data from all sources (python-binance, CCXT, Yahoo Finance)")
    else:
        logger.info(f"Existing data found (last timestamp: {last_timestamp}). Skipping historical fetch.")

def fetch_historical_macro_data(macro_client, storage, lookback_days=365):
    """
    Fetch historical macro data on startup if needed.
    
    Args:
        macro_client: MacroClient instance
        storage: DataStorage instance
        lookback_days: How many days of historical data to fetch (default: 365)
    """
    logger = logging.getLogger(__name__)
    
    # Check if we need to fetch historical macro data
    last_macro_timestamp = storage.get_last_macro_timestamp()
    
    if last_macro_timestamp is None:
        logger.info(f"No existing macro data found. Fetching historical macro data for {lookback_days} days...")
        macro_records = macro_client.fetch_historical_macro(lookback_days=lookback_days)
        if macro_records:
            storage.save_macro_bulk(macro_records)
            logger.info(f"Historical macro data fetch complete. Total records: {len(macro_records)}")
        else:
            logger.warning("Failed to fetch historical macro data")
    else:
        logger.info(f"Existing macro data found (last timestamp: {last_macro_timestamp}). Skipping historical fetch.")

def main():
    logger = logging.getLogger(__name__)
    logger.info("Starting SOFARE-AI Ingestion Service")

    storage = DataStorage()
    
    def on_candle_received(candle):
        storage.save_candle(candle)

    # Choose data source based on environment variable
    data_source = os.getenv('DATA_SOURCE', 'binance')  # Options: binance, coingecko
    historical_lookback = os.getenv('HISTORICAL_LOOKBACK', '10 years ago UTC')

    if data_source == 'coingecko':
        logger.info("Using CoinGecko API for price data")
        # Start CoinGecko Client
        client = CoinGeckoClient(symbol="bitcoin", vs_currency="usd", callback=on_candle_received, update_interval=60)
        client.start()
    else:
        logger.info("Using Binance WebSocket for price data")
        # Create Binance Client
        client = BinanceClient(interval="1m", callback=on_candle_received)
        
        # Fetch historical data on first startup
        fetch_historical_data(client, storage, lookback=historical_lookback)
        
        # Start WebSocket for real-time data
        client.start()

    # Start Macro Client
    macro_client = MacroClient(storage, interval_seconds=60)  # Update every minute
    
    # Fetch historical macro data on first startup
    fetch_historical_macro_data(macro_client, storage, lookback_days=365)  # 1 year instead of 10
    
    # Start real-time macro updates
    macro_client.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping service...")

if __name__ == "__main__":
    main()
