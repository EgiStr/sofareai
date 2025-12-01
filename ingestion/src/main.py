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

def main():
    logger = logging.getLogger(__name__)
    logger.info("Starting SOFARE-AI Ingestion Service")

    storage = DataStorage()
    
    def on_candle_received(candle):
        storage.save_candle(candle)

    # Choose data source based on environment variable
    data_source = os.getenv('DATA_SOURCE', 'binance')  # Options: binance, coingecko

    if data_source == 'coingecko':
        logger.info("Using CoinGecko API for price data")
        # Start CoinGecko Client
        client = CoinGeckoClient(symbol="bitcoin", vs_currency="usd", callback=on_candle_received, update_interval=60)
        client.start()
    else:
        logger.info("Using Binance WebSocket for price data")
        # Start Binance Client (use 1m interval for stability)
        client = BinanceClient(interval="1m", callback=on_candle_received)
        client.start()

    # Start Macro Client
    macro_client = MacroClient(storage, interval_seconds=60)  # Update every minute
    macro_client.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping service...")

if __name__ == "__main__":
    main()
