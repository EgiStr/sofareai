import time
import logging
import sys
from binance_client import BinanceClient
from storage import DataStorage
from macro_client import MacroClient

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

    # Start Binance Client
    client = BinanceClient(interval="10s", callback=on_candle_received)
    client.start()

    # Start Macro Client
    macro_client = MacroClient(storage, interval_seconds=10)
    macro_client.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping service...")

if __name__ == "__main__":
    main()
