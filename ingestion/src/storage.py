import pandas as pd
import os
import logging

class DataStorage:
    def __init__(self, data_dir="/app/data", filename="ohlcv.csv"):
        self.filepath = os.path.join(data_dir, filename)
        self.logger = logging.getLogger(__name__)
        self._ensure_dir()

    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time'])
            df.to_csv(self.filepath, index=False)

    def save_candle(self, candle_data):
        try:
            df = pd.DataFrame([candle_data])
            df.to_csv(self.filepath, mode='a', header=False, index=False)
            self.logger.info(f"Saved candle to {self.filepath}")
        except Exception as e:
            self.logger.error(f"Error saving candle: {e}")

    def save_macro(self, macro_data):
        macro_path = os.path.join(os.path.dirname(self.filepath), "macro.csv")
        try:
            if not os.path.exists(macro_path):
                df = pd.DataFrame(columns=['timestamp', 'fed_funds_rate', 'gold_price', 'dxy', 'sp500', 'vix', 'nasdaq', 'oil_price'])
                df.to_csv(macro_path, index=False)
            
            df = pd.DataFrame([macro_data])
            df.to_csv(macro_path, mode='a', header=False, index=False)
            self.logger.info(f"Saved macro data to {macro_path}")
        except Exception as e:
            self.logger.error(f"Error saving macro data: {e}")
