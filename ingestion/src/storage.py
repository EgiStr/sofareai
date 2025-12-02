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

    def get_last_timestamp(self):
        """Get the last timestamp from the CSV file, returns None if empty."""
        try:
            if os.path.exists(self.filepath):
                df = pd.read_csv(self.filepath)
                if len(df) > 0:
                    return df['timestamp'].max()
        except Exception as e:
            self.logger.error(f"Error reading last timestamp: {e}")
        return None

    def save_candles_bulk(self, candles):
        """
        Save multiple candles at once, avoiding duplicates based on timestamp.
        
        Args:
            candles: List of candle dictionaries
        """
        if not candles:
            return
        
        try:
            new_df = pd.DataFrame(candles)
            
            # Load existing data
            if os.path.exists(self.filepath):
                existing_df = pd.read_csv(self.filepath)
                if len(existing_df) > 0:
                    # Filter out duplicates based on timestamp
                    existing_timestamps = set(existing_df['timestamp'].values)
                    new_df = new_df[~new_df['timestamp'].isin(existing_timestamps)]
            
            if len(new_df) > 0:
                new_df.to_csv(self.filepath, mode='a', header=False, index=False)
                self.logger.info(f"Saved {len(new_df)} historical candles to {self.filepath}")
            else:
                self.logger.info("No new candles to save (all duplicates)")
                
        except Exception as e:
            self.logger.error(f"Error saving bulk candles: {e}")

    def save_candle(self, candle_data):
        try:
            df = pd.DataFrame([candle_data])
            df.to_csv(self.filepath, mode='a', header=False, index=False)
            self.logger.info(f"Saved candle to {self.filepath}")
        except Exception as e:
            self.logger.error(f"Error saving candle: {e}")

    def get_last_macro_timestamp(self):
        """Get the last timestamp from the macro CSV file, returns None if empty."""
        macro_path = os.path.join(os.path.dirname(self.filepath), "macro.csv")
        try:
            if os.path.exists(macro_path):
                df = pd.read_csv(macro_path)
                if len(df) > 0:
                    return pd.to_datetime(df['timestamp']).max()
        except Exception as e:
            self.logger.error(f"Error reading last macro timestamp: {e}")
        return None

    def save_macro_bulk(self, macro_records):
        """
        Save multiple macro records at once, avoiding duplicates based on timestamp.
        
        Args:
            macro_records: List of macro data dictionaries
        """
        if not macro_records:
            return
        
        macro_path = os.path.join(os.path.dirname(self.filepath), "macro.csv")
        
        try:
            # Ensure file exists with headers
            if not os.path.exists(macro_path):
                df = pd.DataFrame(columns=['timestamp', 'fed_funds_rate', 'gold_price', 'dxy', 'sp500', 'vix', 'nasdaq', 'oil_price'])
                df.to_csv(macro_path, index=False)
            
            new_df = pd.DataFrame(macro_records)
            # Keep full timestamp for minute-level data (don't convert to date)
            
            # Load existing data
            existing_df = pd.read_csv(macro_path)
            if len(existing_df) > 0:
                # Filter out duplicates based on full timestamp
                existing_timestamps = set(pd.to_datetime(existing_df['timestamp']).values)
                new_timestamps = pd.to_datetime(new_df['timestamp']).values
                new_df = new_df[~pd.Series(new_timestamps).isin(existing_timestamps)]
            
            if len(new_df) > 0:
                new_df.to_csv(macro_path, mode='a', header=False, index=False)
                self.logger.info(f"Saved {len(new_df)} historical macro records to {macro_path}")
            else:
                self.logger.info("No new macro records to save (all duplicates)")
                
        except Exception as e:
            self.logger.error(f"Error saving bulk macro data: {e}")

    def save_macro(self, macro_data):
        macro_path = os.path.join(os.path.dirname(self.filepath), "macro.csv")
        try:
            if not os.path.exists(macro_path):
                df = pd.DataFrame(columns=['timestamp', 'fed_funds_rate', 'gold_price', 'dxy', 'sp500', 'vix', 'nasdaq', 'oil_price'])
                df.to_csv(macro_path, index=False)
            
            df = pd.DataFrame([macro_data])
            # Keep full timestamp for minute-level data (don't convert to date)
            df.to_csv(macro_path, mode='a', header=False, index=False)
            self.logger.info(f"Saved macro data to {macro_path}")
        except Exception as e:
            self.logger.error(f"Error saving macro data: {e}")
