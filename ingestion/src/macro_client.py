import time
import logging
import pandas as pd
from fredapi import Fred
import yfinance as yf
import threading
import os
from datetime import datetime, timedelta

class MacroClient:
    def __init__(self, storage, interval_seconds=3600):
        self.storage = storage
        self.interval = interval_seconds
        self.logger = logging.getLogger(__name__)
        self.fred_api_key = os.getenv("FRED_API_KEY", "test_key") # Fallback for dev
        self.running = False
        
        try:
            if self.fred_api_key and len(self.fred_api_key) == 32:
                self.fred = Fred(api_key=self.fred_api_key)
            else:
                self.logger.warning(f"Invalid FRED API Key length ({len(self.fred_api_key) if self.fred_api_key else 0}). Expected 32 chars. FRED data disabled.")
                self.fred = None
        except Exception as e:
            self.logger.warning(f"Failed to init FRED: {e}. Macro data might be incomplete.")
            self.fred = None

    def fetch_data(self):
        self.logger.info("Fetching macro data...")
        data = {
            'timestamp': datetime.now(),
            'fed_funds_rate': None,
            'gold_price': None,
            'dxy': None
        }

        # 1. FRED Data (Fed Funds Rate)
        if self.fred:
            try:
                # DGS10 is 10-Year Treasury, often used as proxy if daily fed funds not available
                # FEDFUNDS is monthly. Let's use DGS10 for daily proxy or stick to monthly forward fill.
                # Using DGS10 (Daily) for better granularity
                series = self.fred.get_series('DGS10', limit=1, sort_order='desc')
                if not series.empty:
                    data['fed_funds_rate'] = float(series.iloc[0])
            except Exception as e:
                self.logger.error(f"Error fetching FRED data: {e}")

        # 2. Yahoo Finance (Gold & DXY)
        try:
            # Gold Futures (GC=F), Dollar Index (DX-Y.NYB)
            tickers = yf.Tickers("GC=F DX-Y.NYB")
            
            # Gold
            gold = tickers.tickers['GC=F'].history(period="1d")
            if not gold.empty:
                data['gold_price'] = float(gold['Close'].iloc[-1])
            
            # DXY
            dxy = tickers.tickers['DX-Y.NYB'].history(period="1d")
            if not dxy.empty:
                data['dxy'] = float(dxy['Close'].iloc[-1])
                
        except Exception as e:
            self.logger.error(f"Error fetching Yahoo Finance data: {e}")

        self.logger.info(f"Macro data fetched: {data}")
        self.storage.save_macro(data)

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._run_loop)
        thread.daemon = True
        thread.start()

    def _run_loop(self):
        while self.running:
            try:
                self.fetch_data()
            except Exception as e:
                self.logger.error(f"Error in macro loop: {e}")
            
            # Sleep for interval
            time.sleep(self.interval)

    def stop(self):
        self.running = False
