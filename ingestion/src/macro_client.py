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
            'dxy': None,
            'sp500': None,
            'vix': None,
            'nasdaq': None,
            'oil_price': None
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

        # 2. Yahoo Finance (Gold, DXY, Indices, Oil)
        try:
            # Tickers: Gold (GC=F), DXY (DX-Y.NYB), S&P500 (^GSPC), VIX (^VIX), NASDAQ (^IXIC), Oil (CL=F)
            tickers = yf.Tickers("GC=F DX-Y.NYB ^GSPC ^VIX ^IXIC CL=F")
            
            # Gold
            gold = tickers.tickers['GC=F'].history(period="1d")
            if not gold.empty:
                data['gold_price'] = float(gold['Close'].iloc[-1])
            
            # DXY
            dxy = tickers.tickers['DX-Y.NYB'].history(period="1d")
            if not dxy.empty:
                data['dxy'] = float(dxy['Close'].iloc[-1])
            
            # S&P 500
            sp500 = tickers.tickers['^GSPC'].history(period="1d")
            if not sp500.empty:
                data['sp500'] = float(sp500['Close'].iloc[-1])
            
            # VIX
            try:
                vix = tickers.tickers['^VIX'].history(period="1d")
                if not vix.empty:
                    data['vix'] = float(vix['Close'].iloc[-1])
                else:
                    # Fallback: try different period or set default
                    vix_alt = tickers.tickers['^VIX'].history(period="5d", interval="1d")
                    if not vix_alt.empty:
                        data['vix'] = float(vix_alt['Close'].iloc[-1])
                    else:
                        data['vix'] = 20.0  # Default VIX value
                        self.logger.warning("VIX data unavailable, using default value 20.0")
            except Exception as e:
                self.logger.warning(f"VIX fetch failed: {e}, using default value 20.0")
                data['vix'] = 20.0
            
            # NASDAQ
            nasdaq = tickers.tickers['^IXIC'].history(period="1d")
            if not nasdaq.empty:
                data['nasdaq'] = float(nasdaq['Close'].iloc[-1])
            
            # Oil (WTI Crude)
            oil = tickers.tickers['CL=F'].history(period="1d")
            if not oil.empty:
                data['oil_price'] = float(oil['Close'].iloc[-1])
                
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
