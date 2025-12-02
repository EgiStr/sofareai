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

    def fetch_historical_macro(self, lookback_days=365):
        """
        Fetch historical macro data for the past N days.
        
        Args:
            lookback_days: Number of days to look back (default: 365 for 1 year)
        
        Returns:
            List of macro data dictionaries with timestamps
        """
        self.logger.info(f"Fetching historical macro data for the past {lookback_days} days...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Initialize dataframes dictionary
        all_data = {}
        
        # 1. FRED Data (Fed Funds Rate / 10-Year Treasury)
        if self.fred:
            try:
                fed_series = self.fred.get_series('DGS10', start_date, end_date)
                if not fed_series.empty:
                    all_data['fed_funds_rate'] = fed_series
                    self.logger.info(f"Fetched {len(fed_series)} FRED data points")
            except Exception as e:
                self.logger.error(f"Error fetching historical FRED data: {e}")
        
        # 2. Yahoo Finance Historical Data
        try:
            # Download historical data for all tickers at once
            tickers_str = "GC=F DX-Y.NYB ^GSPC ^VIX ^IXIC CL=F"
            self.logger.info(f"Downloading Yahoo Finance data from {start_date.date()} to {end_date.date()}...")
            
            yf_data = yf.download(
                tickers_str,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False
            )
            
            if not yf_data.empty:
                # Extract Close prices for each ticker
                close_data = yf_data['Close'] if 'Close' in yf_data.columns.get_level_values(0) else yf_data
                
                ticker_mapping = {
                    'GC=F': 'gold_price',
                    'DX-Y.NYB': 'dxy',
                    '^GSPC': 'sp500',
                    '^VIX': 'vix',
                    '^IXIC': 'nasdaq',
                    'CL=F': 'oil_price'
                }
                
                for ticker, col_name in ticker_mapping.items():
                    if ticker in close_data.columns:
                        all_data[col_name] = close_data[ticker]
                
                self.logger.info(f"Fetched {len(close_data)} days of Yahoo Finance data")
                
        except Exception as e:
            self.logger.error(f"Error fetching historical Yahoo Finance data: {e}")
        
        # 3. Merge all data into a single DataFrame
        if not all_data:
            self.logger.warning("No historical macro data fetched")
            return []
        
        # Create a combined DataFrame
        combined_df = pd.DataFrame(all_data)
        combined_df.index = pd.to_datetime(combined_df.index)
        combined_df = combined_df.sort_index()
        
        # Forward fill missing values (markets closed on weekends, etc.)
        combined_df = combined_df.ffill()
        
        # Convert to list of dictionaries - EXPAND TO MINUTE LEVEL
        macro_records = []
        try:
            for timestamp, row in combined_df.iterrows():
                # For each day, create 1440 minute entries (24 hours * 60 minutes)
                day_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                for minute_offset in range(1440):  # 24 * 60 = 1440 minutes per day
                    minute_timestamp = day_start + timedelta(minutes=minute_offset)
                    record = {
                        'timestamp': minute_timestamp,
                        'fed_funds_rate': row.get('fed_funds_rate'),
                        'gold_price': row.get('gold_price'),
                        'dxy': row.get('dxy'),
                        'sp500': row.get('sp500'),
                        'vix': row.get('vix', 20.0),  # Default VIX
                        'nasdaq': row.get('nasdaq'),
                        'oil_price': row.get('oil_price')
                    }
                    macro_records.append(record)
            
            self.logger.info(f"Prepared {len(macro_records)} historical macro records (expanded to minute level)")
        except Exception as e:
            self.logger.error(f"Error expanding macro data to minute level: {e}")
            return []
        return macro_records

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
        
        # Expand hourly data to minute level (60 minutes per hour)
        current_hour = data['timestamp'].replace(minute=0, second=0, microsecond=0)
        minute_records = []
        for minute_offset in range(60):  # 60 minutes per hour
            minute_timestamp = current_hour + timedelta(minutes=minute_offset)
            minute_record = data.copy()
            minute_record['timestamp'] = minute_timestamp
            minute_records.append(minute_record)
        
        # Save all minute records
        for record in minute_records:
            self.storage.save_macro(record)

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
