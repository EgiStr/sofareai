import json
import websocket
import threading
import time
import logging
from binance.client import Client
import ccxt
import pandas as pd

class BinanceClient:
    def __init__(self, symbol="btcusdt", interval="1m", callback=None):
        self.symbol = symbol
        self.interval = interval
        self.callback = callback
        self.base_urls = [
            "wss://stream.binance.com:9443",
            "wss://stream.binance.us:9443"
        ]
        self.ws = None
        self.logger = logging.getLogger(__name__)
        # REST client for historical data (no API key needed for public endpoints)
        # Initialize client without ping to avoid SSL/network issues
        try:
            self.rest_client = Client(requests_params={"verify": False})
            # Skip ping to avoid network issues in container environment
            self.rest_client.ping = lambda: None  # Override ping method
        except Exception as e:
            self.logger.warning(f"Failed to initialize Binance REST client: {e}")
            self.rest_client = None

    def fetch_historical_klines(self, lookback="1 year ago UTC"):
        """
        Fetch historical 1-minute klines from Binance.
        
        Args:
            lookback: Start time string (e.g., "1 year ago UTC", "1 Dec, 2024")
        
        Returns:
            List of processed candle dictionaries
        """
        if self.rest_client is None:
            self.logger.error("Binance REST client not available")
            return []
            
        self.logger.info(f"Fetching historical klines for {self.symbol.upper()} from {lookback}...")
        
        try:
            # get_historical_klines handles pagination automatically
            klines = self.rest_client.get_historical_klines(
                symbol=self.symbol.upper(),
                interval=Client.KLINE_INTERVAL_1MINUTE,
                start_str=lookback
            )
            
            self.logger.info(f"Fetched {len(klines)} historical klines")
            
            processed_candles = []
            for kline in klines:
                processed_data = {
                    'timestamp': kline[0],       # Open time
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': kline[6]       # Close time
                }
                processed_candles.append(processed_data)
            
            return processed_candles
            
        except Exception as e:
            self.logger.error(f"Error fetching historical klines: {e}")
            return []

    def fetch_historical_klines_ccxt(self, lookback="1 year ago UTC"):
        """
        Fetch historical 1-minute klines from multiple exchanges using CCXT library.
        Tries different exchanges in order until one works.
        
        Args:
            lookback: Start time string (e.g., "1 year ago UTC", "1 Dec, 2024")
        
        Returns:
            List of processed candle dictionaries
        """
        self.logger.info(f"Fetching historical 1-minute klines using CCXT for {self.symbol.upper()} from {lookback}...")
        
        # List of exchanges to try in order
        exchanges_to_try = [
            ('binance', 'BTC/USDT'),
            ('kucoin', 'BTC/USDT'),
            ('okx', 'BTC/USDT'),
            ('huobi', 'BTC/USDT'),
            ('bybit', 'BTCUSDT'),  # Bybit uses different symbol format
        ]
        
        for exchange_name, symbol in exchanges_to_try:
            try:
                self.logger.info(f"Trying exchange: {exchange_name} with symbol: {symbol}")
                
                # Initialize CCXT exchange
                exchange_class = getattr(ccxt, exchange_name)
                exchange = exchange_class()
                
                # Parse lookback time
                since = exchange.parse8601(lookback.replace(" ago UTC", "").replace("1 year", "2023-12-01T00:00:00Z"))
                
                # Fetch OHLCV data with pagination
                all_ohlcvs = []
                timeframe = '1m'
                limit = 1000  # Max per request
                
                while True:
                    ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                    
                    if not ohlcvs:
                        break
                        
                    all_ohlcvs.extend(ohlcvs)
                    
                    # Update since for next page
                    since = ohlcvs[-1][0] + 1
                    
                    # Break if we got less than limit (last page)
                    if len(ohlcvs) < limit:
                        break
                
                if all_ohlcvs:
                    self.logger.info(f"Successfully fetched {len(all_ohlcvs)} historical 1-minute klines using {exchange_name}")
                    
                    # Process candles to match existing format
                    processed_candles = []
                    for ohlcv in all_ohlcvs:
                        processed_data = {
                            'timestamp': ohlcv[0],        # Timestamp
                            'open': float(ohlcv[1]),      # Open
                            'high': float(ohlcv[2]),      # High
                            'low': float(ohlcv[3]),       # Low
                            'close': float(ohlcv[4]),     # Close
                            'volume': float(ohlcv[5]),    # Volume
                            'close_time': ohlcv[0] + 60000  # Close time (1 minute later)
                        }
                        processed_candles.append(processed_data)
                    
                    return processed_candles
                else:
                    self.logger.warning(f"{exchange_name} returned no data")
                    
            except Exception as e:
                self.logger.warning(f"Failed to fetch from {exchange_name}: {e}")
                continue
        
        self.logger.error("All exchanges failed to provide historical data")
        return []

    def fetch_historical_klines_yfinance(self, lookback="10 years ago UTC"):
        """
        Fetch historical data from Yahoo Finance with enhanced minute data coverage.
        Gets minute data for extended periods using chunked downloads and multiple intervals.

        Args:
            lookback: Start time string (e.g., "10 years ago UTC", "5 years ago UTC")

        Returns:
            List of processed candle dictionaries
        """
        self.logger.info(f"Fetching historical data using Yahoo Finance for {self.symbol.upper()} from {lookback}...")

        try:
            import yfinance as yf
            from datetime import datetime, timedelta

            # Convert symbol to Yahoo Finance format (btcusdt -> BTC-USD)
            yf_symbol = f"{self.symbol[:-4].upper()}-USD"

            ticker = yf.Ticker(yf_symbol)
            end_date = datetime.now()

            all_candles = []

            # Strategy 1: Try to get as much minute data as possible by chunking
            self.logger.info("Attempting to get extended minute data using chunked downloads...")

            # Try different periods for minute data (Yahoo Finance limits to ~7 days per request)
            minute_periods = [
                (end_date - timedelta(days=7), end_date, "1m"),      # Last 7 days - 1m
                (end_date - timedelta(days=14), end_date - timedelta(days=7), "5m"),  # 7-14 days ago - 5m
                (end_date - timedelta(days=30), end_date - timedelta(days=14), "15m"), # 14-30 days ago - 15m
                (end_date - timedelta(days=60), end_date - timedelta(days=30), "1h"),  # 30-60 days ago - 1h
            ]

            for start_date, end_date_chunk, interval in minute_periods:
                try:
                    self.logger.info(f"Fetching {interval} data from {start_date.date()} to {end_date_chunk.date()}")

                    df_chunk = ticker.history(start=start_date, end=end_date_chunk, interval=interval)

                    if len(df_chunk) > 0:
                        self.logger.info(f"Fetched {len(df_chunk)} {interval} candles")

                        # Convert higher timeframes to minute data by forward filling
                        for index, row in df_chunk.iterrows():
                            timestamp_ms = int(index.timestamp() * 1000)

                            # Calculate interval in minutes
                            if interval == "1m":
                                interval_minutes = 1
                            elif interval == "5m":
                                interval_minutes = 5
                            elif interval == "15m":
                                interval_minutes = 15
                            elif interval == "1h":
                                interval_minutes = 60
                            else:
                                interval_minutes = 1

                            # Create multiple minute candles for higher timeframes
                            for minute_offset in range(interval_minutes):
                                minute_timestamp = timestamp_ms + (minute_offset * 60000)  # 60 seconds * 1000ms

                                processed_data = {
                                    'timestamp': minute_timestamp,
                                    'open': float(row['Open']),
                                    'high': float(row['High']),
                                    'low': float(row['Low']),
                                    'close': float(row['Close']),
                                    'volume': float(row['Volume']) / interval_minutes,  # Distribute volume
                                    'close_time': minute_timestamp + 60000
                                }
                                all_candles.append(processed_data)

                    else:
                        self.logger.warning(f"No {interval} data available for period {start_date.date()} to {end_date_chunk.date()}")

                except Exception as e:
                    self.logger.warning(f"Failed to fetch {interval} data for period: {e}")
                    continue

            # Strategy 2: Fill remaining gaps with daily data converted to minutes
            oldest_minute_date = min([c['timestamp'] for c in all_candles]) if all_candles else end_date
            oldest_minute_datetime = datetime.fromtimestamp(oldest_minute_date / 1000)

            # Get daily data for the remaining historical period
            historical_start = end_date - timedelta(days=365*10)  # 10 years back

            if oldest_minute_datetime > historical_start:
                self.logger.info(f"Fetching daily data to fill gaps from {historical_start.date()} to {oldest_minute_datetime.date()}")

                df_daily = ticker.history(start=historical_start, end=oldest_minute_datetime, interval="1d")

                self.logger.info(f"Fetched {len(df_daily)} daily candles for historical gap filling")

                for index, row in df_daily.iterrows():
                    timestamp_ms = int(index.timestamp() * 1000)

                    # Create 1440 minute candles per day (24 hours * 60 minutes)
                    day_start = datetime.fromtimestamp(timestamp_ms / 1000).replace(hour=0, minute=0, second=0, microsecond=0)

                    for minute_offset in range(1440):  # 24 * 60 = 1440 minutes per day
                        minute_timestamp = int((day_start + timedelta(minutes=minute_offset)).timestamp() * 1000)

                        processed_data = {
                            'timestamp': minute_timestamp,
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close']),
                            'volume': float(row['Volume']) / 1440,  # Distribute volume across minutes
                            'close_time': minute_timestamp + 60000
                        }
                        all_candles.append(processed_data)

            # Sort all candles by timestamp
            all_candles.sort(key=lambda x: x['timestamp'])

            # Remove duplicates based on timestamp
            seen_timestamps = set()
            unique_candles = []
            for candle in all_candles:
                if candle['timestamp'] not in seen_timestamps:
                    unique_candles.append(candle)
                    seen_timestamps.add(candle['timestamp'])

            self.logger.info(f"Total unique minute-level data: {len(unique_candles)} candles")

            return unique_candles

        except Exception as e:
            self.logger.error(f"Error fetching historical data with Yahoo Finance: {e}")
            return []

    def fetch_historical_klines_comprehensive(self, lookback="10 years ago UTC"):
        """
        Comprehensive historical data fetching with maximum minute granularity.
        Tries multiple sources and strategies to get 10 years of minute-level data.

        Args:
            lookback: Start time string (e.g., "10 years ago UTC")

        Returns:
            List of processed candle dictionaries with maximum minute coverage
        """
        self.logger.info(f"Starting comprehensive historical data fetch for {self.symbol.upper()} from {lookback}...")

        all_candles = []

        # Strategy 1: Try Binance REST API (limited but high quality)
        self.logger.info("Strategy 1: Trying Binance REST API...")
        binance_candles = self.fetch_historical_klines(lookback)
        if binance_candles:
            all_candles.extend(binance_candles)
            self.logger.info(f"Added {len(binance_candles)} candles from Binance REST")

        # Strategy 2: Try CCXT exchanges for additional data
        self.logger.info("Strategy 2: Trying CCXT exchanges...")
        ccxt_candles = self.fetch_historical_klines_ccxt(lookback)
        if ccxt_candles:
            # Filter out duplicates with existing data
            existing_timestamps = set(c['timestamp'] for c in all_candles)
            new_ccxt_candles = [c for c in ccxt_candles if c['timestamp'] not in existing_timestamps]
            all_candles.extend(new_ccxt_candles)
            self.logger.info(f"Added {len(new_ccxt_candles)} new candles from CCXT")

        # Strategy 3: Enhanced Yahoo Finance with chunked downloads
        self.logger.info("Strategy 3: Enhanced Yahoo Finance chunked download...")
        yfinance_candles = self.fetch_historical_klines_yfinance(lookback)
        if yfinance_candles:
            # Filter out duplicates
            existing_timestamps = set(c['timestamp'] for c in all_candles)
            new_yf_candles = [c for c in yfinance_candles if c['timestamp'] not in existing_timestamps]
            all_candles.extend(new_yf_candles)
            self.logger.info(f"Added {len(new_yf_candles)} new candles from Yahoo Finance")

        # Strategy 4: Alpha Vantage for extended intraday data
        self.logger.info("Strategy 4: Alpha Vantage intraday data...")
        av_candles = self.fetch_historical_klines_alpha_vantage(lookback)
        if av_candles:
            # Filter out duplicates
            existing_timestamps = set(c['timestamp'] for c in all_candles)
            new_av_candles = [c for c in av_candles if c['timestamp'] not in existing_timestamps]
            all_candles.extend(new_av_candles)
            self.logger.info(f"Added {len(new_av_candles)} new candles from Alpha Vantage")

        # Sort and deduplicate final dataset
        all_candles.sort(key=lambda x: x['timestamp'])

        # Remove duplicates (keep first occurrence)
        seen_timestamps = set()
        unique_candles = []
        for candle in all_candles:
            if candle['timestamp'] not in seen_timestamps:
                unique_candles.append(candle)
                seen_timestamps.add(candle['timestamp'])

        self.logger.info(f"Comprehensive fetch complete: {len(unique_candles)} unique minute-level candles")
        return unique_candles

    def fetch_historical_klines_yfinance_daily(self, lookback="1 year ago UTC"):
        """
        Fetch historical daily klines from Yahoo Finance as final fallback.
        
        Args:
            lookback: Start time string (e.g., "1 year ago UTC", "1 Dec, 2024")
        
        Returns:
            List of processed candle dictionaries (daily data)
        """
        self.logger.info(f"Fetching historical daily klines using Yahoo Finance for {self.symbol.upper()} from {lookback}...")
        
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            
            # Convert symbol to Yahoo Finance format (btcusdt -> BTC-USD)
            yf_symbol = f"{self.symbol[:-4].upper()}-USD"
            
            # Parse lookback time
            if "ago UTC" in lookback:
                days_back = 365 if "1 year" in lookback else 30  # Default to 1 year
                start_date = datetime.now() - timedelta(days=days_back)
            else:
                # Try to parse date string
                start_date = datetime.strptime(lookback.replace(" ago UTC", "").strip(), "%d %b, %Y")
            
            # Download data from Yahoo Finance
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, interval="1d")
            
            self.logger.info(f"Fetched {len(df)} historical daily klines using Yahoo Finance")
            
            # Process candles to match existing format
            processed_candles = []
            for index, row in df.iterrows():
                # Convert to millisecond timestamp
                timestamp_ms = int(index.timestamp() * 1000)
                
                processed_data = {
                    'timestamp': timestamp_ms,           # Timestamp
                    'open': float(row['Open']),          # Open
                    'high': float(row['High']),          # High
                    'low': float(row['Low']),            # Low
                    'close': float(row['Close']),        # Close
                    'volume': float(row['Volume']),      # Volume
                    'close_time': timestamp_ms + 86400000  # Close time (1 day later in ms)
                }
                processed_candles.append(processed_data)
            
            return processed_candles
            
        except Exception as e:
            self.logger.error(f"Error fetching historical klines with Yahoo Finance: {e}")
            return []

    def fetch_historical_klines_alpha_vantage(self, lookback="1 year ago UTC"):
        """
        Fetch historical intraday klines from Alpha Vantage as additional data source.
        
        Args:
            lookback: Start time string (e.g., "1 year ago UTC", "1 Dec, 2024")
        
        Returns:
            List of processed candle dictionaries (1-minute data)
        """
        self.logger.info(f"Fetching historical intraday klines using Alpha Vantage for {self.symbol.upper()} from {lookback}...")
        
        try:
            from alpha_vantage.timeseries import TimeSeries
            from datetime import datetime, timedelta
            import os
            
            # Get API key from environment
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if not api_key:
                self.logger.warning("ALPHA_VANTAGE_API_KEY not found, skipping Alpha Vantage")
                return []
            
            # Initialize Alpha Vantage client
            ts = TimeSeries(key=api_key, output_format='pandas')
            
            # Convert symbol to Alpha Vantage format (btcusdt -> BTC)
            av_symbol = self.symbol[:-4].upper()
            
            # Parse lookback time
            if "ago UTC" in lookback:
                if "10 year" in lookback:
                    years_back = 10
                elif "5 year" in lookback:
                    years_back = 5
                elif "2 year" in lookback:
                    years_back = 2
                elif "1 year" in lookback:
                    years_back = 1
                else:
                    years_back = 1
                start_date = datetime.now() - timedelta(days=years_back*365)
            else:
                # Try to parse date string
                try:
                    start_date = datetime.strptime(lookback.replace(" ago UTC", "").strip(), "%d %b, %Y")
                except:
                    start_date = datetime.now() - timedelta(days=365)
            
            # Fetch intraday data (1-minute intervals)
            self.logger.info(f"Fetching 1-minute intraday data for {av_symbol} from {start_date.date()}")
            data, meta_data = ts.get_intraday(symbol=av_symbol, interval='1min', outputsize='full')
            
            if data.empty:
                self.logger.warning("No data returned from Alpha Vantage")
                return []
            
            # Filter data by start date
            data.index = pd.to_datetime(data.index)
            filtered_data = data[data.index >= start_date]
            
            self.logger.info(f"Fetched {len(filtered_data)} historical 1-minute klines using Alpha Vantage")
            
            # Process candles to match existing format
            processed_candles = []
            for index, row in filtered_data.iterrows():
                # Convert to millisecond timestamp
                timestamp_ms = int(index.timestamp() * 1000)
                
                processed_data = {
                    'timestamp': timestamp_ms,           # Timestamp
                    'open': float(row['1. open']),        # Open
                    'high': float(row['2. high']),        # High
                    'low': float(row['3. low']),          # Low
                    'close': float(row['4. close']),      # Close
                    'volume': float(row['5. volume']),    # Volume
                    'close_time': timestamp_ms + 60000    # Close time (1 minute later in ms)
                }
                processed_candles.append(processed_data)
            
            return processed_candles
            
        except Exception as e:
            self.logger.error(f"Error fetching historical klines with Alpha Vantage: {e}")
            return []

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if 'k' in data:
                kline = data['k']
                if kline['x']:  # Only process closed candles
                    processed_data = {
                        'timestamp': kline['t'],
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),
                        'close_time': kline['T']
                    }
                    self.logger.info(f"Received candle: {processed_data['close']} at {processed_data['timestamp']}")
                    if self.callback:
                        self.callback(processed_data)
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def on_error(self, ws, error):
        self.logger.error(f"WebSocket error: {error}")
        # Don't reconnect here, let the main loop handle it

    def on_close(self, ws, close_status_code, close_msg):
        self.logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")

    def on_open(self, ws):
        self.logger.info("WebSocket connection opened successfully")

    def start(self):
        def run_ws():
            url_index = 0
            reconnect_delay = 5  # Start with 5 seconds
            
            while True:
                try:
                    base_url = self.base_urls[url_index]
                    ws_url = f"{base_url}/ws/{self.symbol}@kline_{self.interval}"
                    self.logger.info(f"Connecting to {ws_url}...")
                    
                    self.ws = websocket.WebSocketApp(
                        ws_url,
                        on_open=self.on_open,
                        on_message=self.on_message,
                        on_error=self.on_error,
                        on_close=self.on_close
                    )
                    
                    # Reset delay on successful connection
                    reconnect_delay = 5
                    
                    self.ws.run_forever()
                    
                except Exception as e:
                    self.logger.error(f"WebSocket run error: {e}")
                
                # Switch to next URL on failure/close
                url_index = (url_index + 1) % len(self.base_urls)
                
                # Exponential backoff with max delay of 5 minutes
                self.logger.info(f"Connection failed. Retrying in {reconnect_delay} seconds...")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 300)  # Max 5 minutes

        wst = threading.Thread(target=run_ws)
        wst.daemon = True
        wst.start()
