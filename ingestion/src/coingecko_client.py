import requests
import time
import logging
import threading
from datetime import datetime, timedelta
import pandas as pd

class CoinGeckoClient:
    def __init__(self, symbol="bitcoin", vs_currency="usd", callback=None, update_interval=60):
        self.symbol = symbol
        self.vs_currency = vs_currency
        self.callback = callback
        self.update_interval = update_interval  # seconds
        self.base_url = "https://api.coingecko.com/api/v3"
        self.logger = logging.getLogger(__name__)
        self.last_timestamp = None

    def get_current_price(self):
        """Get current price data"""
        try:
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': self.symbol,
                'vs_currencies': self.vs_currency,
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true'
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if self.symbol in data:
                coin_data = data[self.symbol]
                current_time = datetime.now()

                # Create OHLCV-like data from current price
                price = coin_data.get(f'{self.vs_currency}', 0)
                volume_24h = coin_data.get(f'{self.vs_currency}_24h_vol', 0)

                # For OHLCV, we use the same price for O,H,L,C since we don't have intraday data
                processed_data = {
                    'timestamp': int(current_time.timestamp() * 1000),  # milliseconds
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume_24h / 24,  # Rough estimate of hourly volume
                    'close_time': int(current_time.timestamp() * 1000)
                }

                self.logger.info(f"CoinGecko price update: {processed_data}")
                if self.callback:
                    self.callback(processed_data)

                return processed_data
        except Exception as e:
            self.logger.error(f"Error fetching CoinGecko price: {e}")
        return None

    def get_historical_data(self, days=1):
        """Get historical OHLCV data"""
        try:
            url = f"{self.base_url}/coins/{self.symbol}/market_chart"
            params = {
                'vs_currency': self.vs_currency,
                'days': days,
                'interval': 'hourly' if days <= 90 else 'daily'
            }
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            candles = []
            if 'prices' in data:
                prices = data['prices']
                volumes = data.get('total_volumes', [])

                for i, (timestamp, price) in enumerate(prices):
                    volume = volumes[i][1] if i < len(volumes) else 0

                    candle = {
                        'timestamp': int(timestamp),  # already in milliseconds
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': volume,
                        'close_time': int(timestamp)
                    }
                    candles.append(candle)

            return candles
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return []

    def start_realtime_updates(self):
        """Start periodic price updates"""
        def update_loop():
            while True:
                try:
                    self.get_current_price()
                    time.sleep(self.update_interval)
                except Exception as e:
                    self.logger.error(f"Error in update loop: {e}")
                    time.sleep(30)  # Wait before retrying

        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
        self.logger.info(f"Started CoinGecko price updates every {self.update_interval} seconds")

    def start(self):
        """Start the client"""
        self.start_realtime_updates()