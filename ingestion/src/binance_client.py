import json
import websocket
import threading
import time
import logging

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
                    self.logger.info(f"Received candle: {processed_data}")
                    if self.callback:
                        self.callback(processed_data)
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def on_error(self, ws, error):
        self.logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.logger.info("WebSocket closed")

    def on_open(self, ws):
        self.logger.info("WebSocket connection opened")

    def start(self):
        def run_ws():
            url_index = 0
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
                    self.ws.run_forever()
                except Exception as e:
                    self.logger.error(f"WebSocket run error: {e}")
                
                # Switch to next URL on failure/close
                url_index = (url_index + 1) % len(self.base_urls)
                self.logger.info(f"Connection failed. Switching to next endpoint in 5 seconds...")
                time.sleep(5)

        wst = threading.Thread(target=run_ws)
        wst.daemon = True
        wst.start()
