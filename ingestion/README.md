# SOFARE-AI Data Ingestion

This service provides cryptocurrency price data ingestion with multiple data source options.

## Data Sources

### Binance WebSocket (Default)
- **Source**: Binance WebSocket API
- **Interval**: 1-minute candles
- **Symbol**: BTCUSDT
- **Pros**: Real-time, high-frequency data
- **Cons**: Requires stable WebSocket connection

### CoinGecko API (Alternative)
- **Source**: CoinGecko REST API
- **Interval**: 1-minute updates (approximated)
- **Symbol**: Bitcoin
- **Pros**: Reliable, no WebSocket issues, free tier available
- **Cons**: Rate limited, less real-time

## Usage

### Using Binance (Default)
```bash
docker compose up ingestion
```

### Using CoinGecko
```bash
DATA_SOURCE=coingecko docker compose up ingestion
```

Or set environment variable:
```bash
export DATA_SOURCE=coingecko
docker compose up ingestion
```

## Data Format

Both sources provide OHLCV data in the following format:
```csv
timestamp,open,high,low,close,volume,close_time
1763972497000,86868.58,86868.58,86868.58,86868.58,0.00736,1763972497999
```

- `timestamp`: Unix timestamp in milliseconds
- `open/high/low/close`: OHLC prices
- `volume`: Trading volume
- `close_time`: Candle close timestamp

## Switching Data Sources

1. Stop the current ingestion service:
```bash
docker compose down ingestion
```

2. Set the data source:
```bash
export DATA_SOURCE=coingecko  # or binance
```

3. Restart the service:
```bash
docker compose up ingestion
```

## Troubleshooting

### Binance Connection Issues
- The service automatically switches between `stream.binance.com` and `stream.binance.us`
- Uses exponential backoff for reconnection (5s to 5min)
- Check network connectivity and firewall settings

### CoinGecko Rate Limits
- Free tier: 10-50 requests/minute
- Consider upgrading to paid tier for higher limits
- The service updates every 60 seconds by default

### Data Quality
- Both sources provide reliable price data
- CoinGecko may have slight delays compared to Binance
- Verify data integrity by checking recent prices against known values