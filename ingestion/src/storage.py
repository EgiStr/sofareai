import logging
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert
from sofare_common import SessionLocal, OHLCV, MacroIndicator

class DataStorage:
    def __init__(self, data_dir="/app/data", filename="ohlcv.csv"):
        self.logger = logging.getLogger(__name__)
        # Initialize DB tables if not exist (though usually done by migration or init script)
        # We can call init_db() but it's async.
        # For now assume tables exist or created by another process.
        pass

    def _get_db(self):
        return SessionLocal()

    def get_last_timestamp(self):
        """Get the last timestamp from the DB, returns None if empty."""
        db = self._get_db()
        try:
            last_record = db.query(OHLCV).filter(OHLCV.symbol == 'BTCUSDT').order_by(OHLCV.timestamp.desc()).first()
            if last_record:
                return int(last_record.timestamp.timestamp() * 1000)
        except Exception as e:
            self.logger.error(f"Error reading last timestamp: {e}")
        finally:
            db.close()
        return None

    def get_last_macro_timestamp(self):
        """Get the last macro timestamp from the DB."""
        db = self._get_db()
        try:
            last_record = db.query(MacroIndicator).order_by(MacroIndicator.timestamp.desc()).first()
            if last_record:
                return int(last_record.timestamp.timestamp() * 1000)
        except Exception as e:
            self.logger.error(f"Error reading last macro timestamp: {e}")
        finally:
            db.close()
        return None

    def save_candle(self, candle):
        self.save_candles_bulk([candle])

    def save_candles_bulk(self, candles):
        if not candles:
            return
        
        db = self._get_db()
        try:
            records = []
            for c in candles:
                ts_val = c['timestamp']
                if isinstance(ts_val, (int, float)):
                    ts = datetime.fromtimestamp(ts_val / 1000.0)
                else:
                    ts = ts_val

                records.append({
                    "timestamp": ts,
                    "symbol": "BTCUSDT",
                    "open": float(c['open']),
                    "high": float(c['high']),
                    "low": float(c['low']),
                    "close": float(c['close']),
                    "volume": float(c['volume'])
                })

            stmt = insert(OHLCV).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=['timestamp', 'symbol'],
                set_={
                    'open': stmt.excluded.open,
                    'high': stmt.excluded.high,
                    'low': stmt.excluded.low,
                    'close': stmt.excluded.close,
                    'volume': stmt.excluded.volume
                }
            )
            db.execute(stmt)
            db.commit()
        except Exception as e:
            self.logger.error(f"Error saving candles: {e}")
            db.rollback()
        finally:
            db.close()

    def save_macro_bulk(self, records):
        if not records:
            return
            
        db = self._get_db()
        try:
            db_records = []
            for r in records:
                ts_val = r['timestamp']
                if isinstance(ts_val, (int, float)):
                    ts = datetime.fromtimestamp(ts_val / 1000.0)
                else:
                    ts = ts_val

                # Pivot fields
                for key, value in r.items():
                    if key == 'timestamp':
                        continue
                    if value is None:
                        continue
                        
                    db_records.append({
                        "timestamp": ts,
                        "name": key,
                        "value": float(value)
                    })

            if not db_records:
                return

            stmt = insert(MacroIndicator).values(db_records)
            stmt = stmt.on_conflict_do_update(
                index_elements=['timestamp', 'name'],
                set_={'value': stmt.excluded.value}
            )
            db.execute(stmt)
            db.commit()
        except Exception as e:
            self.logger.error(f"Error saving macro data: {e}")
            db.rollback()
        finally:
            db.close()
