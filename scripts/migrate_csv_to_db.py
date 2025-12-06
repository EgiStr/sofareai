import pandas as pd
import os
import sys
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert

# Add packages to path
sys.path.append(os.path.join(os.getcwd(), "packages"))
from sofare_common import SessionLocal, OHLCV, MacroIndicator

def migrate():
    db = SessionLocal()
    try:
        # Migrate OHLCV
        if os.path.exists("data/ohlcv.csv"):
            print("Migrating OHLCV...")
            df = pd.read_csv("data/ohlcv.csv")
            records = []
            for _, row in df.iterrows():
                records.append({
                    "timestamp": datetime.fromtimestamp(row['timestamp'] / 1000.0),
                    "symbol": "BTCUSDT",
                    "open": row['open'],
                    "high": row['high'],
                    "low": row['low'],
                    "close": row['close'],
                    "volume": row['volume']
                })
            
            # Batch insert
            chunk_size = 10000
            for i in range(0, len(records), chunk_size):
                chunk = records[i:i+chunk_size]
                stmt = insert(OHLCV).values(chunk)
                stmt = stmt.on_conflict_do_nothing()
                db.execute(stmt)
                db.commit()
                print(f"Migrated {i+len(chunk)}/{len(records)} OHLCV records")

        # Migrate Macro
        if os.path.exists("data/macro.csv"):
            print("Migrating Macro...")
            df = pd.read_csv("data/macro.csv")
            records = []
            for _, row in df.iterrows():
                # Try to parse timestamp
                try:
                    ts = pd.to_datetime(row['timestamp'])
                except:
                    continue
                    
                for col in df.columns:
                    if col == 'timestamp': continue
                    if pd.isna(row[col]): continue
                    
                    records.append({
                        "timestamp": ts,
                        "name": col,
                        "value": row[col]
                    })
            
            # Batch insert
            chunk_size = 10000
            for i in range(0, len(records), chunk_size):
                chunk = records[i:i+chunk_size]
                stmt = insert(MacroIndicator).values(chunk)
                stmt = stmt.on_conflict_do_nothing()
                db.execute(stmt)
                db.commit()
                print(f"Migrated {i+len(chunk)}/{len(records)} Macro records")
                
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    migrate()
