from sqlalchemy import Column, Integer, Float, String, DateTime, Index
from .database import Base

class OHLCV(Base):
    __tablename__ = "ohlcv"

    timestamp = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    
    __table_args__ = (
        Index('idx_ohlcv_symbol_time', 'symbol', 'timestamp'),
    )

class MacroIndicator(Base):
    __tablename__ = "macro_indicators"

    timestamp = Column(DateTime(timezone=True), primary_key=True)
    name = Column(String, primary_key=True)
    value = Column(Float)

    __table_args__ = (
        Index('idx_macro_name_time', 'name', 'timestamp'),
    )
